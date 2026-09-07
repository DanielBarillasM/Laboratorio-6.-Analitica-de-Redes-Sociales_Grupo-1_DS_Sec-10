"""Centralidades, participantes puente y experimentos de remoción."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .networks import _stable_graph


WEIGHT = "weight"
DISTANCE = "distance"


def add_distance(graph: nx.Graph, weight: str = WEIGHT, distance: str = DISTANCE) -> nx.Graph:
    """Copia el grafo agregando distance = 1 / weight.

    El peso mide fuerza de conexión (videos o autores compartidos): dos nodos con
    peso alto están *más cerca*, no más lejos. Los algoritmos de caminos mínimos
    suman el atributo como costo, así que necesitan su inverso.
    """

    prepared = _stable_graph(graph)
    for source, target, data in prepared.edges(data=True):
        value = float(data.get(weight, 1.0))
        if value <= 0:
            raise ValueError(f"El peso de ({source}, {target}) debe ser positivo; se recibió {value}.")
        data[distance] = 1.0 / value
    return prepared


def _component_lookup(graph: nx.Graph) -> tuple[dict[str, int], dict[str, int]]:
    components = sorted(
        (sorted(group) for group in nx.connected_components(graph)),
        key=lambda group: (-len(group), group[0] if group else ""),
    )
    identifier = {node: index + 1 for index, group in enumerate(components) for node in group}
    size = {node: len(group) for group in components for node in group}
    return identifier, size


def centrality_measures(graph: nx.Graph, red: str, weight: str = WEIGHT, distance: str = DISTANCE) -> pd.DataFrame:
    """Calcula grado, grado ponderado, intermediación, centralidad armónica y PageRank.

    Intermediación y centralidad armónica usan `distance`; grado ponderado y
    PageRank usan `weight`. Se elige la centralidad armónica en lugar de la
    cercanía clásica porque la red está fragmentada: la armónica trata las
    distancias infinitas como aporte cero y sigue siendo comparable entre
    componentes, mientras que la cercanía clásica no está definida globalmente.
    """

    columns = [
        "red", "node", "node_id", "node_type", "label", "grado", "grado_ponderado",
        "intermediacion", "armonica", "pagerank", "componente", "componente_tamano",
    ]
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    prepared = add_distance(graph, weight, distance) if graph.number_of_edges() else _stable_graph(graph)
    nodes = sorted(prepared.nodes)
    order = len(nodes)
    betweenness = nx.betweenness_centrality(prepared, weight=distance, normalized=True)
    harmonic = _harmonic_centrality(prepared, nodes, distance)
    pagerank = nx.pagerank(prepared, weight=weight) if prepared.number_of_edges() else dict.fromkeys(nodes, 1 / order)
    identifier, size = _component_lookup(prepared)

    table = pd.DataFrame([
        {
            "red": red,
            "node": node,
            "node_id": prepared.nodes[node].get("node_id", node),
            "node_type": prepared.nodes[node].get("node_type", "desconocido"),
            "label": prepared.nodes[node].get("label", node),
            "grado": prepared.degree(node),
            "grado_ponderado": float(prepared.degree(node, weight=weight)),
            "intermediacion": float(betweenness[node]),
            "armonica": float(harmonic[node] / (order - 1)) if order > 1 else 0.0,
            "pagerank": float(pagerank[node]),
            "componente": identifier[node],
            "componente_tamano": size[node],
        }
        for node in nodes
    ], columns=columns)
    return table.sort_values(["intermediacion", "grado_ponderado", "node"], ascending=[False, False, True]).reset_index(drop=True)


def _harmonic_centrality(graph: nx.Graph, nodes: list[str], distance: str) -> dict[str, float]:
    """Suma de 1/d sobre destinos alcanzables, acumulada en orden fijo de nodos.

    Se recorre la vecindad ordenada en lugar de usar la utilidad de NetworkX para
    que el resultado sea idéntico entre ejecuciones: la suma en coma flotante
    depende del orden de los sumandos.
    """

    scores: dict[str, float] = {}
    for node in nodes:
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight=distance)
        scores[node] = sum(1 / length for _, length in sorted(lengths.items()) if length > 0)
    return scores


def centrality_rationale() -> pd.DataFrame:
    """Justifica qué mide cada centralidad y con qué atributo se calcula."""

    return pd.DataFrame([
        {"medida": "grado", "atributo": "sin peso", "interpreta": "número de vecinos distintos", "justificacion": "Mide alcance directo; en la proyección autor-autor cuenta con cuántos autores se coincidió al menos una vez."},
        {"medida": "grado_ponderado", "atributo": "weight", "interpreta": "intensidad acumulada de coparticipación", "justificacion": "El peso es un conteo de videos (autores) compartidos, por lo que sumarlo mide volumen de copresencia, no distancia."},
        {"medida": "intermediacion", "atributo": "distance = 1/weight", "interpreta": "posición de paso entre pares", "justificacion": "Identifica participantes puente. Requiere costos: una coparticipación fuerte debe ser un camino barato, de ahí el inverso del peso."},
        {"medida": "armonica", "atributo": "distance = 1/weight", "interpreta": "cercanía promedio tolerante a la desconexión", "justificacion": "Se prefiere a la cercanía clásica porque la red tiene múltiples componentes y nodos aislados; la armónica suma 1/d y asigna cero a los pares inalcanzables."},
        {"medida": "pagerank", "atributo": "weight", "interpreta": "importancia por vecindad ponderada", "justificacion": "Complementa al grado al descontar la centralidad heredada de vecinos muy conectados; el peso actúa como probabilidad de transición, no como costo."},
        {"medida": "vectores_propios", "atributo": "no calculada", "interpreta": "-", "justificacion": "Se descarta: en un grafo con muchos componentes la centralidad de vectores propios se concentra en la componente dominante y asigna cero al resto, lo que la vuelve no comparable en esta red."},
    ])


def author_profiles(comments: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """Resume recurrencia y diversidad de participación por autor."""

    work = comments.dropna(subset=["author_channel_id", "video_id"]).copy()
    work["author_channel_id"] = work["author_channel_id"].astype(str)
    work["video_id"] = work["video_id"].astype(str)
    reference = videos.dropna(subset=["video_id"]).drop_duplicates("video_id").copy()
    reference["video_id"] = reference["video_id"].astype(str)
    work = work.merge(
        reference[["video_id", "channel_id", "category", "source_query"]].rename(
            columns={"channel_id": "canal_video", "category": "categoria_video", "source_query": "consulta_video"}
        ),
        on="video_id", how="left",
    )
    grouped = work.groupby("author_channel_id")
    profiles = grouped.agg(
        label=("author_name", "first"),
        author_handle=("author_handle", "first"),
        comentarios=("comment_id", "count"),
        videos_comentados=("video_id", "nunique"),
        canales_distintos=("canal_video", "nunique"),
        categorias_distintas=("categoria_video", "nunique"),
        consultas_distintas=("consulta_video", "nunique"),
    ).reset_index()
    profiles["comentarios_por_video"] = profiles["comentarios"] / profiles["videos_comentados"]
    profiles["diversidad_canales"] = profiles["canales_distintos"] / profiles["videos_comentados"]
    profiles["entropia_participacion"] = [
        _normalized_entropy(group.groupby("video_id").size().to_numpy())
        for _, group in grouped
    ]
    profiles["es_recurrente"] = profiles["videos_comentados"] > 1
    profiles["node"] = "author::" + profiles["author_channel_id"]
    return profiles.sort_values(["videos_comentados", "comentarios", "author_channel_id"], ascending=[False, False, True]).reset_index(drop=True)


def _normalized_entropy(counts: np.ndarray) -> float:
    """Entropía de Shannon normalizada del reparto de comentarios entre videos."""

    total = counts.sum()
    if total == 0 or len(counts) < 2:
        return 0.0
    probabilities = counts / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / float(np.log(len(counts)))


def video_profiles(bipartite_graph: nx.Graph, video_projection: nx.Graph, videos: pd.DataFrame) -> pd.DataFrame:
    """Resume alcance y audiencias compartidas por video."""

    reference = videos.dropna(subset=["video_id"]).drop_duplicates("video_id").copy()
    reference["video_id"] = reference["video_id"].astype(str)
    rows: list[dict] = []
    for node in sorted(video_projection.nodes):
        video_id = video_projection.nodes[node].get("node_id", node.removeprefix("video::"))
        authors = {
            neighbor for neighbor in bipartite_graph.neighbors(node)
        } if node in bipartite_graph else set()
        shared = {
            author for author in authors
            if any(other != node for other in bipartite_graph.neighbors(author))
        }
        neighbours = list(video_projection.neighbors(node))
        rows.append({
            "video_id": video_id,
            "node": node,
            "label": video_projection.nodes[node].get("label", node),
            "channel_id": video_projection.nodes[node].get("channel_id", ""),
            "channel_name": video_projection.nodes[node].get("channel_name", ""),
            "category": video_projection.nodes[node].get("category", ""),
            "comentarios": int(video_projection.nodes[node].get("comments_total", 0)),
            "autores_unicos": len(authors),
            "autores_compartidos": len(shared),
            "videos_conectados": len(neighbours),
            "canales_conectados": len({video_projection.nodes[other].get("channel_id", "") for other in neighbours} - {""}),
            "categorias_conectadas": len({video_projection.nodes[other].get("category", "") for other in neighbours} - {""}),
        })
    profiles = pd.DataFrame(rows)
    profiles = profiles.merge(reference[["video_id", "view_count", "source_query"]], on="video_id", how="left")
    profiles["proporcion_audiencia_compartida"] = np.where(
        profiles["autores_unicos"] > 0, profiles["autores_compartidos"] / profiles["autores_unicos"], 0.0
    )
    return profiles.sort_values(["videos_conectados", "autores_compartidos", "video_id"], ascending=[False, False, True]).reset_index(drop=True)


def articulation_table(graph: nx.Graph, red: str) -> pd.DataFrame:
    """Lista los puntos de articulación: nodos cuya remoción desconecta su componente."""

    columns = ["red", "node", "node_id", "node_type", "label", "grado", "componente_tamano"]
    if graph.number_of_edges() == 0:
        return pd.DataFrame(columns=columns)
    prepared = _stable_graph(graph)
    _, size = _component_lookup(prepared)
    points = sorted(nx.articulation_points(prepared))
    table = pd.DataFrame([
        {
            "red": red,
            "node": node,
            "node_id": prepared.nodes[node].get("node_id", node),
            "node_type": prepared.nodes[node].get("node_type", "desconocido"),
            "label": prepared.nodes[node].get("label", node),
            "grado": prepared.degree(node),
            "componente_tamano": size[node],
        }
        for node in points
    ], columns=columns)
    return table.sort_values(["grado", "node"], ascending=[False, True]).reset_index(drop=True)


def _connected_pairs(graph: nx.Graph, excluded: str | None = None) -> int:
    """Pares de nodos conectados, ignorando al nodo excluido en ambos extremos."""

    total = 0
    for component in nx.connected_components(graph):
        size = len(component) - (1 if excluded in component else 0)
        total += size * (size - 1) // 2
    return total


def removal_impact(graph: nx.Graph, nodes: list[str], red: str) -> pd.DataFrame:
    """Compara componentes y fragmentación antes y después de quitar cada nodo.

    La fragmentación es la proporción de pares *desconectados* medida siempre
    sobre el mismo conjunto de nodos (todos menos el removido), de modo que el
    incremento no se deba al simple hecho de tener un nodo menos.
    """

    columns = [
        "red", "node", "node_id", "node_type", "label", "grado", "aristas_eliminadas",
        "componentes_antes", "componentes_despues", "componentes_nuevos",
        "componente_mayor_antes", "componente_mayor_despues", "cambio_componente_mayor_pct",
        "fragmentacion_antes", "fragmentacion_despues", "incremento_fragmentacion", "es_punto_articulacion",
    ]
    if graph.number_of_nodes() == 0 or not nodes:
        return pd.DataFrame(columns=columns)

    prepared = _stable_graph(graph)
    articulation = set(nx.articulation_points(prepared)) if prepared.number_of_edges() else set()
    components_before = list(nx.connected_components(prepared))
    largest_before = max((len(group) for group in components_before), default=0)
    order = prepared.number_of_nodes()
    comparable_pairs = (order - 1) * (order - 2) // 2

    rows: list[dict] = []
    for node in sorted(set(nodes)):
        if node not in prepared:
            continue
        reduced = prepared.copy()
        degree = prepared.degree(node)
        reduced.remove_node(node)
        components_after = list(nx.connected_components(reduced))
        largest_after = max((len(group) for group in components_after), default=0)
        isolated = degree == 0
        pairs_before = _connected_pairs(prepared, excluded=node)
        pairs_after = _connected_pairs(reduced)
        fragmentation_before = 1 - pairs_before / comparable_pairs if comparable_pairs else 0.0
        fragmentation_after = 1 - pairs_after / comparable_pairs if comparable_pairs else 0.0
        rows.append({
            "red": red,
            "node": node,
            "node_id": prepared.nodes[node].get("node_id", node),
            "node_type": prepared.nodes[node].get("node_type", "desconocido"),
            "label": prepared.nodes[node].get("label", node),
            "grado": degree,
            "aristas_eliminadas": degree,
            "componentes_antes": len(components_before),
            "componentes_despues": len(components_after),
            "componentes_nuevos": len(components_after) - (len(components_before) - (1 if isolated else 0)),
            "componente_mayor_antes": largest_before,
            "componente_mayor_despues": largest_after,
            "cambio_componente_mayor_pct": 100 * (largest_after - largest_before) / largest_before if largest_before else 0.0,
            "fragmentacion_antes": float(fragmentation_before),
            "fragmentacion_despues": float(fragmentation_after),
            "incremento_fragmentacion": float(fragmentation_after - fragmentation_before),
            "es_punto_articulacion": node in articulation,
        })
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(
        ["incremento_fragmentacion", "componentes_nuevos", "node"], ascending=[False, False, True]
    ).reset_index(drop=True)


def removal_candidates(centrality: pd.DataFrame, articulation: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Reúne puntos de articulación y nodos líderes en intermediación y peso."""

    candidates: list[str] = list(articulation["node"]) if not articulation.empty else []
    if not centrality.empty:
        positive = centrality[centrality["intermediacion"] > 0]
        candidates += list(positive.nlargest(top_n, ["intermediacion", "grado_ponderado"])["node"])
        candidates += list(centrality.nlargest(top_n, ["grado_ponderado", "grado"])["node"])
    return sorted(set(candidates))


def bridge_authors(centrality: pd.DataFrame, profiles: pd.DataFrame, articulation: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    """Clasifica autores puente distinguiendo recurrencia, intermediación y articulación."""

    articulation_nodes = set(articulation["node"]) if not articulation.empty else set()
    table = centrality.merge(
        profiles.drop(columns=["label"]), on="node", how="left", suffixes=("", "_perfil")
    )
    table["videos_comentados"] = table["videos_comentados"].fillna(0).astype(int)
    table["comentarios"] = table["comentarios"].fillna(0).astype(int)
    table["es_recurrente"] = table["videos_comentados"] > 1
    table["es_punto_articulacion"] = table["node"].isin(articulation_nodes)
    table["intermediacion_positiva"] = table["intermediacion"] > 0
    if not impact.empty:
        gain = impact.set_index("node")["componentes_nuevos"]
        table["componentes_nuevos_si_se_remueve"] = table["node"].map(gain).fillna(0).astype(int)
    else:
        table["componentes_nuevos_si_se_remueve"] = 0
    table["tipo_puente"] = np.select(
        [table["es_punto_articulacion"], table["intermediacion_positiva"]],
        ["puente estructural (su remoción fragmenta)", "puente redundante (conecta con rutas alternativas)"],
        default="sin función de puente observada",
    )
    bridges = table[table["es_punto_articulacion"] | table["intermediacion_positiva"]].copy()
    columns = [
        "node", "node_id", "label", "author_handle", "comentarios", "videos_comentados",
        "canales_distintos", "categorias_distintas", "diversidad_canales", "entropia_participacion",
        "grado", "grado_ponderado", "intermediacion", "armonica", "pagerank",
        "es_recurrente", "intermediacion_positiva", "es_punto_articulacion",
        "componentes_nuevos_si_se_remueve", "tipo_puente", "componente", "componente_tamano",
    ]
    bridges = bridges[[column for column in columns if column in bridges.columns]]
    return bridges.sort_values(
        ["es_punto_articulacion", "intermediacion", "videos_comentados", "node"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def articulator_videos(centrality: pd.DataFrame, profiles: pd.DataFrame, articulation: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    """Selecciona videos que conectan audiencias y marca los que sostienen la red."""

    articulation_nodes = set(articulation["node"]) if not articulation.empty else set()
    table = centrality.merge(profiles.drop(columns=["label"]), on="node", how="left", suffixes=("", "_perfil"))
    table["es_punto_articulacion"] = table["node"].isin(articulation_nodes)
    table["intermediacion_positiva"] = table["intermediacion"] > 0
    if not impact.empty:
        gain = impact.set_index("node")["componentes_nuevos"]
        table["componentes_nuevos_si_se_remueve"] = table["node"].map(gain).fillna(0).astype(int)
    else:
        table["componentes_nuevos_si_se_remueve"] = 0
    table["tipo_articulador"] = np.select(
        [table["es_punto_articulacion"], table["intermediacion_positiva"]],
        ["articulador (su remoción segmenta la red)", "conector redundante (audiencias unidas por varias rutas)"],
        default="sin función articuladora observada",
    )
    selected = table[table["es_punto_articulacion"] | table["intermediacion_positiva"]].copy()
    columns = [
        "node", "node_id", "label", "channel_name", "category", "comentarios", "autores_unicos",
        "autores_compartidos", "proporcion_audiencia_compartida", "videos_conectados", "canales_conectados",
        "categorias_conectadas", "grado", "grado_ponderado", "intermediacion", "armonica", "pagerank",
        "intermediacion_positiva", "es_punto_articulacion", "componentes_nuevos_si_se_remueve",
        "tipo_articulador", "componente", "componente_tamano",
    ]
    selected = selected[[column for column in columns if column in selected.columns]]
    return selected.sort_values(
        ["es_punto_articulacion", "intermediacion", "videos_conectados", "node"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def community_structure(graph: nx.Graph, membership: dict[str, int], centrality: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Describe la estructura interna de las comunidades y marca las `top_n` principales."""

    columns = [
        "comunidad", "autores", "aristas_internas", "aristas_externas", "densidad_interna",
        "grado_medio_interno", "puntos_articulacion_internos", "autor_top_intermediacion",
        "intermediacion_top", "caida_componente_mayor_pct", "dependiente_de_pocos_nodos", "es_principal",
    ]
    if not membership:
        return pd.DataFrame(columns=columns)
    prepared = _stable_graph(graph)
    scores = centrality.set_index("node")["intermediacion"].to_dict() if not centrality.empty else {}
    labels = centrality.set_index("node")["label"].to_dict() if not centrality.empty else {}
    rows: list[dict] = []
    for community in sorted(set(membership.values())):
        members = sorted(node for node, value in membership.items() if value == community)
        subgraph = _stable_graph(prepared.subgraph(members).copy())
        internal = subgraph.number_of_edges()
        external = sum(1 for node in members for neighbor in prepared.neighbors(node) if membership.get(neighbor) != community)
        size = len(members)
        largest_before = max((len(group) for group in nx.connected_components(subgraph)), default=0)
        internal_points = sorted(nx.articulation_points(subgraph)) if internal else []
        leader = max(members, key=lambda node: (scores.get(node, 0.0), node)) if members else ""
        reduced = subgraph.copy()
        if leader in reduced:
            reduced.remove_node(leader)
        largest_after = max((len(group) for group in nx.connected_components(reduced)), default=0)
        drop = 100 * (largest_before - largest_after) / largest_before if largest_before else 0.0
        rows.append({
            "comunidad": community,
            "autores": size,
            "aristas_internas": internal,
            "aristas_externas": external,
            "densidad_interna": nx.density(subgraph) if size > 1 else 0.0,
            "grado_medio_interno": 2 * internal / size if size else 0.0,
            "puntos_articulacion_internos": len(internal_points),
            "autor_top_intermediacion": labels.get(leader, leader),
            "intermediacion_top": float(scores.get(leader, 0.0)),
            "caida_componente_mayor_pct": float(drop),
            "dependiente_de_pocos_nodos": bool(internal_points) or drop > 20,
        })
    table = pd.DataFrame(rows, columns=columns).sort_values(["autores", "comunidad"], ascending=[False, True])
    table["es_principal"] = np.arange(len(table)) < top_n
    return table.reset_index(drop=True)
