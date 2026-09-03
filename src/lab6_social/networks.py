"""Construcción, proyección y descripción de redes de participación."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite


def _author_node(identifier: str) -> str:
    return f"author::{identifier}"


def _video_node(identifier: str) -> str:
    return f"video::{identifier}"


def build_bipartite_network(videos: pd.DataFrame, comments: pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Crea la red autor-video; el peso cuenta comentarios del autor en el video."""

    graph = nx.Graph(name="Red bipartita autor-video")
    video_rows = videos.dropna(subset=["video_id"]).drop_duplicates("video_id")
    comment_rows = comments.dropna(subset=["author_channel_id", "video_id"])
    comment_counts = comment_rows.groupby("video_id").size()
    for row in video_rows.itertuples(index=False):
        graph.add_node(
            _video_node(str(row.video_id)), node_id=str(row.video_id), node_type="video", bipartite=1,
            label=str(row.title), channel_id=str(row.channel_id), channel_name=str(row.channel_name),
            category=str(row.category), view_count=float(row.view_count) if pd.notna(row.view_count) else np.nan,
            comments_total=int(comment_counts.get(row.video_id, 0)),
        )
    author_summary = comment_rows.groupby("author_channel_id").agg(
        label=("author_name", "first"), author_handle=("author_handle", "first"),
        comments_total=("comment_id", "count"), videos_count=("video_id", "nunique"),
    ).reset_index()
    for row in author_summary.itertuples(index=False):
        graph.add_node(
            _author_node(str(row.author_channel_id)), node_id=str(row.author_channel_id), node_type="author", bipartite=0,
            label=str(row.label), author_handle=str(row.author_handle), comments_total=int(row.comments_total),
            videos_count=int(row.videos_count),
        )
    edges = comment_rows.groupby(["author_channel_id", "video_id"]).size().rename("weight").reset_index()
    for row in edges.itertuples(index=False):
        graph.add_edge(
            _author_node(str(row.author_channel_id)), _video_node(str(row.video_id)), weight=int(row.weight),
            meaning="comentarios del autor en el video",
        )
    node_table = pd.DataFrame([{"node": node, **attrs} for node, attrs in graph.nodes(data=True)])
    edge_table = pd.DataFrame([{"source": s, "target": t, **attrs} for s, t, attrs in graph.edges(data=True)])
    return graph, node_table, edge_table


def build_projections(graph: nx.Graph) -> tuple[nx.Graph, nx.Graph]:
    authors = [node for node, data in graph.nodes(data=True) if data["node_type"] == "author"]
    videos = [node for node, data in graph.nodes(data=True) if data["node_type"] == "video"]
    return bipartite.weighted_projected_graph(graph, authors), bipartite.weighted_projected_graph(graph, videos)


def graph_metrics(graph: nx.Graph, graph_name: str) -> dict[str, float | int | str]:
    nodes = graph.number_of_nodes()
    components = list(nx.connected_components(graph)) if nodes else []
    largest = max(components, key=len, default=set())
    largest_graph = graph.subgraph(largest).copy()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    active = int((degrees > 0).sum()) if len(degrees) else 0
    return {
        "red": graph_name, "nodos": nodes, "aristas": graph.number_of_edges(), "nodos_activos": active,
        "nodos_aislados": nodes - active, "densidad": nx.density(graph) if nodes > 1 else 0.0,
        "grado_medio": float(degrees.mean()) if len(degrees) else 0.0,
        "grado_mediano": float(np.median(degrees)) if len(degrees) else 0.0,
        "grado_maximo": int(degrees.max()) if len(degrees) else 0,
        "componentes": len(components), "componente_mayor": len(largest),
        "componente_mayor_pct": 100 * len(largest) / nodes if nodes else 0.0,
        "transitividad": nx.transitivity(graph) if nodes else 0.0,
        "clustering_medio": nx.average_clustering(graph, weight="weight") if nodes else 0.0,
        "cohesion_nodos_lcc": nx.node_connectivity(largest_graph) if len(largest_graph) > 1 else 0,
        "cohesion_aristas_lcc": nx.edge_connectivity(largest_graph) if len(largest_graph) > 1 else 0,
        "diametro_lcc": nx.diameter(largest_graph) if len(largest_graph) > 1 else 0,
    }


def degree_table(graph: nx.Graph, graph_name: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"red": graph_name, "node": node, "node_type": graph.nodes[node].get("node_type", "projection"),
         "degree": degree, "weighted_degree": graph.degree(node, weight="weight")}
        for node, degree in graph.degree()
    ])


def detect_author_communities(author_projection: nx.Graph, seed: int = 42) -> tuple[dict[str, int], float, str]:
    active_nodes = [node for node, degree in author_projection.degree() if degree > 0]
    active_graph = author_projection.subgraph(active_nodes).copy()
    if active_graph.number_of_edges() == 0:
        return {}, 0.0, "sin aristas"
    try:
        communities = nx.community.louvain_communities(active_graph, weight="weight", seed=seed)
        algorithm = "Louvain ponderado"
    except AttributeError:
        communities = nx.community.greedy_modularity_communities(active_graph, weight="weight")
        algorithm = "modularidad voraz ponderada"
    ordered = sorted(communities, key=len, reverse=True)
    membership = {node: index + 1 for index, group in enumerate(ordered) for node in group}
    modularity = nx.community.modularity(active_graph, ordered, weight="weight")
    return membership, float(modularity), algorithm


def community_characterization(membership: dict[str, int], comments: pd.DataFrame, top_terms: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = comments.copy()
    work["node"] = work["author_channel_id"].astype(str).map(_author_node)
    work["community"] = work["node"].map(membership)
    work = work.dropna(subset=["community"])
    work["community"] = work["community"].astype(int)
    rows: list[dict] = []
    for community, group in work.groupby("community"):
        terms = Counter(" ".join(group["texto_limpio"].fillna("")).split())
        rows.append({
            "community": community, "authors": group["author_channel_id"].nunique(), "comments": len(group),
            "videos": group["video_id"].nunique(), "channels": group["channel_id"].nunique(),
            "top_terms": ", ".join(word for word, _ in terms.most_common(top_terms)),
        })
    summary = pd.DataFrame(rows).sort_values(["authors", "comments"], ascending=False)
    return summary, work[["comment_id", "author_channel_id", "video_id", "community"]]
