"""Centralidades, participantes puente y experimentos de remoción del Laboratorio 6."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab6_social.centrality import (  # noqa: E402
    articulation_table,
    articulator_videos,
    author_profiles,
    bridge_authors,
    centrality_measures,
    centrality_rationale,
    community_structure,
    removal_candidates,
    removal_impact,
    video_profiles,
)
from lab6_social.io import find_dataset, load_comments, load_videos  # noqa: E402
from lab6_social.networks import (  # noqa: E402
    build_bipartite_network,
    build_projections,
    detect_author_communities,
)
from lab6_social.visualization_network_final import (  # noqa: E402
    save_articulator_videos,
    save_bridge_authors,
    save_centrality_panel,
    save_removal_impact,
)


SEED = 42
DATA = ROOT / "Data"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
AUTHOR_NETWORK = "proyeccion_autores"
VIDEO_NETWORK = "proyeccion_videos"


def write(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    return frame


def plural(count: int, singular: str, plural_form: str) -> str:
    return singular if count == 1 else plural_form


def describe_nodes(frame: pd.DataFrame, column: str, top_n: int = 3) -> str:
    """Enumera las etiquetas principales de una tabla ya ordenada por relevancia."""

    if frame.empty:
        return "ninguno"
    top = frame.nlargest(top_n, column) if column in frame else frame.head(top_n)
    return "; ".join(f"{str(row.label)[:46]} ({getattr(row, column):.4g})" for row in top.itertuples())


def main() -> None:
    for directory in (TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    videos, _ = load_videos(find_dataset(DATA, "videos"))
    comments = load_comments(find_dataset(DATA, "comments"))

    bipartite_graph, _, _ = build_bipartite_network(videos, comments)
    author_projection, video_projection = build_projections(bipartite_graph)

    author_centrality = centrality_measures(author_projection, AUTHOR_NETWORK)
    video_centrality = centrality_measures(video_projection, VIDEO_NETWORK)
    profiles = author_profiles(comments, videos)
    videos_meta = video_profiles(bipartite_graph, video_projection, videos)

    author_points = articulation_table(author_projection, AUTHOR_NETWORK)
    video_points = articulation_table(video_projection, VIDEO_NETWORK)

    author_impact = removal_impact(
        author_projection, removal_candidates(author_centrality, author_points), AUTHOR_NETWORK
    )
    video_impact = removal_impact(
        video_projection, removal_candidates(video_centrality, video_points), VIDEO_NETWORK
    )
    impact = pd.concat([author_impact, video_impact], ignore_index=True)
    impact = impact.sort_values(
        ["incremento_fragmentacion", "componentes_nuevos", "red", "node"], ascending=[False, False, True, True]
    ).reset_index(drop=True)

    centrality_authors = author_centrality.merge(
        profiles[["node", "comentarios", "videos_comentados", "canales_distintos", "categorias_distintas",
                  "consultas_distintas", "comentarios_por_video", "diversidad_canales",
                  "entropia_participacion", "es_recurrente"]],
        on="node", how="left",
    )
    centrality_videos = video_centrality.merge(
        videos_meta[["node", "channel_name", "category", "comentarios", "autores_unicos", "autores_compartidos",
                     "proporcion_audiencia_compartida", "videos_conectados", "canales_conectados",
                     "categorias_conectadas", "view_count"]],
        on="node", how="left",
    )
    write(centrality_authors, "centralidad_autores.csv")
    write(centrality_videos, "centralidad_videos.csv")
    write(centrality_rationale(), "justificacion_centralidades.csv")

    recurrent = profiles[profiles["es_recurrente"]].merge(
        author_centrality[["node", "grado", "grado_ponderado", "intermediacion", "armonica", "pagerank"]],
        on="node", how="left",
    ).sort_values(["videos_comentados", "intermediacion", "comentarios", "author_channel_id"],
                  ascending=[False, False, False, True]).reset_index(drop=True)
    write(recurrent, "autores_recurrentes.csv")

    bridges = bridge_authors(author_centrality, profiles, author_points, author_impact)
    articulators = articulator_videos(video_centrality, videos_meta, video_points, video_impact)
    write(bridges, "autores_puente.csv")
    write(articulators, "videos_articuladores.csv")
    write(pd.concat([author_points, video_points], ignore_index=True), "puntos_articulacion.csv")
    write(impact, "impacto_remocion.csv")

    membership, modularity, algorithm = detect_author_communities(author_projection, SEED)
    communities = community_structure(author_projection, membership, author_centrality, top_n=3)
    communities["algoritmo"] = algorithm
    communities["modularidad_global"] = modularity
    write(communities, "comunidades_principales_estructura.csv")

    structural_bridges = bridges[bridges["es_punto_articulacion"]]
    redundant_bridges = bridges[~bridges["es_punto_articulacion"]]
    structural_videos = articulators[articulators["es_punto_articulacion"]]
    max_videos = int(profiles["videos_comentados"].max())
    rho_global = centrality_authors["videos_comentados"].corr(centrality_authors["intermediacion"], method="spearman")
    rho_recurrent = recurrent["videos_comentados"].corr(recurrent["intermediacion"], method="spearman")
    top_betweenness = author_centrality.nlargest(1, "intermediacion").iloc[0]
    videos_with_comments = int(comments["video_id"].nunique())
    videos_without_comments = len(videos) - videos_with_comments
    isolated_authors = int((author_centrality["grado"] == 0).sum())
    isolated_videos = int((video_centrality["grado"] == 0).sum())
    connected_videos = len(video_centrality) - isolated_videos
    author_bridge_component = int(structural_bridges["componente_tamano"].max()) if not structural_bridges.empty else 0
    video_fragmentation = 100 * float(impact.loc[impact["red"] == VIDEO_NETWORK, "fragmentacion_antes"].max()) if not impact.empty else 0.0
    worst = impact.iloc[0] if not impact.empty else None
    worst_line = (
        f"{worst['label']} en la {worst['red'].replace('_', ' ')}: la componente mayor cae de "
        f"{int(worst['componente_mayor_antes'])} a {int(worst['componente_mayor_despues'])} nodos "
        f"({worst['cambio_componente_mayor_pct']:.1f} %) y {plural(int(worst['componentes_nuevos']), 'aparece', 'aparecen')} "
        f"{int(worst['componentes_nuevos'])} {plural(int(worst['componentes_nuevos']), 'componente nuevo', 'componentes nuevos')}"
    ) if worst is not None else "sin candidatos evaluados"

    interpretation = pd.DataFrame([
        {"pregunta": "1. ¿Qué autores participan en varios videos?",
         "respuesta": f"{len(recurrent)} de {len(profiles)} autores ({100 * len(recurrent) / len(profiles):.1f} %) comentaron en más de un video; el máximo observado es {max_videos} videos. Encabezan: {describe_nodes(recurrent, 'videos_comentados')}. La participación es casi siempre puntual: {len(profiles) - len(recurrent)} autores aparecen en un solo video."},
        {"pregunta": "2. ¿Qué autores tienen mayor intermediación?",
         "respuesta": f"Solo {int((author_centrality['intermediacion'] > 0).sum())} autores tienen intermediación positiva; el resto está dentro de una sola camarilla de video y no aparece en ningún camino mínimo. Los mayores son: {describe_nodes(author_centrality, 'intermediacion')}."},
        {"pregunta": "3. ¿Coinciden recurrencia e intermediación?",
         "respuesta": f"Coinciden como condición necesaria, no como jerarquía. Comentar en más de un video es la única forma de obtener intermediación positiva, de modo que la correlación de Spearman sobre los {len(profiles)} autores es rho={rho_global:.3f} por construcción; entre los {len(recurrent)} autores recurrentes cae a rho={rho_recurrent:.3f}. Ningún autor con el máximo de {max_videos} videos encabeza la intermediación: la lidera {top_betweenness.label}, con {int(centrality_authors.loc[centrality_authors['node'] == top_betweenness.node, 'videos_comentados'].iloc[0])} videos y {top_betweenness.intermediacion:.3f}. Lo decisivo es qué videos se conectan, no cuántos."},
        {"pregunta": "4. ¿Qué autores funcionan realmente como puentes?",
         "respuesta": f"{len(structural_bridges)} autores son puntos de articulación y su remoción parte la red: {describe_nodes(structural_bridges, 'intermediacion')}. Otros {len(redundant_bridges)} tienen intermediación positiva pero rutas alternativas ({describe_nodes(redundant_bridges, 'intermediacion')}), así que conectan sin ser indispensables."},
        {"pregunta": "5. ¿Qué videos conectan audiencias?",
         "respuesta": f"{connected_videos} de {len(video_centrality)} videos comparten al menos un autor con otro video y forman una sola componente. Los de mayor intermediación son: {describe_nodes(video_centrality, 'intermediacion')}. Su capacidad de conectar audiencias se apoya en los {len(recurrent)} autores que comentaron en más de un video, no en el volumen de comentarios."},
        {"pregunta": "6. ¿Qué nodos son puntos de articulación?",
         "respuesta": f"{len(author_points)} autores y {len(video_points)} videos. Los autores puente pertenecen a la componente mayor de la proyección autor–autor ({author_bridge_component} nodos), donde sostienen la unión entre camarillas de videos distintos; los videos articuladores están en la única componente conectada de la proyección video–video ({connected_videos} videos)."},
        {"pregunta": "7. ¿Qué sucede al removerlos?",
         "respuesta": f"De los {len(impact)} nodos evaluados, {int((impact['componentes_nuevos'] > 0).sum())} fragmentan la red al ser removidos. El caso de mayor efecto es {worst_line}. Fuera de los puntos de articulación, quitar nodos de grado ponderado alto no cambia el número de componentes: la red es un conjunto de camarillas por video, redundante hacia adentro y frágil solo en los pocos enlaces entre camarillas."},
        {"pregunta": "8. ¿Cuáles resultados pueden explicarse por cobertura parcial?",
         "respuesta": f"Casi todos. Solo {videos_with_comments} de {len(videos)} videos tienen comentarios recolectados, así que {videos_without_comments} videos quedan aislados por ausencia de datos y otros {isolated_videos - videos_without_comments} tienen comentarios pero ningún autor compartido; la proyección video–video ya parte de una fragmentación de {video_fragmentation:.2f} % de pares desconectados. También {isolated_authors} autores quedan aislados por ser el único comentarista de su video. El bajo número de puentes mide la cobertura de la muestra tanto como el comportamiento de las audiencias."},
        {"pregunta": "9. ¿Qué diferencia existe entre centralidad observada e influencia social real?",
         "respuesta": "La centralidad describe la posición de un nodo dentro de la red que se pudo observar: coincidencias de comentarios en una muestra recolectada por consultas de búsqueda y en una fecha concreta. No mide persuasión, autoridad ni alcance. Un autor puede encabezar la intermediación por haber comentado en dos videos con pocos comentarios, y un video muy visto puede no aparecer en la red por no tener comentarios recolectados. Describir la estructura observada no autoriza a inferir influencia ni a generalizar a los usuarios de YouTube."},
    ])
    write(interpretation, "interpretacion_estructural.csv")

    write(pd.DataFrame([
        {"pregunta": "3.5 ¿Qué autores funcionan como puentes entre contenidos que de otra forma permanecerían separados?",
         "respuesta": f"{len(structural_bridges)} autores son puntos de articulación en la proyección autor–autor: al removerlos la red gana componentes. {len(structural_videos)} videos cumplen el papel equivalente en la proyección video–video. La evidencia está en autores_puente.csv, videos_articuladores.csv, puntos_articulacion.csv e impacto_remocion.csv.",
         "matiz": f"De los {len(recurrent)} autores recurrentes, solo {len(structural_bridges)} son puentes estructurales; el resto comenta en videos que ya estaban conectados por otras personas. Una arista sigue significando copresencia de comentarios, no conversación ni acuerdo."},
    ]), "respuesta_35_autores_puente.csv")

    summary = {
        "semilla": SEED,
        "autores": {
            "nodos": len(author_centrality), "aristas": author_projection.number_of_edges(),
            "recurrentes": len(recurrent), "intermediacion_positiva": int((author_centrality["intermediacion"] > 0).sum()),
            "puntos_articulacion": len(author_points), "puentes_estructurales": len(structural_bridges),
            "puentes_redundantes": len(redundant_bridges), "aislados": isolated_authors,
        },
        "videos": {
            "nodos": len(video_centrality), "aristas": video_projection.number_of_edges(),
            "conectados": int((video_centrality["grado"] > 0).sum()),
            "puntos_articulacion": len(video_points), "articuladores": len(structural_videos), "aislados": isolated_videos,
        },
        "remocion": {
            "nodos_evaluados": len(impact),
            "con_fragmentacion": int((impact["componentes_nuevos"] > 0).sum()) if not impact.empty else 0,
            "maximo_incremento_fragmentacion": float(impact["incremento_fragmentacion"].max()) if not impact.empty else 0.0,
        },
        "comunidades": {
            "algoritmo": algorithm, "total": len(communities), "modularidad": modularity,
            "dependientes_de_pocos_nodos": int(communities["dependiente_de_pocos_nodos"].sum()) if not communities.empty else 0,
        },
    }
    (TABLES / "resumen_redes_finales.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_centrality_panel(centrality_authors, "Centralidad de autores (proyección autor–autor)", FIGURES / "centralidad_autores.png")
    save_centrality_panel(centrality_videos, "Centralidad de videos (proyección video–video)", FIGURES / "centralidad_videos.png")
    save_bridge_authors(author_projection, bridges, centrality_authors, FIGURES / "autores_puente.png", SEED)
    save_articulator_videos(video_projection, articulators, FIGURES / "videos_articuladores.png", SEED)
    save_removal_impact(impact, FIGURES / "impacto_remocion.png")

    print(f"Autores: {len(author_centrality)} nodos, {len(recurrent)} recurrentes, {len(structural_bridges)} puentes estructurales")
    print(f"Videos: {len(video_centrality)} nodos, {summary['videos']['conectados']} conectados, {len(structural_videos)} articuladores")
    print(f"Puntos de articulación: {len(author_points)} autores + {len(video_points)} videos")
    print(f"Remoción: {len(impact)} nodos evaluados, {summary['remocion']['con_fragmentacion']} fragmentan la red")
    print(f"Comunidades: {len(communities)} ({algorithm}), modularidad={modularity:.4f}")


if __name__ == "__main__":
    main()
