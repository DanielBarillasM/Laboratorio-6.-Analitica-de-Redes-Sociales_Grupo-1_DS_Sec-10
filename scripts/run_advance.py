"""Genera todas las tablas, métricas y figuras del avance del Laboratorio 6."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import networkx as nx
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab6_social.analysis import (  # noqa: E402
    author_participation,
    concentration_table,
    consistency_report,
    hashtag_frequencies,
    integrate_data,
    missing_report,
    ngram_frequencies,
    participation_tables,
    popularity_association,
    quality_report,
)
from lab6_social.io import detect_table_format, find_dataset, load_comments, load_videos, read_table  # noqa: E402
from lab6_social.networks import (  # noqa: E402
    build_bipartite_network,
    build_projections,
    community_characterization,
    degree_table,
    detect_author_communities,
    graph_metrics,
)
from lab6_social.preprocessing import add_clean_text  # noqa: E402
from lab6_social.visualization import (  # noqa: E402
    save_bipartite,
    save_communities,
    save_concentration,
    save_degree_distributions,
    save_eda_overview,
    save_frequency_plot,
    save_popularity,
    save_projection,
)


SEED = 42
DATA = ROOT / "Data"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
PROCESSED = ROOT / "data" / "processed"


def export_graph_edges(graph: nx.Graph, name: str) -> pd.DataFrame:
    table = pd.DataFrame([
        {"source": source, "target": target, "weight": data.get("weight", 1)}
        for source, target, data in graph.edges(data=True)
    ])
    table.to_csv(TABLES / f"aristas_{name}.csv", index=False, encoding="utf-8-sig")
    return table


def component_table(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, graph in graphs.items():
        components = sorted((len(group) for group in nx.connected_components(graph)), reverse=True)
        rows.extend(
            {"red": name, "componente": index + 1, "nodos": size}
            for index, size in enumerate(components)
        )
    return pd.DataFrame(rows)


def top_join(values: pd.Series, n: int = 3) -> str:
    return " | ".join(str(value) for value in values.value_counts().head(n).index)


def main() -> None:
    for directory in (TABLES, FIGURES, PROCESSED):
        directory.mkdir(parents=True, exist_ok=True)

    video_path = find_dataset(DATA, "videos")
    comment_path = find_dataset(DATA, "comments")
    raw_videos = read_table(video_path)
    raw_comments = read_table(comment_path)
    videos, repairs = load_videos(video_path)
    comments = load_comments(comment_path)
    comments, cleaning_audit = add_clean_text(comments)

    formats = pd.DataFrame([
        {"dataset": "videos", "archivo": video_path.name, "extension": video_path.suffix.lower(), "formato_real": detect_table_format(video_path)},
        {"dataset": "comments", "archivo": comment_path.name, "extension": comment_path.suffix.lower(), "formato_real": detect_table_format(comment_path)},
    ])
    formats.to_csv(TABLES / "formatos_detectados.csv", index=False, encoding="utf-8-sig")
    repairs.to_csv(TABLES / "video_ids_recuperados.csv", index=False, encoding="utf-8-sig")

    quality = pd.concat([
        quality_report(raw_videos, "videos_original", "video_id"),
        quality_report(raw_comments, "comments_original", "comment_id"),
        quality_report(videos, "videos_normalizado", "video_id"),
        quality_report(comments, "comments_normalizado", "comment_id"),
    ], ignore_index=True)
    quality.to_csv(TABLES / "calidad_resumen.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        missing_report(raw_videos, "videos"), missing_report(raw_comments, "comments")
    ], ignore_index=True).to_csv(TABLES / "calidad_variables.csv", index=False, encoding="utf-8-sig")
    consistency_report(videos, comments).to_csv(TABLES / "consistencia_identificadores.csv", index=False, encoding="utf-8-sig")
    cleaning_audit.to_csv(TABLES / "auditoria_limpieza_texto.csv", index=False, encoding="utf-8-sig")
    examples = comments.loc[
        comments["texto_original"].str.strip() != comments["texto_limpio"],
        ["comment_id", "texto_original", "texto_limpio"],
    ].head(15)
    examples.to_csv(TABLES / "ejemplos_limpieza.csv", index=False, encoding="utf-8-sig")
    treatment = pd.DataFrame([
        {"variable": "viewer_rating", "tratamiento": "Excluir", "justificacion": "100 % faltante; no aporta información."},
        {"variable": "is_pinned", "tratamiento": "Conservar para auditoría, no modelar", "justificacion": "Es constante (False)."},
        {"variable": "published_time/published_text", "tratamiento": "Describir con cautela", "justificacion": "Tiempo relativo dependiente de la fecha de recolección."},
        {"variable": "reply_count", "tratamiento": "Conteo descriptivo", "justificacion": "No identifica autores de respuestas y no genera aristas."},
        {"variable": "source_query", "tratamiento": "Variable de muestreo", "justificacion": "Explica recolección, no necesariamente el tema definitivo."},
        {"variable": "nombres/handles", "tratamiento": "Solo etiquetas", "justificacion": "Los nodos se identifican mediante IDs estables."},
        {"variable": "lematización", "tratamiento": "No aplicada en el avance", "justificacion": "Sin un modelo morfosintáctico español validado puede deformar nombres y términos; se evalúa para la fase final."},
        {"variable": "emojis", "tratamiento": "Retirados de texto_limpio, conservados en texto_original", "justificacion": "Se preservan para sentimiento en la fase final."},
    ])
    treatment.to_csv(TABLES / "tratamiento_variables.csv", index=False, encoding="utf-8-sig")

    integrated, integration_summary = integrate_data(videos, comments)
    integration_summary.to_csv(TABLES / "integracion_resumen.csv", index=False, encoding="utf-8-sig")
    videos.to_csv(PROCESSED / "videos_normalizados.csv", index=False, encoding="utf-8-sig")
    comments.to_csv(PROCESSED / "comentarios_limpios.csv", index=False, encoding="utf-8-sig")

    unigrams = ngram_frequencies(comments["texto_limpio"], 1, 30)
    bigrams = ngram_frequencies(comments["texto_limpio"], 2, 30)
    hashtags = hashtag_frequencies(comments["texto_original"], 30)
    unigrams.to_csv(TABLES / "unigramas_frecuentes.csv", index=False, encoding="utf-8-sig")
    bigrams.to_csv(TABLES / "bigramas_frecuentes.csv", index=False, encoding="utf-8-sig")
    hashtags.to_csv(TABLES / "hashtags_frecuentes.csv", index=False, encoding="utf-8-sig")

    per_video, per_channel = participation_tables(videos, comments)
    per_video.to_csv(TABLES / "participacion_por_video.csv", index=False, encoding="utf-8-sig")
    per_channel.to_csv(TABLES / "participacion_por_canal.csv", index=False, encoding="utf-8-sig")
    authors = author_participation(comments)
    authors.to_csv(TABLES / "participacion_por_autor.csv", index=False, encoding="utf-8-sig")
    concentration = concentration_table(per_video, per_channel)
    concentration.to_csv(TABLES / "concentracion_participacion.csv", index=False, encoding="utf-8-sig")
    association = popularity_association(per_video)
    association.to_csv(TABLES / "asociacion_visibilidad_participacion.csv", index=False, encoding="utf-8-sig")
    videos["category"].value_counts().rename_axis("category").reset_index(name="videos").to_csv(
        TABLES / "categorias.csv", index=False, encoding="utf-8-sig"
    )
    videos["source_query"].value_counts().rename_axis("source_query").reset_index(name="videos").to_csv(
        TABLES / "consultas_videos.csv", index=False, encoding="utf-8-sig"
    )

    bipartite_graph, nodes, edges = build_bipartite_network(videos, comments)
    author_projection, video_projection = build_projections(bipartite_graph)
    nodes.to_csv(TABLES / "nodos_bipartita.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(TABLES / "aristas_bipartita.csv", index=False, encoding="utf-8-sig")
    export_graph_edges(author_projection, "autores")
    export_graph_edges(video_projection, "videos")

    graphs = {
        "bipartita_autor_video": bipartite_graph,
        "proyeccion_autores": author_projection,
        "proyeccion_videos": video_projection,
    }
    metrics = pd.DataFrame([graph_metrics(graph, name) for name, graph in graphs.items()])
    metrics.to_csv(TABLES / "metricas_redes.csv", index=False, encoding="utf-8-sig")
    degrees = pd.concat([degree_table(graph, name) for name, graph in graphs.items()], ignore_index=True)
    degrees.to_csv(TABLES / "distribuciones_grado.csv", index=False, encoding="utf-8-sig")
    degrees[degrees["degree"] <= 1].to_csv(
        TABLES / "nodos_perifericos_y_aislados.csv", index=False, encoding="utf-8-sig"
    )
    component_table(graphs).to_csv(TABLES / "componentes_redes.csv", index=False, encoding="utf-8-sig")

    membership, modularity, algorithm = detect_author_communities(author_projection, SEED)
    community_summary, comment_membership = community_characterization(membership, comments)
    membership_table = pd.DataFrame([{"node": node, "community": community} for node, community in membership.items()])
    membership_table.to_csv(TABLES / "membresia_comunidades.csv", index=False, encoding="utf-8-sig")
    comment_membership.to_csv(TABLES / "comentarios_comunidades.csv", index=False, encoding="utf-8-sig")
    if not community_summary.empty:
        enriched = comment_membership.merge(
            integrated[["comment_id", "title", "channel_name_video", "category"]], on="comment_id", how="left"
        )
        descriptors = []
        for community, group in enriched.groupby("community"):
            descriptors.append({
                "community": community,
                "top_videos": top_join(group["title"]),
                "top_channels": top_join(group["channel_name_video"]),
                "top_categories": top_join(group["category"]),
            })
        community_summary = community_summary.merge(pd.DataFrame(descriptors), on="community", how="left")
    community_summary["algoritmo"] = algorithm
    community_summary["modularidad_global"] = modularity
    community_summary["sentimiento"] = "Pendiente para fase final con herramienta validada para español"
    community_summary.to_csv(TABLES / "resumen_comunidades.csv", index=False, encoding="utf-8-sig")

    top_video = per_video.iloc[0]
    top_channel = per_channel.iloc[0]
    recurrent = authors[authors["videos"] > 1]
    top5_video_share = concentration.query("unidad == 'video' and top_n == 5")["participacion_pct"].iloc[0]
    rho_all = association.iloc[0]["spearman_rho"]
    mandatory = pd.DataFrame([
        {"pregunta": "¿Qué videos y canales concentran la mayor participación?", "evidencia": f"El video líder es '{top_video.title}' ({top_video.comentarios} comentarios); el canal líder es '{top_channel.channel_name}' ({top_channel.comentarios}). El top 5 de videos reúne {top5_video_share:.1f} %."},
        {"pregunta": "¿Existen audiencias compartidas?", "evidencia": f"La proyección video-video contiene {video_projection.number_of_edges()} pares conectados por autores compartidos; la proyección autor-autor contiene {author_projection.number_of_edges()} pares."},
        {"pregunta": "¿Qué autores funcionan como puentes?", "evidencia": f"En el avance se observan {len(recurrent)} autores presentes en más de un video; su intermediación y efecto de remoción se reservan para la fase final."},
        {"pregunta": "¿Qué temas y sentimientos caracterizan comunidades?", "evidencia": f"Se detectaron {community_summary.shape[0]} comunidades con modularidad {modularity:.3f}; sus términos, videos y categorías están tabulados. El sentimiento formal en español queda pendiente."},
        {"pregunta": "¿Coinciden visualizaciones y participación?", "evidencia": f"La asociación de Spearman en los {len(per_video)} videos es rho={rho_all:.3f}; las visualizaciones y comentarios son cortes de cobertura distintos."},
        {"pregunta": "¿Qué limita las conclusiones?", "evidencia": f"Solo {int((per_video.comentarios > 0).sum())} de {len(per_video)} videos tienen comentarios recolectados; la selección depende de consultas y los conteos son fotografías temporales."},
    ])
    mandatory.to_csv(TABLES / "preguntas_obligatorias.csv", index=False, encoding="utf-8-sig")

    comments_with_replies = int((comments["reply_count"] > 0).sum())
    source_counts = comments["source_group"].value_counts()
    zero_like = int((comments["like_count"] == 0).sum())
    additional = pd.DataFrame([
        {"pregunta": "¿Qué tan recurrente es la audiencia entre videos?", "respuesta": f"{len(recurrent)} de {comments.author_channel_id.nunique()} autores comentaron en más de un video."},
        {"pregunta": "¿Cuántos comentarios muestran conversación observable?", "respuesta": f"{comments_with_replies} de {len(comments)} comentarios tienen al menos una respuesta, pero no se conoce quién respondió."},
        {"pregunta": "¿Cómo se distribuyen los comentarios según la estrategia de búsqueda?", "respuesta": "; ".join(f"{key}: {value}" for key, value in source_counts.items()) + "."},
        {"pregunta": "¿Cuántos comentarios no muestran likes?", "respuesta": f"{zero_like} de {len(comments)} tienen conteo cero después de normalizar espacios en blanco."},
    ])
    additional.to_csv(TABLES / "preguntas_adicionales.csv", index=False, encoding="utf-8-sig")

    scope = pd.DataFrame([
        {"criterio": "Calidad, limpieza y preprocesamiento", "puntos_rubrica": 18, "puntos_cubiertos": 18, "estado": "completo"},
        {"criterio": "Análisis exploratorio", "puntos_rubrica": 18, "puntos_cubiertos": 18, "estado": "completo"},
        {"criterio": "Red bipartita autor-video", "puntos_rubrica": 10, "puntos_cubiertos": 10, "estado": "completo"},
        {"criterio": "Proyecciones", "puntos_rubrica": 8, "puntos_cubiertos": 8, "estado": "completo"},
        {"criterio": "Topología y fragmentación", "puntos_rubrica": 12, "puntos_cubiertos": 12, "estado": "completo"},
        {"criterio": "Comunidades", "puntos_rubrica": 10, "puntos_cubiertos": 8, "estado": "parcial: falta sentimiento"},
        {"criterio": "Nodos centrales y participantes puente", "puntos_rubrica": 7, "puntos_cubiertos": 0, "estado": "fase final"},
        {"criterio": "Contenido y sentimiento", "puntos_rubrica": 5, "puntos_cubiertos": 0, "estado": "fase final"},
        {"criterio": "Interpretación, limitaciones y conclusiones finales", "puntos_rubrica": 12, "puntos_cubiertos": 0, "estado": "fase final; hay discusión preliminar"},
    ])
    scope.to_csv(TABLES / "alcance_avance_75.csv", index=False, encoding="utf-8-sig")
    evidence = pd.DataFrame([
        {"ejercicio": "1", "estado": "completo", "evidencia": "formatos_detectados.csv; integracion_resumen.csv; video_ids_recuperados.csv"},
        {"ejercicio": "2", "estado": "completo", "evidencia": "calidad_resumen.csv; calidad_variables.csv; auditoria_limpieza_texto.csv; tratamiento_variables.csv"},
        {"ejercicio": "3", "estado": "completo", "evidencia": "participacion_por_video.csv; participacion_por_canal.csv; frecuencias_contenido.png; preguntas_obligatorias.csv; preguntas_adicionales.csv"},
        {"ejercicio": "4", "estado": "completo", "evidencia": "nodos_bipartita.csv; aristas_bipartita.csv; red_bipartita_completa.png"},
        {"ejercicio": "5", "estado": "completo", "evidencia": "aristas_autores.csv; aristas_videos.csv; proyeccion_autores.png; proyeccion_videos.png"},
        {"ejercicio": "6", "estado": "completo", "evidencia": "metricas_redes.csv; distribuciones_grado.csv; componentes_redes.csv; nodos_perifericos_y_aislados.csv"},
        {"ejercicio": "7", "estado": "parcial avanzado", "evidencia": "membresia_comunidades.csv; resumen_comunidades.csv; comunidades_autores.png; sentimiento pendiente"},
        {"ejercicio": "8", "estado": "fase final", "evidencia": "centralidades, puentes y remoción pendientes"},
        {"ejercicio": "9", "estado": "fase final", "evidencia": "sentimiento validado para español pendiente"},
        {"ejercicio": "10", "estado": "fase final", "evidencia": "síntesis y conclusiones definitivas pendientes"},
    ])
    evidence.to_csv(TABLES / "evidencia_ejercicios.csv", index=False, encoding="utf-8-sig")

    summary = {
        "files": {"videos": video_path.name, "comments": comment_path.name},
        "formats": {"videos": detect_table_format(video_path), "comments": detect_table_format(comment_path)},
        "counts": {
            "videos": len(videos), "channels": int(videos.channel_id.nunique()), "comments": len(comments),
            "authors": int(comments.author_channel_id.nunique()), "videos_with_comments": int(comments.video_id.nunique()),
            "repaired_video_ids": len(repairs),
        },
        "integration_pct": float(integration_summary.loc[integration_summary.metrica == "porcentaje_asociado", "valor"].iloc[0]),
        "top_video": {"title": str(top_video.title), "comments": int(top_video.comentarios)},
        "top_channel": {"name": str(top_channel.channel_name), "comments": int(top_channel.comentarios)},
        "top5_video_share": float(top5_video_share),
        "spearman_all": float(rho_all),
        "networks": metrics.to_dict("records"),
        "communities": {"algorithm": algorithm, "count": len(community_summary), "modularity": modularity},
        "scope_points": int(scope.puntos_cubiertos.sum()),
        "scope_total": int(scope.puntos_rubrica.sum()),
    }
    (TABLES / "resumen_avance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    save_eda_overview(videos, comments, per_video, FIGURES / "eda_panorama.png")
    save_frequency_plot(unigrams, bigrams, hashtags, FIGURES / "frecuencias_contenido.png")
    save_concentration(concentration, FIGURES / "concentracion_participacion.png")
    save_popularity(per_video, FIGURES / "visibilidad_vs_participacion.png")
    save_bipartite(bipartite_graph, FIGURES / "red_bipartita_completa.png", SEED)
    save_projection(author_projection, "Proyección autor–autor", FIGURES / "proyeccion_autores.png", SEED)
    save_projection(video_projection, "Proyección video–video", FIGURES / "proyeccion_videos.png", SEED)
    save_degree_distributions(degrees, FIGURES / "distribuciones_grado.png")
    save_communities(author_projection, membership, FIGURES / "comunidades_autores.png", SEED)

    print(f"Datos: {len(videos)} videos, {len(comments)} comentarios, {comments.author_channel_id.nunique()} autores")
    print(f"Integración: {summary['integration_pct']:.1f} %")
    print(f"Red bipartita: {bipartite_graph.number_of_nodes()} nodos, {bipartite_graph.number_of_edges()} aristas")
    print(f"Comunidades: {len(community_summary)}, modularidad={modularity:.4f}")
    print(f"Cobertura planificada: {summary['scope_points']}/{summary['scope_total']} puntos")


if __name__ == "__main__":
    main()
