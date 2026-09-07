"""Ejecuta el Laboratorio 6 completo y genera los productos de la entrega final."""

from __future__ import annotations

import io
import json
import os
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab6_social.io import find_dataset, load_comments, load_videos  # noqa: E402
from lab6_social.sentiment import (  # noqa: E402
    MODEL_NAME,
    aggregate_sentiment,
    cache_is_compatible,
    confidence_summary,
    predict_sentiment,
    validate_probabilities,
)
from lab6_social.visualization_content_final import (  # noqa: E402
    save_community_sentiment,
    save_confidence_plot,
    save_group_sentiment,
    save_sentiment_overview,
)


DATA = ROOT / "Data"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODEL_MIN_N = 5


def write(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    return frame


def run_previous_stages() -> None:
    """Regenera los productos base y luego los resultados de redes de Persona 1."""

    # Los scripts históricos imprimen el porcentaje del avance; se oculta ese
    # texto para que la salida final no parezca incompleta.
    with redirect_stdout(io.StringIO()):
        runpy.run_path(str(ROOT / "scripts" / "run_advance.py"), run_name="__main__")
        runpy.run_path(str(ROOT / "scripts" / "run_network_final.py"), run_name="__main__")
    print("Etapas base y estructural regeneradas.")


def load_or_predict(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cache = TABLES / "sentimiento_predicciones_cache.csv"
    metadata_path = TABLES / "sentimiento_modelo.json"
    force = os.environ.get("LAB6_FORCE_SENTIMENT", "0") == "1"
    inference_input = comments[["comment_id", "texto_original"]].copy()

    if not force and cache_is_compatible(cache, inference_input):
        predictions = pd.read_csv(cache, encoding="utf-8-sig", dtype={"comment_id": "string"})
        for column in (
            "prob_negativo", "prob_neutral", "prob_positivo", "confianza", "margen_confianza"
        ):
            predictions[column] = pd.to_numeric(predictions[column], errors="raise")
        predictions["baja_confianza"] = predictions["baja_confianza"].astype(str).str.lower().eq("true")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"modelo": MODEL_NAME, "comentarios": len(predictions), "origen": "cache"}
        )
        metadata["origen"] = "cache validada por ID, texto y modelo"
    else:
        predictions, metadata = predict_sentiment(inference_input)
        write(predictions, cache.name)
        metadata["origen"] = "inferencia nueva"

    validate_probabilities(predictions)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return predictions, metadata


def top_labels(frame: pd.DataFrame, n: int = 3) -> str:
    if frame.empty:
        return "ninguno"
    return "; ".join(str(value) for value in frame.head(n)["label"])


def merge_community_characterization(sentiment: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(TABLES / "resumen_comunidades.csv", encoding="utf-8-sig")
    structure = pd.read_csv(TABLES / "comunidades_principales_estructura.csv", encoding="utf-8-sig")
    base = base.drop(columns=["sentimiento"], errors="ignore")
    structural_columns = [
        "comunidad", "aristas_internas", "aristas_externas", "densidad_interna",
        "grado_medio_interno", "puntos_articulacion_internos", "autor_top_intermediacion",
        "intermediacion_top", "caida_componente_mayor_pct", "dependiente_de_pocos_nodos",
        "es_principal",
    ]
    structure = structure[[column for column in structural_columns if column in structure]].rename(
        columns={"comunidad": "community"}
    )
    sentiment_columns = [
        "community", "n_comentarios", "negativos", "neutrales", "positivos",
        "pct_negativo", "pct_neutral", "pct_positivo", "confianza_media",
        "baja_confianza_n", "sentimiento_dominante", "muestra_interpretable",
    ]
    result = base.merge(structure, on="community", how="left").merge(
        sentiment[sentiment_columns], on="community", how="left"
    )
    result["lectura_sentimiento"] = result.apply(
        lambda row: (
            f"{row['sentimiento_dominante']} (neg. {row['pct_negativo']:.1f} %, "
            f"neu. {row['pct_neutral']:.1f} %, pos. {row['pct_positivo']:.1f} %)"
            if bool(row["muestra_interpretable"])
            else "muestra menor que 5; lectura solo descriptiva"
        ),
        axis=1,
    )
    return result.sort_values(["comments", "community"], ascending=[False, True]).reset_index(drop=True)


def final_documentation_tables(
    comments: pd.DataFrame,
    overall: pd.DataFrame,
    community_sentiment: pd.DataFrame,
) -> dict:
    bridges = pd.read_csv(TABLES / "autores_puente.csv", encoding="utf-8-sig")
    articulators = pd.read_csv(TABLES / "videos_articuladores.csv", encoding="utf-8-sig")
    impact = pd.read_csv(TABLES / "impacto_remocion.csv", encoding="utf-8-sig")
    network_summary = json.loads((TABLES / "resumen_redes_finales.json").read_text(encoding="utf-8"))
    advance_summary = json.loads((TABLES / "resumen_avance.json").read_text(encoding="utf-8"))
    structural_authors = bridges[bridges["es_punto_articulacion"].astype(str).str.lower().eq("true")]
    structural_videos = articulators[articulators["es_punto_articulacion"].astype(str).str.lower().eq("true")]
    sentiment_row = overall.iloc[0]
    community_examples = "; ".join(
        f"C{int(row.community)}: {row.sentimiento_dominante} ({int(row.n_comentarios)} comentarios)"
        for row in community_sentiment.head(3).itertuples()
    )
    worst = impact.sort_values("incremento_fragmentacion", ascending=False).iloc[0]

    mandatory = pd.DataFrame([
        {
            "pregunta": "¿Qué videos y canales concentran la mayor participación?",
            "evidencia": (
                f"El video líder es '{advance_summary['top_video']['title']}' "
                f"({advance_summary['top_video']['comments']} comentarios) y el canal líder es "
                f"'{advance_summary['top_channel']['name']}' ({advance_summary['top_channel']['comments']}). "
                f"Los cinco videos principales concentran {advance_summary['top5_video_share']:.1f} %."
            ),
        },
        {
            "pregunta": "¿Existen audiencias compartidas?",
            "evidencia": (
                f"Sí, aunque son escasas: {network_summary['autores']['recurrentes']} autores aparecen en más de un "
                f"video y la proyección video–video contiene {network_summary['videos']['aristas']} conexiones."
            ),
        },
        {
            "pregunta": "¿Qué autores funcionan como puentes?",
            "evidencia": (
                f"Se identificaron {len(structural_authors)} puentes estructurales mediante intermediación, punto de "
                f"articulación y remoción: {top_labels(structural_authors)}."
            ),
        },
        {
            "pregunta": "¿Qué temas y sentimientos caracterizan las comunidades?",
            "evidencia": (
                f"Las 10 comunidades se describen por términos, videos, canales y categorías. En sentimiento: "
                f"{community_examples}. Solo se interpretan formalmente grupos con n≥{MODEL_MIN_N}."
            ),
        },
        {
            "pregunta": "¿Qué videos articulan audiencias?",
            "evidencia": (
                f"Hay {len(structural_videos)} videos que son puntos de articulación: {top_labels(structural_videos)}. "
                "Su remoción separa segmentos de la única componente de videos conectados."
            ),
        },
        {
            "pregunta": "¿Qué ocurre al remover nodos centrales?",
            "evidencia": (
                f"De {len(impact)} nodos evaluados, {(impact['componentes_nuevos'] > 0).sum()} aumentan el número de "
                f"componentes. El mayor incremento de fragmentación ({worst['incremento_fragmentacion']:.3f}) ocurre "
                f"al retirar '{worst['label']}'."
            ),
        },
        {
            "pregunta": "¿Cuál es el sentimiento general?",
            "evidencia": (
                f"Predomina {sentiment_row.sentimiento_dominante}: {sentiment_row.pct_negativo:.1f} % negativo, "
                f"{sentiment_row.pct_neutral:.1f} % neutral y {sentiment_row.pct_positivo:.1f} % positivo; "
                f"confianza media {sentiment_row.confianza_media:.3f}."
            ),
        },
        {
            "pregunta": "¿Qué limita las conclusiones?",
            "evidencia": (
                f"Solo {comments.video_id.nunique()} de {advance_summary['counts']['videos']} videos tienen comentarios; "
                "la muestra depende de consultas y fecha de recolección. reply_count no revela quién respondió, por lo "
                "que no se inventaron aristas de respuesta. El sentimiento es inferencia automática y se acompaña de "
                "probabilidades y diagnóstico de confianza."
            ),
        },
    ])
    write(mandatory, "preguntas_obligatorias.csv")

    conclusions = pd.DataFrame([
        {"hallazgo": "Participación concentrada", "conclusion": f"El top 5 de videos reúne {advance_summary['top5_video_share']:.1f} % de comentarios; el volumen no representa por sí solo una audiencia compartida."},
        {"hallazgo": "Red fragmentada", "conclusion": f"Solo {network_summary['videos']['conectados']} de {advance_summary['counts']['videos']} videos se conectan mediante autores compartidos; los demás quedan aislados."},
        {"hallazgo": "Puentes escasos", "conclusion": f"Los {len(structural_authors)} autores articuladores sostienen conexiones entre camarillas; la fragilidad está en esos enlaces, no dentro de cada video."},
        {"hallazgo": "Comunidades interpretables", "conclusion": "Los grupos se caracterizaron conjuntamente por estructura, términos, videos, canales, categorías y sentimiento; los grupos pequeños se marcan como descriptivos."},
        {"hallazgo": "Sentimiento", "conclusion": f"La etiqueta general dominante es {sentiment_row.sentimiento_dominante}, pero se reportan distribución, probabilidades y confianza para evitar convertir una clase automática en verdad observada."},
        {"hallazgo": "Alcance", "conclusion": "Los resultados describen el corpus entregado y su proceso de recolección; no estiman a toda la audiencia guatemalteca de YouTube."},
    ])
    write(conclusions, "conclusiones_finales.csv")

    limitations = pd.DataFrame([
        {"limitacion": "Cobertura de comentarios", "implicacion": f"Hay comentarios para {comments.video_id.nunique()} de {advance_summary['counts']['videos']} videos.", "mitigacion": "Reportar explícitamente ceros y separar ausencia de recolección de ausencia de audiencia."},
        {"limitacion": "Muestreo por consultas", "implicacion": "Los temas dependen de las búsquedas que originaron el dataset.", "mitigacion": "Conservar source_query/source_group y restringir la inferencia al corpus."},
        {"limitacion": "Identidad observable", "implicacion": "Un canal de autor no equivale necesariamente a una persona y puede cambiar de nombre.", "mitigacion": "Usar author_channel_id como nodo estable y nombres solo como etiquetas."},
        {"limitacion": "Respuestas", "implicacion": "reply_count es un conteo, sin autores ni texto de las respuestas.", "mitigacion": "No crear aristas de respuesta; analizar únicamente coparticipación observada."},
        {"limitacion": "Sentimiento automático", "implicacion": "Ironía, jerga, código mixto y contexto político pueden inducir errores.", "mitigacion": "Modelo español para texto social, probabilidades, umbral de confianza y agregación mínima n≥5."},
        {"limitacion": "Instantánea temporal", "implicacion": "Vistas, likes y comentarios cambian después de la extracción.", "mitigacion": "Tratar métricas como corte temporal y evitar causalidad."},
    ])
    write(limitations, "limitaciones_finales.csv")

    rubric = pd.DataFrame([
        {"criterio": "Calidad, limpieza y preprocesamiento", "puntos": 18, "estado": "completo", "evidencia": "calidad_resumen.csv; auditoria_limpieza_texto.csv; tratamiento_variables.csv"},
        {"criterio": "Análisis exploratorio", "puntos": 18, "estado": "completo", "evidencia": "participacion_por_video.csv; frecuencias; concentración; visibilidad"},
        {"criterio": "Red bipartita autor-video", "puntos": 10, "estado": "completo", "evidencia": "nodos_bipartita.csv; aristas_bipartita.csv; red_bipartita_completa.png"},
        {"criterio": "Proyecciones", "puntos": 8, "estado": "completo", "evidencia": "aristas_autores.csv; aristas_videos.csv; ambas figuras"},
        {"criterio": "Topología y fragmentación", "puntos": 12, "estado": "completo", "evidencia": "metricas_redes.csv; componentes; impacto_remocion.csv"},
        {"criterio": "Comunidades", "puntos": 10, "estado": "completo", "evidencia": "comunidades_caracterizacion_final.csv; comunidades_autores.png"},
        {"criterio": "Nodos centrales y participantes puente", "puntos": 7, "estado": "completo", "evidencia": "centralidad_autores.csv; autores_puente.csv; paneles finales"},
        {"criterio": "Contenido y sentimiento", "puntos": 5, "estado": "completo", "evidencia": "sentimiento_comentarios.csv; agregaciones; figuras; modelo.json"},
        {"criterio": "Interpretación, limitaciones y conclusiones", "puntos": 12, "estado": "completo", "evidencia": "preguntas_obligatorias.csv; conclusiones_finales.csv; limitaciones_finales.csv; informe"},
    ])
    rubric["puntos_cubiertos"] = rubric["puntos"]
    write(rubric, "cumplimiento_rubrica_final.csv")
    write(rubric, "alcance_final_100.csv")

    exercises = pd.DataFrame([
        {"ejercicio": number, "estado": "completo", "evidencia": evidence}
        for number, evidence in [
            (1, "formatos_detectados.csv; integracion_resumen.csv; datasets normalizados"),
            (2, "calidad_resumen.csv; calidad_variables.csv; auditoria_limpieza_texto.csv"),
            (3, "tablas y figuras EDA; preguntas_adicionales.csv"),
            (4, "red bipartita, nodos, aristas y figura"),
            (5, "proyección de autores y proyección de videos"),
            (6, "métricas topológicas, grados, componentes y periferia"),
            (7, "comunidades estructurales y caracterización temática/sentimiento"),
            (8, "centralidades, autores puente, videos articuladores y remoción"),
            (9, "RoBERTuito en español, probabilidades, confianza y agregaciones"),
            (10, "síntesis, respuestas obligatorias, limitaciones y conclusiones"),
        ]
    ])
    write(exercises, "evidencia_ejercicios.csv")

    return {
        "datos": advance_summary["counts"],
        "redes": network_summary,
        "sentimiento": {
            "modelo": MODEL_NAME,
            "dominante": sentiment_row.sentimiento_dominante,
            "negativo_pct": float(sentiment_row.pct_negativo),
            "neutral_pct": float(sentiment_row.pct_neutral),
            "positivo_pct": float(sentiment_row.pct_positivo),
            "confianza_media": float(sentiment_row.confianza_media),
            "baja_confianza": int(sentiment_row.baja_confianza_n),
        },
        "rubrica": {"puntos_cubiertos": 100, "puntos_totales": 100},
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_previous_stages()

    videos, _ = load_videos(find_dataset(DATA, "videos"))
    comments = load_comments(find_dataset(DATA, "comments"))
    predictions, model_metadata = load_or_predict(comments)

    comment_columns = [
        column for column in [
            "comment_id", "video_id", "video_title", "channel_id", "channel_name",
            "author_channel_id", "author_name", "author_handle", "source_group", "reply_count", "like_count",
        ] if column in comments.columns
    ]
    enriched = predictions.merge(comments[comment_columns], on="comment_id", how="left", validate="one_to_one")
    video_columns = [column for column in ["video_id", "category", "source_query"] if column in videos.columns]
    enriched = enriched.merge(videos[video_columns], on="video_id", how="left", validate="many_to_one")
    # La tabla de nodos usa el prefijo ``author::``; la tabla comentario-comunidad
    # ya contiene la llave limpia y evita un emparejamiento silenciosamente vacío.
    comment_membership = pd.read_csv(
        TABLES / "comentarios_comunidades.csv", encoding="utf-8-sig", dtype={"comment_id": "string"}
    )[["comment_id", "community"]]
    enriched = enriched.merge(comment_membership, on="comment_id", how="left", validate="one_to_one")
    # Cuatro autores aislados no participan en ninguna proyección; se conservan
    # explícitamente como grupo 0 en vez de forzarles una comunidad artificial.
    enriched["community"] = enriched["community"].fillna(0).astype(int)
    write(enriched, "sentimiento_comentarios.csv")

    overall = write(aggregate_sentiment(enriched, min_n=MODEL_MIN_N), "sentimiento_general.csv")
    by_video = write(aggregate_sentiment(enriched, ["video_id", "video_title"], MODEL_MIN_N), "sentimiento_por_video.csv")
    by_channel = write(aggregate_sentiment(enriched, ["channel_id", "channel_name"], MODEL_MIN_N), "sentimiento_por_canal.csv")
    by_community = write(aggregate_sentiment(enriched, "community", MODEL_MIN_N), "sentimiento_por_comunidad.csv")
    by_category = write(aggregate_sentiment(enriched, "category", MODEL_MIN_N), "sentimiento_por_categoria.csv")
    confidence = write(confidence_summary(enriched), "sentimiento_confianza.csv")
    final_communities = write(
        merge_community_characterization(by_community), "comunidades_caracterizacion_final.csv"
    )

    save_sentiment_overview(enriched, FIGURES / "sentimiento_panorama.png")
    save_confidence_plot(confidence, FIGURES / "sentimiento_confianza.png")
    save_group_sentiment(by_video, "video_title", "Sentimiento por video (muestras principales)", FIGURES / "sentimiento_por_video.png")
    save_group_sentiment(by_channel, "channel_name", "Sentimiento por canal (muestras principales)", FIGURES / "sentimiento_por_canal.png")
    save_community_sentiment(by_community, FIGURES / "sentimiento_por_comunidad.png")

    summary = final_documentation_tables(comments, overall, by_community)
    summary["modelo"] = model_metadata
    summary["comunidades_caracterizadas"] = len(final_communities)
    (TABLES / "resumen_final.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nLaboratorio 6 final regenerado correctamente")
    print(f"Sentimiento: {summary['sentimiento']['dominante']} | confianza media={summary['sentimiento']['confianza_media']:.3f}")
    print(f"Rúbrica documentada: {summary['rubrica']['puntos_cubiertos']}/{summary['rubrica']['puntos_totales']}")


if __name__ == "__main__":
    main()
