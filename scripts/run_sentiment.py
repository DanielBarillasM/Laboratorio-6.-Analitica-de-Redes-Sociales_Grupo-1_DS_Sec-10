

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab6_social.io import load_comments, load_videos  # noqa: E402
from lab6_social.preprocessing import add_clean_text  # noqa: E402
from lab6_social.sentiment import aggregate_sentiment, predict_sentiment  # noqa: E402

DATA = ROOT / "Data"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


def main() -> None:
    comments = load_comments(DATA / "youtube_comments.csv")
    videos, _ = load_videos(DATA / "youtube_videos.csv")
    comments, _audit = add_clean_text(comments)
    predictions, _metadata = predict_sentiment(comments)
    prediction_columns = [
        "comment_id", "sentimiento", "etiqueta_modelo", "prob_negativo",
        "prob_neutral", "prob_positivo", "confianza", "margen_confianza",
        "baja_confianza",
    ]
    comments = comments.merge(predictions[prediction_columns], on="comment_id", how="left", validate="one_to_one")

    videos_small = videos[["video_id", "title", "channel_id", "channel_name", "category"]]
    merged = comments.merge(videos_small, on="video_id", how="left", suffixes=("", "_video"))

    # Comunidad del autor (ejercicio 7), si la tabla existe.
    community_path = TABLES / "membresia_comunidades.csv"
    if community_path.exists():
        membership = pd.read_csv(community_path)
        membership["author_channel_id"] = membership["node"].str.replace("author::", "", regex=False)
        merged = merged.merge(
            membership[["author_channel_id", "community"]], on="author_channel_id", how="left"
        )

    out_cols = [
        "comment_id", "video_id", "author_channel_id", "channel_id", "channel_name",
        "category", "texto_original", "texto_limpio", "sentimiento",
        "etiqueta_modelo", "prob_negativo", "prob_neutral", "prob_positivo",
        "confianza", "margen_confianza", "baja_confianza",
    ]
    if "community" in merged.columns:
        out_cols.append("community")
    merged[out_cols].to_csv(TABLES / "sentimiento_comentarios.csv", index=False)

    by_video = aggregate_sentiment(merged, ["title"], min_n=3)
    by_video.to_csv(TABLES / "sentimiento_por_video.csv", index=False)

    by_channel = aggregate_sentiment(merged, ["channel_name"], min_n=3)
    by_channel.to_csv(TABLES / "sentimiento_por_canal.csv", index=False)

    if "community" in merged.columns:
        by_community = aggregate_sentiment(merged, "community", min_n=3)
        by_community.to_csv(TABLES / "sentimiento_por_comunidad.csv", index=False)

        # Completa la columna "sentimiento" (placeholder del ejercicio 7) en
        # resumen_comunidades.csv, si esa tabla existe.
        resumen_comunidades_path = TABLES / "resumen_comunidades.csv"
        if resumen_comunidades_path.exists():
            resumen = pd.read_csv(resumen_comunidades_path)
            scores = by_community.set_index("community")["sentimiento_dominante"]
            counts = by_community.set_index("community")["n_comentarios"]

            def _label(community_id: object) -> str:
                try:
                    cid = int(float(community_id))
                except (TypeError, ValueError):
                    return "Sin comentarios con lexico de sentimiento"
                if cid not in scores.index:
                    return "Sin comentarios con lexico de sentimiento"
                tag = scores[cid]
                return f"sentimiento dominante {tag} sobre {int(counts[cid])} comentarios"

            resumen["sentimiento"] = resumen["community"].map(_label)
            resumen.to_csv(resumen_comunidades_path, index=False)

    # Cobertura y distribución general
    con_sentimiento = merged[merged["sentimiento"].notna()]
    resumen = pd.DataFrame([
        {"metrica": "comentarios_totales", "valor": len(merged)},
        {"metrica": "comentarios_con_sentimiento", "valor": len(con_sentimiento)},
        {"metrica": "cobertura_sentimiento_pct", "valor": round(100 * len(con_sentimiento) / len(merged), 1)},
        {"metrica": "positivos", "valor": int((con_sentimiento["sentimiento"] == "positivo").sum())},
        {"metrica": "negativos", "valor": int((con_sentimiento["sentimiento"] == "negativo").sum())},
        {"metrica": "neutrales", "valor": int((con_sentimiento["sentimiento"] == "neutral").sum())},
        {"metrica": "confianza_promedio", "valor": round(con_sentimiento["confianza"].mean(), 3)},
    ])
    resumen.to_csv(TABLES / "sentimiento_resumen.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    con_sentimiento["sentimiento"].value_counts().reindex(
        ["positivo", "neutral", "negativo"]
    ).plot(kind="bar", ax=ax, color=["#16A34A", "#94A3B8", "#DC2626"])
    ax.set_title("Distribución de sentimiento en comentarios (con léxico)")
    ax.set_xlabel("Etiqueta")
    ax.set_ylabel("Comentarios")
    fig.tight_layout()
    fig.savefig(FIGURES / "sentimiento_distribucion.png", dpi=150)
    plt.close(fig)

    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
