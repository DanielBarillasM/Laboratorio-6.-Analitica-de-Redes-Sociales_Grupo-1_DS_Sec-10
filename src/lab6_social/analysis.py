"""Indicadores de calidad, integración y análisis exploratorio."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer


HASHTAG_RE = re.compile(r"(?<!\w)#([\wáéíóúüñÁÉÍÓÚÜÑ]+)", flags=re.UNICODE)


def quality_report(frame: pd.DataFrame, dataset: str, primary_key: str) -> pd.DataFrame:
    constants = [column for column in frame.columns if frame[column].nunique(dropna=False) <= 1]
    numeric = frame.select_dtypes(include="number")
    outlier_parts: list[str] = []
    for column in numeric:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        outlier_parts.append(f"{column}:{count}")
    return pd.DataFrame([
        {"dataset": dataset, "metrica": "filas", "valor": len(frame)},
        {"dataset": dataset, "metrica": "columnas", "valor": frame.shape[1]},
        {"dataset": dataset, "metrica": "celdas_faltantes", "valor": int(frame.isna().sum().sum())},
        {"dataset": dataset, "metrica": "filas_duplicadas", "valor": int(frame.duplicated().sum())},
        {"dataset": dataset, "metrica": f"{primary_key}_faltantes", "valor": int(frame[primary_key].isna().sum())},
        {"dataset": dataset, "metrica": f"{primary_key}_duplicados", "valor": int(frame[primary_key].duplicated().sum())},
        {"dataset": dataset, "metrica": "variables_constantes", "valor": ", ".join(constants) or "ninguna"},
        {"dataset": dataset, "metrica": "atipicos_iqr", "valor": "; ".join(outlier_parts) or "sin numéricas"},
    ])


def missing_report(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": dataset,
        "variable": frame.columns,
        "tipo": frame.dtypes.astype(str).values,
        "faltantes": frame.isna().sum().values,
        "faltantes_pct": (100 * frame.isna().mean()).round(3).values,
        "unicos": [frame[column].nunique(dropna=True) for column in frame.columns],
    })


def consistency_report(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    joined = comments.merge(
        videos[["video_id", "title", "channel_id", "channel_name"]],
        on="video_id", how="left", suffixes=("_comment", "_video"), indicator=True,
    )
    checks = [
        ("videos: channel_id identifica un solo channel_name", int((videos.groupby("channel_id")["channel_name"].nunique() > 1).sum())),
        ("videos: channel_id identifica un solo channel_handle", int((videos.groupby("channel_id")["channel_handle"].nunique() > 1).sum())),
        ("videos: owner_handle coincide con channel_handle", int((videos["owner_handle"] != videos["channel_handle"]).sum())),
        ("videos: upload_date coincide con publish_date", int((videos["upload_date"] != videos["publish_date"]).sum())),
        ("comentarios sin video asociado", int((joined["_merge"] != "both").sum())),
        ("comentarios con channel_id inconsistente", int((joined["channel_id_comment"] != joined["channel_id_video"]).sum())),
        ("comentarios con título inconsistente", int((joined["video_title"] != joined["title"]).sum())),
    ]
    return pd.DataFrame(checks, columns=["verificacion", "inconsistencias"])


def integrate_data(videos: pd.DataFrame, comments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_fields = ["video_id", "title", "channel_id", "channel_name", "category", "view_count", "source_query"]
    merged = comments.merge(
        videos[video_fields], on="video_id", how="left", suffixes=("_comment", "_video"),
        validate="many_to_one", indicator=True,
    )
    summary = pd.DataFrame([
        {"metrica": "comentarios_totales", "valor": len(comments)},
        {"metrica": "comentarios_asociados", "valor": int((merged["_merge"] == "both").sum())},
        {"metrica": "comentarios_sin_video", "valor": int((merged["_merge"] != "both").sum())},
        {"metrica": "porcentaje_asociado", "valor": float(100 * (merged["_merge"] == "both").mean())},
        {"metrica": "videos_con_comentarios", "valor": comments["video_id"].nunique()},
        {"metrica": "videos_sin_comentarios", "valor": int((~videos["video_id"].isin(comments["video_id"])).sum())},
    ])
    return merged.drop(columns="_merge"), summary


def ngram_frequencies(texts: pd.Series, n: int, top_k: int = 25) -> pd.DataFrame:
    clean = texts.fillna("").astype(str)
    vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(clean)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    output = pd.DataFrame({"ngram": vectorizer.get_feature_names_out(), "frecuencia": counts})
    output = output.sort_values(["frecuencia", "ngram"], ascending=[False, True]).head(top_k)
    output["probabilidad"] = output["frecuencia"] / counts.sum()
    output["n"] = n
    return output.reset_index(drop=True)


def hashtag_frequencies(texts: pd.Series, top_k: int = 25) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        counts.update(tag.lower() for tag in HASHTAG_RE.findall(text))
    total = sum(counts.values())
    return pd.DataFrame([
        {"hashtag": tag, "frecuencia": count, "probabilidad": count / total if total else 0.0}
        for tag, count in counts.most_common(top_k)
    ])


def participation_tables(videos: pd.DataFrame, comments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_video = comments.groupby("video_id").agg(
        comentarios=("comment_id", "count"), autores_unicos=("author_channel_id", "nunique"),
        likes=("like_count", "sum"), respuestas=("reply_count", "sum"),
    ).reset_index()
    per_video = videos[["video_id", "title", "channel_id", "channel_name", "category", "view_count"]].merge(
        per_video, on="video_id", how="left"
    )
    for column in ("comentarios", "autores_unicos", "likes", "respuestas"):
        per_video[column] = per_video[column].fillna(0).astype(int)
    per_video["comentarios_por_1000_vistas"] = np.where(
        per_video["view_count"] > 0, 1000 * per_video["comentarios"] / per_video["view_count"], np.nan
    )

    per_channel = per_video.groupby(["channel_id", "channel_name"], dropna=False).agg(
        videos=("video_id", "nunique"), visualizaciones=("view_count", "sum"),
        comentarios=("comentarios", "sum"), autores_video_suma=("autores_unicos", "sum"),
    ).reset_index()
    author_channel = comments[["author_channel_id", "video_id"]].merge(
        videos[["video_id", "channel_id"]], on="video_id", how="left"
    )
    unique_authors = author_channel.groupby("channel_id")["author_channel_id"].nunique()
    per_channel["autores_unicos"] = per_channel["channel_id"].map(unique_authors).fillna(0).astype(int)
    per_channel["comentarios_por_1000_vistas"] = np.where(
        per_channel["visualizaciones"] > 0,
        1000 * per_channel["comentarios"] / per_channel["visualizaciones"], np.nan,
    )
    return (
        per_video.sort_values("comentarios", ascending=False).reset_index(drop=True),
        per_channel.sort_values("comentarios", ascending=False).reset_index(drop=True),
    )


def concentration_table(per_video: pd.DataFrame, per_channel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for unit, frame in (("video", per_video), ("canal", per_channel)):
        values = frame["comentarios"].sort_values(ascending=False)
        total = values.sum()
        for top_n in (1, 3, 5, 10):
            rows.append({
                "unidad": unit, "top_n": top_n,
                "comentarios_acumulados": int(values.head(top_n).sum()),
                "participacion_pct": float(100 * values.head(top_n).sum() / total) if total else 0.0,
            })
    return pd.DataFrame(rows)


def popularity_association(per_video: pd.DataFrame) -> pd.DataFrame:
    rho, p_value = spearmanr(per_video["view_count"], per_video["comentarios"])
    active = per_video[per_video["comentarios"] > 0]
    rho_active, p_active = spearmanr(active["view_count"], active["comentarios"])
    return pd.DataFrame([
        {"universo": "todos_los_videos", "n": len(per_video), "spearman_rho": rho, "p_value": p_value},
        {"universo": "videos_con_comentarios", "n": len(active), "spearman_rho": rho_active, "p_value": p_active},
    ])


def author_participation(comments: pd.DataFrame) -> pd.DataFrame:
    return comments.groupby(["author_channel_id", "author_name", "author_handle"], dropna=False).agg(
        comentarios=("comment_id", "count"), videos=("video_id", "nunique"),
        likes_recibidos=("like_count", "sum"), respuestas_recibidas=("reply_count", "sum"),
    ).reset_index().sort_values(["videos", "comentarios"], ascending=False)

