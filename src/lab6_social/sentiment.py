"""Sentimiento en español y agregaciones reproducibles para comentarios."""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"
MODEL_TASK = "sentimiento en español (POS/NEU/NEG)"
LABELS = ("NEG", "NEU", "POS")
SPANISH_LABELS = {"NEG": "negativo", "NEU": "neutral", "POS": "positivo"}


def prepare_model_text(value: object) -> str:
    """Normaliza solo HTML y espacios, conservando negación, emojis y puntuación."""

    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def text_fingerprint(comment_id: object, text: object) -> str:
    payload = f"{comment_id}\0{prepare_model_text(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_probabilities(frame: pd.DataFrame, tolerance: float = 1e-5) -> None:
    required = {"prob_negativo", "prob_neutral", "prob_positivo", "sentimiento", "confianza"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas de sentimiento: {sorted(missing)}")
    probabilities = frame[["prob_negativo", "prob_neutral", "prob_positivo"]].astype(float)
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=tolerance):
        raise ValueError("Las probabilidades de cada comentario deben sumar 1.")
    if not frame["sentimiento"].isin(SPANISH_LABELS.values()).all():
        raise ValueError("Se encontraron etiquetas de sentimiento no válidas.")


def cache_is_compatible(cache_path: Path, comments: pd.DataFrame, model_name: str = MODEL_NAME) -> bool:
    """Comprueba que la caché corresponde al mismo modelo, IDs y textos originales."""

    if not cache_path.exists():
        return False
    try:
        cached = pd.read_csv(cache_path, encoding="utf-8-sig", dtype={"comment_id": "string"})
    except Exception:
        return False
    required = {"comment_id", "texto_sha256", "modelo"}
    if not required.issubset(cached.columns) or len(cached) != len(comments):
        return False
    expected = pd.DataFrame({
        "comment_id": comments["comment_id"].astype("string"),
        "texto_sha256": [
            text_fingerprint(comment_id, text)
            for comment_id, text in zip(comments["comment_id"], comments["texto_original"])
        ],
    }).sort_values("comment_id").reset_index(drop=True)
    observed = cached[["comment_id", "texto_sha256"]].sort_values("comment_id").reset_index(drop=True)
    return bool((cached["modelo"] == model_name).all() and expected.equals(observed))


def predict_sentiment(
    comments: pd.DataFrame,
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 128,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    """Clasifica comentarios con RoBERTuito y conserva probabilidades por clase."""

    required = {"comment_id", "texto_original"}
    missing = required.difference(comments.columns)
    if missing:
        raise ValueError(f"Faltan columnas para inferencia: {sorted(missing)}")

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(42)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(target_device)
    model.eval()

    id_to_label = {int(index): str(label).upper() for index, label in model.config.id2label.items()}
    if set(id_to_label.values()) != set(LABELS):
        raise ValueError(f"Etiquetas inesperadas en el modelo: {id_to_label}")

    texts = comments["texto_original"].map(prepare_model_text).tolist()
    records: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(target_device) for name, tensor in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
            for offset, values in enumerate(probabilities):
                row_index = start + offset
                probability_by_label = {id_to_label[index]: float(value) for index, value in enumerate(values)}
                winner = max(LABELS, key=lambda label: (probability_by_label[label], -LABELS.index(label)))
                ordered = sorted(probability_by_label.values(), reverse=True)
                confidence = probability_by_label[winner]
                records.append({
                    "comment_id": str(comments.iloc[row_index]["comment_id"]),
                    "texto_original": comments.iloc[row_index]["texto_original"],
                    "texto_modelo": batch[offset],
                    "texto_sha256": text_fingerprint(comments.iloc[row_index]["comment_id"], batch[offset]),
                    "sentimiento": SPANISH_LABELS[winner],
                    "etiqueta_modelo": winner,
                    "prob_negativo": probability_by_label["NEG"],
                    "prob_neutral": probability_by_label["NEU"],
                    "prob_positivo": probability_by_label["POS"],
                    "confianza": confidence,
                    "margen_confianza": ordered[0] - ordered[1],
                    "baja_confianza": confidence < 0.60,
                    "modelo": model_name,
                    "revision_modelo": getattr(model.config, "_commit_hash", None) or "main",
                    "transformers_version": transformers.__version__,
                    "max_length": max_length,
                })
    result = pd.DataFrame(records)
    validate_probabilities(result)
    metadata = {
        "modelo": model_name,
        "revision": str(records[0]["revision_modelo"]) if records else "main",
        "transformers": transformers.__version__,
        "dispositivo": target_device,
        "batch_size": batch_size,
        "max_length": max_length,
        "comentarios": len(result),
    }
    return result, metadata


def aggregate_sentiment(
    frame: pd.DataFrame,
    group_columns: str | Sequence[str] | None = None,
    min_n: int = 5,
) -> pd.DataFrame:
    """Agrega etiquetas y probabilidades, marcando grupos con muestra suficiente."""

    validate_probabilities(frame)
    if group_columns is None:
        groups: list[str] = []
    elif isinstance(group_columns, str):
        groups = [group_columns]
    else:
        groups = list(group_columns)

    work = frame.copy()
    work["__total"] = 1
    for label in SPANISH_LABELS.values():
        work[f"__{label}"] = (work["sentimiento"] == label).astype(int)

    aggregations = {
        "n_comentarios": ("__total", "sum"),
        "negativos": ("__negativo", "sum"),
        "neutrales": ("__neutral", "sum"),
        "positivos": ("__positivo", "sum"),
        "prob_negativo_media": ("prob_negativo", "mean"),
        "prob_neutral_media": ("prob_neutral", "mean"),
        "prob_positivo_media": ("prob_positivo", "mean"),
        "confianza_media": ("confianza", "mean"),
        "baja_confianza_n": ("baja_confianza", "sum"),
    }
    if groups:
        result = work.groupby(groups, dropna=False).agg(**aggregations).reset_index()
    else:
        values = {name: getattr(work[column], operation)() for name, (column, operation) in aggregations.items()}
        result = pd.DataFrame([values])
        result.insert(0, "grupo", "general")

    count_names = {"negativo": "negativos", "neutral": "neutrales", "positivo": "positivos"}
    for label, count_column in count_names.items():
        result[f"pct_{label}"] = 100 * result[count_column] / result["n_comentarios"]
    count_columns = ["negativos", "neutrales", "positivos"]
    dominant_names = {
        "negativos": "negativo",
        "neutrales": "neutral",
        "positivos": "positivo",
    }
    result["sentimiento_dominante"] = result[count_columns].idxmax(axis=1).map(dominant_names)
    result["muestra_interpretable"] = result["n_comentarios"] >= min_n
    result["minimo_muestra"] = min_n
    sort_columns = ["n_comentarios", *groups]
    ascending = [False, *([True] * len(groups))]
    return result.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def confidence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume confianza en tres bandas declaradas de antemano."""

    validate_probabilities(frame)
    work = frame.copy()
    work["nivel_confianza"] = pd.cut(
        work["confianza"],
        bins=[0.0, 0.60, 0.80, 1.000001],
        labels=["baja (<0.60)", "media (0.60–0.80)", "alta (>0.80)"],
        include_lowest=True,
        right=False,
    )
    result = work.groupby("nivel_confianza", observed=False).size().rename("comentarios").reset_index()
    result["porcentaje"] = 100 * result["comentarios"] / len(work) if len(work) else 0.0
    return result
