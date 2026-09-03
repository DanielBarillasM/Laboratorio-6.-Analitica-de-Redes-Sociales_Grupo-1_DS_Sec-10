"""Carga tolerante a extensiones incorrectas y normalización de identificadores."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


EXCEL_SIGNATURES = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")


def detect_table_format(path: str | Path) -> str:
    """Detecta CSV o Excel por la firma binaria, no solo por la extensión."""

    path = Path(path)
    with path.open("rb") as stream:
        signature = stream.read(8)
    return "excel" if signature.startswith(EXCEL_SIGNATURES) else "csv"


def read_table(path: str | Path) -> pd.DataFrame:
    """Lee CSV UTF-8/Windows o libros Excel incluso con extensión equivocada."""

    path = Path(path)
    if detect_table_format(path) == "excel":
        with path.open("rb") as stream:
            return pd.read_excel(stream, engine="openpyxl")

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("No fue posible decodificar el CSV. " + " | ".join(errors))


def _normalize_identifier(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _video_id_from_url(url: object) -> str | None:
    if pd.isna(url):
        return None
    parsed = urlparse(str(url).strip())
    query_id = parse_qs(parsed.query).get("v", [None])[0]
    if query_id:
        return query_id.strip()
    match = re.search(r"youtu\.be/([^?&/]+)", str(url))
    return match.group(1) if match else None


def repair_video_ids(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repara IDs perdidos por fórmulas de Excel usando la URL verificable."""

    result = frame.copy()
    result["video_id"] = _normalize_identifier(result["video_id"])
    formula_like = result["video_id"].str.startswith("=", na=False)
    result.loc[formula_like, "video_id"] = result.loc[formula_like, "video_id"].str[1:]
    missing_before = result["video_id"].isna()
    recovered = result.loc[missing_before, "video_url"].map(_video_id_from_url)
    result.loc[missing_before, "video_id"] = recovered.astype("string")
    audit = pd.DataFrame(
        {
            "row_index": result.index[missing_before],
            "video_url": frame.loc[missing_before, "video_url"].astype("string"),
            "video_id_recuperado": result.loc[missing_before, "video_id"],
            "metodo": "parámetro v de video_url",
        }
    )
    return result, audit.reset_index(drop=True)


def load_videos(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = read_table(path)
    required = {"video_id", "title", "channel_id", "video_url", "view_count"}
    missing = required.difference(videos.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en videos: {sorted(missing)}")
    videos, repairs = repair_video_ids(videos)
    for column in ("channel_id", "channel_name", "channel_handle", "owner_handle"):
        if column in videos:
            videos[column] = _normalize_identifier(videos[column])
    videos["publish_date"] = pd.to_datetime(videos["publish_date"], errors="coerce", utc=True)
    videos["upload_date"] = pd.to_datetime(videos["upload_date"], errors="coerce", utc=True)
    videos["view_count"] = pd.to_numeric(videos["view_count"], errors="coerce")
    return videos, repairs


def parse_count(value: object) -> float:
    """Convierte conteos con espacios, separadores y sufijos K/M/mil/millón."""

    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "-", "n/a"}:
        return 0.0
    text = text.replace("vistas", "").replace("vista", "").strip()
    multiplier = 1.0
    suffixes = {
        "k": 1_000.0,
        "mil": 1_000.0,
        "m": 1_000_000.0,
        "millón": 1_000_000.0,
        "millones": 1_000_000.0,
    }
    for suffix, factor in suffixes.items():
        if text.endswith(suffix):
            multiplier = factor
            text = text[: -len(suffix)].strip()
            break
    if multiplier > 1 and "," in text and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        return float(text) * multiplier if text else 0.0
    except ValueError:
        return np.nan


def load_comments(path: str | Path) -> pd.DataFrame:
    comments = read_table(path)
    required = {"video_id", "comment_id", "author_channel_id", "text"}
    missing = required.difference(comments.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en comentarios: {sorted(missing)}")
    for column in ("video_id", "comment_id", "channel_id", "author_channel_id", "author_handle"):
        if column in comments:
            comments[column] = _normalize_identifier(comments[column])
    comments["like_count"] = comments["like_count_text"].map(parse_count).astype("Int64")
    comments["reply_count"] = pd.to_numeric(comments["reply_count"], errors="coerce").fillna(0).astype(int)
    comments["texto_original"] = comments["text"].astype("string").fillna("")
    return comments


def find_dataset(data_dir: str | Path, kind: str) -> Path:
    """Encuentra el archivo de videos o comentarios sin depender de su sufijo."""

    data_dir = Path(data_dir)
    candidates = sorted(path for path in data_dir.iterdir() if path.is_file() and kind.lower() in path.name.lower())
    if not candidates:
        raise FileNotFoundError(f"No se encontró un archivo de {kind} en {data_dir}")
    return candidates[0]

