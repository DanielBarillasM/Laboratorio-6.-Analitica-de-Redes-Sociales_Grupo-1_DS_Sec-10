"""Preprocesamiento auditable de comentarios en español."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd


SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra", "cual", "cuando",
    "de", "del", "desde", "donde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
    "erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado",
    "estamos", "estan", "estar", "estas", "este", "esto", "estos", "fue", "fueron", "ha", "hace", "han",
    "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "mis", "mucho", "muy",
    "no", "nos", "o", "otra", "otro", "para", "pero", "poco", "por", "porque", "que", "quien", "se", "sea",
    "ser", "si", "sin", "sobre", "son", "su", "sus", "tambien", "te", "tiene", "todo", "tu", "un", "una",
    "uno", "unos", "y", "ya", "yo",
}

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[\w.-]+", flags=re.UNICODE)
HASHTAG_RE = re.compile(r"(?<!\w)#([\wáéíóúüñÁÉÍÓÚÜÑ]+)", flags=re.UNICODE)
TOKEN_RE = re.compile(r"[a-záéíóúüñ]+", flags=re.IGNORECASE)


def _strip_accents_for_stopword(token: str) -> str:
    normalized = unicodedata.normalize("NFD", token)
    return "".join(character for character in normalized if unicodedata.category(character) != "Mn")


def clean_text(value: object) -> str:
    """Limpia texto para frecuencias/redes conservando el original por separado."""

    text = html.unescape("" if value is None or pd.isna(value) else str(value)).lower()
    text = URL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = MENTION_RE.sub(" ", text)
    tokens = TOKEN_RE.findall(text)
    return " ".join(token for token in tokens if _strip_accents_for_stopword(token) not in SPANISH_STOPWORDS)


def add_clean_text(comments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = comments.copy()
    result["texto_limpio"] = result["texto_original"].map(clean_text)
    modified = result["texto_original"].str.strip() != result["texto_limpio"]
    audit = pd.DataFrame([
        {"metrica": "registros_entrada", "valor": len(result)},
        {"metrica": "textos_modificados", "valor": int(modified.sum())},
        {"metrica": "textos_vacios_antes", "valor": int(result["texto_original"].str.strip().eq("").sum())},
        {"metrica": "textos_vacios_despues", "valor": int(result["texto_limpio"].str.strip().eq("").sum())},
        {"metrica": "duplicados_texto_antes", "valor": int(result["texto_original"].duplicated().sum())},
        {"metrica": "duplicados_texto_despues", "valor": int(result["texto_limpio"].duplicated().sum())},
        {"metrica": "registros_eliminados", "valor": 0},
    ])
    return result, audit

