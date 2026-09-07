from pathlib import Path

import pandas as pd
import pytest

from lab6_social.sentiment import (
    MODEL_NAME,
    aggregate_sentiment,
    cache_is_compatible,
    confidence_summary,
    prepare_model_text,
    text_fingerprint,
    validate_probabilities,
)


def predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {"comment_id": "1", "sentimiento": "positivo", "prob_negativo": 0.05, "prob_neutral": 0.15, "prob_positivo": 0.80, "confianza": 0.80, "baja_confianza": False, "video_id": "v1"},
        {"comment_id": "2", "sentimiento": "neutral", "prob_negativo": 0.10, "prob_neutral": 0.70, "prob_positivo": 0.20, "confianza": 0.70, "baja_confianza": False, "video_id": "v1"},
        {"comment_id": "3", "sentimiento": "negativo", "prob_negativo": 0.55, "prob_neutral": 0.30, "prob_positivo": 0.15, "confianza": 0.55, "baja_confianza": True, "video_id": "v2"},
    ])


def test_prepare_model_text_preserves_negation_emoji_and_punctuation():
    text = prepare_model_text("  No me gusta 😡!!!\r\n  Nunca. ")
    assert text == "No me gusta 😡!!! Nunca."


def test_fingerprint_is_deterministic_and_depends_on_id():
    assert text_fingerprint("1", "hola") == text_fingerprint("1", "hola")
    assert text_fingerprint("1", "hola") != text_fingerprint("2", "hola")


def test_probability_validation_accepts_valid_rows():
    validate_probabilities(predictions())


def test_probability_validation_rejects_invalid_sum():
    frame = predictions()
    frame.loc[0, "prob_positivo"] = 0.20
    with pytest.raises(ValueError, match="sumar 1"):
        validate_probabilities(frame)


def test_aggregate_reports_counts_percentages_and_minimum_sample():
    result = aggregate_sentiment(predictions(), "video_id", min_n=2).set_index("video_id")
    assert result.loc["v1", "n_comentarios"] == 2
    assert result.loc["v1", "pct_positivo"] == pytest.approx(50.0)
    assert bool(result.loc["v1", "muestra_interpretable"]) is True
    assert bool(result.loc["v2", "muestra_interpretable"]) is False


def test_general_aggregation_has_one_row():
    result = aggregate_sentiment(predictions(), min_n=2)
    assert result.loc[0, "grupo"] == "general"
    assert result.loc[0, "n_comentarios"] == 3
    assert result.loc[0, "sentimiento_dominante"] == "negativo"


def test_confidence_bands_include_every_comment():
    summary = confidence_summary(predictions())
    assert summary["comentarios"].sum() == 3
    assert summary["porcentaje"].sum() == pytest.approx(100.0)


def test_cache_compatibility_checks_model_ids_and_text(tmp_path: Path):
    comments = pd.DataFrame({"comment_id": ["1", "2"], "texto_original": ["hola", "adiós"]})
    cache = pd.DataFrame({
        "comment_id": ["1", "2"],
        "texto_sha256": [text_fingerprint("1", "hola"), text_fingerprint("2", "adiós")],
        "modelo": [MODEL_NAME, MODEL_NAME],
    })
    path = tmp_path / "cache.csv"
    cache.to_csv(path, index=False, encoding="utf-8-sig")
    assert cache_is_compatible(path, comments)
    comments.loc[1, "texto_original"] = "otro"
    assert not cache_is_compatible(path, comments)


def test_aggregation_is_deterministic():
    left = aggregate_sentiment(predictions(), "video_id", min_n=1)
    right = aggregate_sentiment(predictions().sample(frac=1, random_state=42), "video_id", min_n=1)
    pd.testing.assert_frame_equal(left, right)

