import pandas as pd

from lab6_social.preprocessing import add_clean_text, clean_text


def test_cleaning_urls_mentions_hashtags_and_accents():
    result = clean_text("¡Hola @usuario! Mira https://example.com #Guatemala está aquí 😊")
    assert "usuario" not in result
    assert "http" not in result
    assert "guatemala" in result
    assert "aquí" in result


def test_cleaning_audit_does_not_delete_records():
    frame = pd.DataFrame({"texto_original": ["Hola mundo", "https://x.com"]})
    cleaned, audit = add_clean_text(frame)
    assert len(cleaned) == 2
    assert audit.loc[audit["metrica"] == "registros_eliminados", "valor"].iloc[0] == 0


def test_original_is_not_overwritten():
    frame = pd.DataFrame({"texto_original": ["TEXTO ORIGINAL #Tema"]})
    cleaned, _ = add_clean_text(frame)
    assert cleaned.loc[0, "texto_original"] == "TEXTO ORIGINAL #Tema"
    assert cleaned.loc[0, "texto_limpio"] != cleaned.loc[0, "texto_original"]

