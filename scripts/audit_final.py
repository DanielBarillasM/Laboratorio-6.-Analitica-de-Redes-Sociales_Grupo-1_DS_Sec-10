"""Auditoría local de integridad para la entrega final del Laboratorio 6."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def main() -> None:
    videos = pd.read_csv(ROOT / "Data" / "youtube_videos.csv")
    comments = pd.read_csv(ROOT / "Data" / "youtube_comments.csv")
    require(videos.shape == (293, 20), "youtube_videos.csv: 293 × 20")
    require(comments.shape == (406, 17), "youtube_comments.csv: 406 × 17")
    require(videos["video_id"].is_unique and comments["comment_id"].is_unique, "llaves primarias únicas")
    require(comments["video_id"].isin(videos["video_id"]).all(), "integración comentario–video al 100 %")

    mandatory = [
        "metricas_redes.csv", "autores_puente.csv", "videos_articuladores.csv",
        "impacto_remocion.csv", "comunidades_caracterizacion_final.csv",
        "sentimiento_comentarios.csv", "sentimiento_general.csv",
        "sentimiento_por_video.csv", "sentimiento_por_canal.csv",
        "sentimiento_por_comunidad.csv", "cumplimiento_rubrica_final.csv",
        "preguntas_obligatorias.csv", "conclusiones_finales.csv", "limitaciones_finales.csv",
    ]
    require(all((TABLES / name).exists() for name in mandatory), "tablas obligatorias presentes")

    sentiment = pd.read_csv(TABLES / "sentimiento_comentarios.csv")
    probabilities = sentiment[["prob_negativo", "prob_neutral", "prob_positivo"]]
    require(len(sentiment) == len(comments), "una predicción por comentario")
    require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), "probabilidades de sentimiento válidas")
    require(set(sentiment["sentimiento"]) <= {"negativo", "neutral", "positivo"}, "etiquetas de sentimiento válidas")
    require(sentiment["community"].notna().all(), "aislados conservados como grupo 0")
    require((sentiment["community"] == 0).sum() == 4, "cuatro comentarios/autores aislados asignados al grupo 0")
    for filename in ("sentimiento_por_video.csv", "sentimiento_por_canal.csv", "sentimiento_por_comunidad.csv"):
        aggregation = pd.read_csv(TABLES / filename)
        require(aggregation["minimo_muestra"].eq(5).all(), f"muestra mínima n=5 en {filename}")

    networks = json.loads((TABLES / "resumen_redes_finales.json").read_text(encoding="utf-8"))
    require(networks["autores"]["puentes_estructurales"] == 7, "siete autores puente estructurales")
    require(networks["videos"]["articuladores"] == 5, "cinco videos articuladores")
    require(networks["comunidades"]["total"] == 10, "diez comunidades detectadas")

    rubric = pd.read_csv(TABLES / "cumplimiento_rubrica_final.csv")
    require(rubric["estado"].eq("completo").all(), "todos los criterios locales figuran completos")
    require(rubric["puntos"].sum() == rubric["puntos_cubiertos"].sum() == 100, "trazabilidad local 100/100")

    notebook_path = ROOT / "notebooks" / "Lab6_Analitica_Redes_Sociales.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    require(code_cells and all(cell.execution_count is not None for cell in code_cells), "notebook final completamente ejecutado")
    require(not errors, "notebook sin errores de ejecución")
    report_source = (ROOT / "reports" / "informe_final.tex").read_text(encoding="utf-8")
    stale_claims = ("74 de 100", "cobertura del 36.9", "se reserva para", "sentimiento léxico")
    require(not any(claim in report_source for claim in stale_claims), "informe sin afirmaciones obsoletas del avance")
    require("RoBERTuito" in report_source and "100\\% del corpus" in report_source, "método y cobertura final documentados")
    require((ROOT / "reports" / "informe_final.pdf").stat().st_size > 100_000, "informe PDF final compilado")
    require((ROOT / "entrega" / "Ficha_Repositorio_Laboratorio_6.docx").exists(), "ficha DOCX del repositorio presente")
    print("\nAuditoría local completada: 10 ejercicios y 100 puntos con evidencia reproducible.")
    print("Control externo pendiente de quien entrega: adjuntar en Canvas el enlace colaborativo con historial.")


if __name__ == "__main__":
    main()
