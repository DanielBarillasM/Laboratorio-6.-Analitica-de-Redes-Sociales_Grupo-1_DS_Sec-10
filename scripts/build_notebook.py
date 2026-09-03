"""Construye y ejecuta el notebook narrativo del avance."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((TABLES / "resumen_avance.json").read_text(encoding="utf-8"))
    counts = summary["counts"]
    bip = next(item for item in summary["networks"] if item["red"] == "bipartita_autor_video")
    authors = next(item for item in summary["networks"] if item["red"] == "proyeccion_autores")
    video_format = summary["formats"]["videos"].upper()
    comment_format = summary["formats"]["comments"].upper()
    repair_note = (
        "No fue necesario reconstruir ningún `video_id`."
        if counts["repaired_video_ids"] == 0
        else f"Se reconstruyeron **{counts['repaired_video_ids']} `video_id`** desde sus URL."
    )

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "lab6": {"entrega": "avance", "porcentaje": 75, "random_state": 42},
    }
    notebook["cells"] = [
        markdown("""
<style>
:root{--navy:#102a43;--blue:#2563eb;--teal:#0f9d91;--orange:#f59e0b;--soft:#f1f5f9;--ink:#243b53}
.hero{padding:34px;border-radius:20px;background:linear-gradient(135deg,var(--navy),#1d4ed8);color:white;box-shadow:0 12px 28px #102a4333;margin-bottom:22px}
.hero h1{font-size:2.25rem;margin:.3rem 0}.hero p{font-size:1.03rem;opacity:.94}.tag{display:inline-block;padding:6px 12px;border-radius:999px;background:#ffffff22;border:1px solid #ffffff55;font-weight:700}
.section{border-left:6px solid var(--teal);padding:8px 16px;margin:28px 0 14px;background:linear-gradient(90deg,#e6fffa,white);border-radius:0 12px 12px 0}.section h2{color:var(--navy);margin:.2rem 0}
.insight,.warning,.method{padding:14px 17px;border-radius:12px;margin:14px 0}.insight{background:#ecfdf5;border:1px solid #6ee7b7}.warning{background:#fff7ed;border:1px solid #fdba74}.method{background:#eff6ff;border:1px solid #93c5fd}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}.metric{background:white;border:1px solid #dbeafe;border-radius:12px;padding:13px;text-align:center;box-shadow:0 4px 12px #102a4311}.metric b{font-size:1.35rem;color:var(--blue)}
.footer{margin-top:32px;padding:18px;text-align:center;border-top:1px solid #cbd5e1;color:#52667a}
</style>
<div class="hero"><span class="tag">AVANCE · 75%</span><h1>Laboratorio 6 · Analítica de Redes Sociales</h1><p>Participación, copresencia y comunidades en una muestra de YouTube</p><p><b>Universidad del Valle de Guatemala</b> · CC3084 Data Science · Sección 10 · Grupo 1</p><p>Jorge Gabriel Palacios Sales — 231385 · Pablo Daniel Barillas Moreno — 22193 · Roberto Emiliano Otoniel — 23968</p></div>
"""),
        markdown("""<div class="section"><h2>1 · Ejecución integral y reproducible</h2></div>
La primera celda ejecuta directamente todo `scripts/run_advance.py`: carga e integración, auditoría de calidad, limpieza, EDA, construcción de redes, proyecciones, topología, comunidades, tablas y figuras. Por tanto, basta usar **Run All / Ejecutar todo** desde la raíz del repositorio o desde `/notebooks`. La carga inspecciona la firma binaria para aceptar CSV y Excel aunque la extensión sea incorrecta; los datos originales permanecen intactos."""),
        code("""from pathlib import Path
import json
import runpy
import sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "Data").exists():
    ROOT = ROOT.parent
if not (ROOT / "Data").exists() or not (ROOT / "scripts" / "run_advance.py").exists():
    raise FileNotFoundError("Ejecute el notebook dentro del repositorio, desde la raíz o desde /notebooks.")
sys.path.insert(0, str(ROOT / "src"))

print("Ejecutando el pipeline completo del Laboratorio 6...")
runpy.run_path(str(ROOT / "scripts" / "run_advance.py"), run_name="__main__")
print("Pipeline regenerado correctamente.")
print()

TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
summary = json.loads((TABLES / "resumen_avance.json").read_text(encoding="utf-8"))
STYLES = [
    {"selector":"th", "props":[("background-color","#102a43"),("color","white"),("font-weight","700")]},
    {"selector":"td", "props":[("border-bottom","1px solid #dbeafe"),("padding","7px")]},
    {"selector":"tr:nth-child(even)", "props":[("background-color","#f8fafc")]},
]
def show_csv(name, rows=10):
    frame = pd.read_csv(TABLES / name)
    table_uuid = name.replace(".", "_").replace("-", "_")
    display(frame.head(rows).style.set_uuid(table_uuid).set_table_styles(STYLES).hide(axis="index"))
print("Artefactos recién generados cargados correctamente.")"""),
        markdown(f"""<div class="section"><h2>2 · Datos e integración</h2></div>
<div class="metric-grid"><div class="metric"><b>{counts['videos']}</b><br>videos</div><div class="metric"><b>{counts['channels']}</b><br>canales</div><div class="metric"><b>{counts['comments']}</b><br>comentarios</div><div class="metric"><b>{counts['authors']}</b><br>autores</div></div>
Los comentarios se integraron por `video_id`: **{summary['integration_pct']:.1f}%** obtuvo coincidencia. Los formatos reales detectados fueron **{video_format}** para videos y **{comment_format}** para comentarios. {repair_note}"""),
        code("show_csv('formatos_detectados.csv'); show_csv('integracion_resumen.csv'); show_csv('video_ids_recuperados.csv')"),
        markdown("""<div class="section"><h2>3 · Calidad y preprocesamiento</h2></div>
Se conservaron IDs como identificadores y nombres/handles solo como etiquetas. `texto_original` permanece para auditoría y sentimiento; `texto_limpio` normaliza minúsculas, HTML, URL, hashtags, menciones, puntuación, números, stopwords y emojis. No se aplicó lematización sin un modelo morfosintáctico validado para español."""),
        code("show_csv('calidad_resumen.csv', 20); show_csv('auditoria_limpieza_texto.csv'); show_csv('ejemplos_limpieza.csv', 5)"),
        markdown("""<div class="warning"><b>Variables delicadas.</b> `viewer_rating` está completamente vacío, `is_pinned` es constante y `reply_count` no revela quién respondió. Por ello, las respuestas nunca se convierten en aristas entre usuarios.</div>"""),
        markdown("""<div class="section"><h2>4 · Análisis exploratorio</h2></div>"""),
        code("display(Image(filename=str(FIGURES / 'eda_panorama.png'), width=1000)); display(Image(filename=str(FIGURES / 'frecuencias_contenido.png'), width=1000))"),
        markdown(f"""<div class="insight"><b>Concentración.</b> El video con más participación fue <i>{summary['top_video']['title']}</i>, con {summary['top_video']['comments']} comentarios. El canal líder fue {summary['top_channel']['name']} con {summary['top_channel']['comments']} comentarios. Los cinco videos principales concentran {summary['top5_video_share']:.1f}% de todos los comentarios.</div>"""),
        code("display(Image(filename=str(FIGURES / 'concentracion_participacion.png'), width=800)); display(Image(filename=str(FIGURES / 'visibilidad_vs_participacion.png'), width=800)); show_csv('asociacion_visibilidad_participacion.csv')"),
        markdown(f"""La correlación de Spearman entre visualizaciones y comentarios para todos los videos es **{summary['spearman_all']:.3f}**. Esto describe una asociación débil en la muestra; no prueba causalidad y está condicionada por la cobertura desigual de comentarios."""),
        markdown("""<div class="section"><h2>5 · Red bipartita autor–video</h2></div>
Una arista indica que un autor publicó al menos un comentario en un video; su peso es el número de comentarios de ese autor en ese video. No representa amistad, respuesta directa ni aprobación."""),
        code("display(Image(filename=str(FIGURES / 'red_bipartita_completa.png'), width=1100)); show_csv('metricas_redes.csv')"),
        markdown(f"""La red completa conserva videos sin comentarios: contiene **{bip['nodos']} nodos**, **{bip['aristas']} aristas** y **{bip['nodos_aislados']} aislados**. La componente mayor reúne {bip['componente_mayor_pct']:.1f}% de los nodos."""),
        markdown("""<div class="section"><h2>6 · Proyecciones y topología</h2></div>
En la proyección autor–autor el peso cuenta videos compartidos. En la proyección video–video cuenta autores compartidos. La primera aproxima copresencia de audiencias; la segunda, solapamiento entre contenidos."""),
        code("display(Image(filename=str(FIGURES / 'proyeccion_autores.png'), width=850)); display(Image(filename=str(FIGURES / 'proyeccion_videos.png'), width=850)); display(Image(filename=str(FIGURES / 'distribuciones_grado.png'), width=1000))"),
        markdown(f"""La proyección de autores tiene **{authors['aristas']} aristas**, transitividad **{authors['transitividad']:.3f}** y una componente principal con {authors['componente_mayor_pct']:.1f}% de los autores. La cohesión de la componente mayor se reporta mediante conectividad de nodos y aristas. La transitividad alta es esperable porque comentar un mismo video crea cliques; no implica relaciones sociales directas."""),
        markdown("""<div class="section"><h2>7 · Comunidades</h2></div>
Se aplicó Louvain ponderado a autores con al menos una coparticipación. Los aislados se conservaron en métricas y tablas, pero no aportan información a la optimización de modularidad."""),
        code("display(Image(filename=str(FIGURES / 'comunidades_autores.png'), width=900)); show_csv('resumen_comunidades.csv', 10)"),
        markdown(f"""Se obtuvieron **{summary['communities']['count']} comunidades** con modularidad **{summary['communities']['modularity']:.3f}**. Se caracterizan por autores, intensidad, videos, canales, categorías y vocabulario. El sentimiento se incorporará con una herramienta validada para español en la fase final."""),
        markdown("""<div class="section"><h2>8 · Preguntas y evidencia</h2></div>"""),
        code("show_csv('preguntas_obligatorias.csv', 10); show_csv('preguntas_adicionales.csv', 10)"),
        markdown("""<div class="section"><h2>9 · Alcance y siguiente fase</h2></div>
El avance cubre íntegramente los ejercicios 1–6 y la mayor parte del 7. La fase final añadirá sentimiento en español, centralidades justificadas, autores puente, videos articuladores, pruebas de remoción, caracterización completa de comunidades y conclusiones integradas."""),
        code("show_csv('alcance_avance_75.csv', 12); print('Cobertura:', summary['scope_points'], '/', summary['scope_total'])"),
        markdown("""### Referencias metodológicas
- Hagberg, A., Schult, D. y Swart, P. (2008). *Exploring network structure, dynamics, and function using NetworkX*.
- Blondel, V. et al. (2008). *Fast unfolding of communities in large networks*. Journal of Statistical Mechanics.
- Freeman, L. (1979). *Centrality in social networks: Conceptual clarification*. Social Networks.
<div class="footer">Laboratorio 6 · Grupo 1 · Data Science, Sección 10 · Avance reproducible</div>"""),
    ]
    for index, cell in enumerate(notebook["cells"], start=1):
        cell["id"] = f"lab6-cell-{index:02d}"

    path = NOTEBOOKS / "Lab6_Avance_75.ipynb"
    nbf.write(notebook, path)
    executed = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    for cell in executed.cells:
        if cell.cell_type == "code":
            cell.metadata.pop("execution", None)
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "text/html" in data:
                    data.pop("text/plain", None)
    nbf.write(executed, path)
    print(f"Notebook generado y ejecutado: {path}")


if __name__ == "__main__":
    main()
