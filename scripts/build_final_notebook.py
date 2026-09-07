"""Construye y ejecuta el notebook narrativo final del Laboratorio 6."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Lab6_Analitica_Redes_Sociales.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


STYLE = r"""
<style>
:root { --navy:#102A43; --blue:#2563EB; --teal:#0F9D91; --slate:#475569; --paper:#F8FAFC; }
.hero {padding:34px 38px;border-radius:20px;background:linear-gradient(125deg,#102A43,#2563EB 62%,#0F9D91);color:white;box-shadow:0 14px 30px #0f172a26;margin:8px 0 24px}
.hero h1{font-size:2.35rem;margin:0 0 8px}.hero p{font-size:1.05rem;opacity:.94;margin:5px 0}
.tag{display:inline-block;padding:5px 10px;margin:10px 5px 0 0;background:#ffffff20;border:1px solid #ffffff55;border-radius:999px}
.section{padding:18px 24px;border-left:6px solid var(--teal);background:linear-gradient(90deg,#ECFEFF,#F8FAFC);border-radius:8px;margin:26px 0 14px}
.section h2{margin:0;color:var(--navy)}
.note{padding:14px 18px;border-radius:10px;background:#EFF6FF;border:1px solid #BFDBFE;color:#1E3A8A}
.warn{padding:14px 18px;border-radius:10px;background:#FFF7ED;border:1px solid #FED7AA;color:#9A3412}
table.dataframe{font-size:13px;border-collapse:collapse} table.dataframe th{background:#102A43;color:white;padding:7px} table.dataframe td{padding:6px;border-bottom:1px solid #E2E8F0}
</style>
<div class="hero">
  <h1>Laboratorio 6 · Analítica de Redes Sociales</h1>
  <p>Participación, comunidades, puentes y sentimiento en YouTube</p>
  <p><b>CC3084 · Data Science · Sección 10 · Grupo 1</b></p>
  <span class="tag">293 videos</span><span class="tag">406 comentarios</span><span class="tag">332 autores</span><span class="tag">Entrega final</span>
</div>
"""


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook.cells = [
        md(STYLE + """
**Integrantes:** Jorge Gabriel Palacios Sales (231385) · Pablo Daniel Barillas Moreno (22193) · Roberto Emiliano Otoniel (23968)  
**Universidad del Valle de Guatemala · Segundo semestre 2026**

Este notebook es el punto de entrada reproducible de la entrega. La primera celda de código ejecuta todo el pipeline; las siguientes celdas leen y explican los productos generados.
"""),
        code(r"""
from pathlib import Path
import json, subprocess, sys
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name.lower() == "notebooks":
    ROOT = ROOT.parent
assert (ROOT / "scripts" / "run_final.py").exists(), "Ejecute el notebook dentro del repositorio."

completed = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "run_final.py")],
    cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print(completed.stdout)
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
summary = json.loads((TABLES / "resumen_final.json").read_text(encoding="utf-8"))
"""),
        code(r"""
display(HTML('''
<style>
.cards{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px;margin:12px 0 20px}
.card{padding:16px;border-radius:12px;background:white;border:1px solid #CBD5E1;box-shadow:0 5px 15px #0f172a12}
.card b{font-size:1.6rem;color:#2563EB;display:block}.card span{color:#475569}
</style>
<div class='cards'>
 <div class='card'><b>293</b><span>videos</span></div><div class='card'><b>406</b><span>comentarios</span></div>
 <div class='card'><b>332</b><span>autores</span></div><div class='card'><b>100 %</b><span>integración</span></div>
</div>'''))
"""),
        md("""<div class="section"><h2>1–2 · Integración, calidad y preprocesamiento</h2></div>

Los archivos se detectan por contenido y no solo por extensión, por lo que el cargador acepta CSV y Excel. En videos, cada fila es un video y `video_id` es la llave primaria; en comentarios, cada fila es un comentario principal y `comment_id` es la llave primaria. `video_id` actúa como llave foránea. `channel_id` identifica al canal propietario y `author_channel_id` al autor; `category` describe el video y `source_query` documenta el muestreo.

Se auditan duplicados, ausencias, tipos, constantes, atípicos por IQR y consistencia de IDs/nombres/handles. Los conteos admiten comas, espacios y sufijos `K`, `mil`, `M`, `millón/millones`; los vacíos descriptivos pasan a cero y los valores imposibles a faltante. Para frecuencias se usa `texto_limpio`; para sentimiento se conserva `texto_original` para no perder negaciones, emojis ni puntuación."""),
        code(r"""
display(pd.read_csv(TABLES / "formatos_detectados.csv"))
display(pd.read_csv(TABLES / "calidad_resumen.csv"))
display(pd.read_csv(TABLES / "integracion_resumen.csv"))
display(pd.read_csv(TABLES / "tratamiento_variables.csv").head(8))
"""),
        md("""<div class="note"><b>Decisión de red:</b> <code>reply_count</code> solo expresa cuántas respuestas recibió un comentario. Como no identifica a quienes respondieron, no se fabrican aristas autor–autor.</div>"""),
        md("""<div class="section"><h2>3 · Exploración de participación y contenido</h2></div>

Se cuantifica participación por video, canal y autor; concentración; asociación entre visualizaciones y comentarios; y frecuencias de unigramas, bigramas y hashtags."""),
        code(r"""
display(Image(filename=str(FIGURES / "eda_panorama.png"), width=900))
display(Image(filename=str(FIGURES / "frecuencias_contenido.png"), width=900))
display(pd.read_csv(TABLES / "concentracion_participacion.csv"))
"""),
        md("""El top 5 de videos reúne **75.4 %** de los comentarios. En los 293 videos, vistas y comentarios presentan asociación débil (Spearman ρ=0.080, p=0.170); al restringir a los 19 videos con comentarios, ρ=0.811 (p<0.001). El contraste evidencia un sesgo de cobertura y no debe interpretarse como causalidad ni popularidad universal."""),
        md("""<div class="section"><h2>4–5 · Red bipartita y proyecciones</h2></div>

La red bipartita conecta autores con videos comentados. La proyección autor–autor representa copresencia en uno o más videos; la proyección video–video conecta videos con al menos un autor compartido. El peso conserva la cantidad de participaciones compartidas."""),
        code(r"""
display(pd.read_csv(TABLES / "metricas_redes.csv"))
display(Image(filename=str(FIGURES / "red_bipartita_completa.png"), width=920))
display(Image(filename=str(FIGURES / "proyeccion_videos.png"), width=860))
"""),
        md("""<div class="section"><h2>6 · Topología, componentes y fragmentación</h2></div>

Se reportan tamaño, aristas, densidad, componentes, componente mayor, grados y periferia. La remoción se usa como experimento estructural: compara componentes y fragmentación antes y después de retirar candidatos centrales."""),
        code(r"""
display(pd.read_csv(TABLES / "componentes_redes.csv").groupby("red").head(5))
display(pd.read_csv(TABLES / "impacto_remocion.csv").head(12))
display(Image(filename=str(FIGURES / "impacto_remocion.png"), width=900))
"""),
        md("""La red video–video posee solo **10 videos conectados** y **283 aislados**. La proyección de autores tiene transitividad 0.984, agrupamiento ponderado 0.486 y conectividad de nodos/aristas 1/5 en su componente mayor; la de videos tiene transitividad 0.316, agrupamiento 0.006 y conectividad 1/1. De 25 nodos candidatos evaluados, 12 fragmentan sus respectivas proyecciones. La estructura es redundante dentro de camarillas, pero frágil en sus pocos enlaces entre videos."""),
        md("""<div class="section"><h2>7 · Comunidades</h2></div>

Se eligió la proyección autor–autor porque agrupa patrones de coparticipación y permite asociar directamente el contenido de los comentarios. Louvain maximiza modularidad usando como peso el número de videos compartidos; con semilla 42 identifica 10 comunidades (modularidad **0.395**). Cada grupo se caracteriza por autores, intensidad, términos, videos, canales, categorías, estructura interna y sentimiento. Cuatro autores aislados se conservan como grupo 0 y no se fuerzan dentro de una comunidad."""),
        code(r"""
communities = pd.read_csv(TABLES / "comunidades_caracterizacion_final.csv")
display(communities[["community","authors","comments","videos","top_terms","sentimiento_dominante","pct_negativo","pct_neutral","pct_positivo","dependiente_de_pocos_nodos"]])
display(Image(filename=str(FIGURES / "comunidades_autores.png"), width=850))
"""),
        md("""**Tres comunidades principales.** C1 reúne 126 autores y 161 comentarios en tres videos/canales; predominan “pueblo”, “dinero” y “diputados”, con 80.7 % negativo. C2 contiene 61 autores y 83 comentarios alrededor de Gobierno/Noti7 y temas de presidencia/Guatemala; su distribución es 44.6 % negativa, 27.7 % neutral y 27.7 % positiva. C3 agrupa 55 autores y 60 comentarios, principalmente sobre USAC/corrupción en Quorum y TN23; alcanza 61.7 % negativo. C2 y C3 dependen de pocos nodos articuladores; C1 es internamente una camarilla densa."""),
        md("""<div class="section"><h2>8 · Centralidad, puentes y articuladores</h2></div>

Grado y grado ponderado reflejan alcance directo; intermediación identifica conexiones entre regiones; cercanía armónica funciona en redes desconectadas; PageRank ponderado combina cantidad y calidad de vecinos. Un puente estructural debe además ser punto de articulación y demostrar impacto al removerse."""),
        code(r"""
authors = pd.read_csv(TABLES / "autores_puente.csv")
videos = pd.read_csv(TABLES / "videos_articuladores.csv")
display(authors[["label","videos_comentados","intermediacion","es_punto_articulacion","tipo_puente"]])
display(videos[["label","videos_conectados","intermediacion","es_punto_articulacion","tipo_articulador"]])
display(Image(filename=str(FIGURES / "autores_puente.png"), width=880))
display(Image(filename=str(FIGURES / "videos_articuladores.png"), width=880))
"""),
        md("""Se hallaron **7 autores puente estructurales** y **5 videos articuladores**. La recurrencia es necesaria para conectar videos, pero no basta: importa cuáles camarillas une cada autor y si existen rutas alternativas."""),
        md("""<div class="section"><h2>9 · Contenido y sentimiento en español</h2></div>

Se emplea `pysentimiento/robertuito-sentiment-analysis`, modelo para texto social en español con etiquetas NEG/NEU/POS. Se guardan las tres probabilidades, la clase, confianza, margen, revisión del modelo y huella SHA-256 del texto. Una caché solo se reutiliza si coinciden modelo, IDs y textos."""),
        code(r"""
general = pd.read_csv(TABLES / "sentimiento_general.csv")
confidence = pd.read_csv(TABLES / "sentimiento_confianza.csv")
display(general)
display(confidence)
display(Image(filename=str(FIGURES / "sentimiento_panorama.png"), width=900))
display(Image(filename=str(FIGURES / "sentimiento_por_comunidad.png"), width=900))
"""),
        md("""El corpus es **61.3 % negativo, 18.5 % neutral y 20.2 % positivo**. La confianza media es 0.818; 61 comentarios (15.0 %) quedan bajo 0.60. Las comparaciones por grupo solo se interpretan formalmente con al menos cinco comentarios. La comunidad 5 contrasta con el patrón general: 84.0 % positivo (n=25)."""),
        md("""<div class="warn"><b>Cautela:</b> una etiqueta automática no equivale a emoción verificada. Ironía, jerga, contexto político y mezcla lingüística pueden causar errores. Por eso se reportan probabilidades y confianza, y no solo la clase ganadora.</div>"""),
        md("""<div class="section"><h2>10 · Síntesis, limitaciones y conclusiones</h2></div>"""),
        code(r"""
display(pd.read_csv(TABLES / "preguntas_obligatorias.csv"))
display(pd.read_csv(TABLES / "conclusiones_finales.csv"))
display(pd.read_csv(TABLES / "limitaciones_finales.csv"))
"""),
        md("""La evidencia describe el corpus entregado, no a toda la audiencia guatemalteca de YouTube. La cobertura parcial, el muestreo por consultas y el carácter temporal de vistas/likes limitan la generalización. Aun así, la combinación de topología, remoción, comunidades, contenido y sentimiento permite distinguir participación masiva de conexión estructural."""),
        md("""<div class="section"><h2>Auditoría de la rúbrica</h2></div>"""),
        code(r"""
rubric = pd.read_csv(TABLES / "cumplimiento_rubrica_final.csv")
display(rubric)
print(f"Cobertura documentada: {rubric.puntos_cubiertos.sum()}/{rubric.puntos.sum()} puntos")
assert rubric.puntos_cubiertos.sum() == rubric.puntos.sum() == 100
"""),
        md("""<div class="note"><b>Reproducibilidad:</b> ejecute <code>python scripts/run_final.py</code> para reconstruir tablas y figuras. Use <code>LAB6_FORCE_SENTIMENT=1</code> únicamente si desea repetir la inferencia en vez de reutilizar la caché validada.</div>"""),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    client = NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(notebook, OUTPUT)
    print(f"Notebook final creado y ejecutado: {OUTPUT}")


if __name__ == "__main__":
    main()
