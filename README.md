<div align="center">

# Laboratorio 6 · Analítica de Redes Sociales

### Participación, comunidades, puentes y sentimiento en YouTube

**CC3084 · Data Science · Sección 10 · Grupo 1**<br>
Universidad del Valle de Guatemala · Segundo semestre 2026

![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)
![Estado](https://img.shields.io/badge/estado-entrega%20final-0F9D91?style=flat-square)
![Rúbrica](https://img.shields.io/badge/cobertura-100%2F100-16A34A?style=flat-square)
![Tests](https://img.shields.io/badge/tests-40%20passed-16A34A?style=flat-square)

</div>

---

## Resultado

Este repositorio contiene la entrega final reproducible del Laboratorio 6. Integra calidad de datos, exploración, red bipartita autor–video, proyecciones, topología, comunidades, centralidades, experimentos de remoción y sentimiento en español.

| Indicador | Resultado |
|---|---:|
| Videos / canales | 293 / 97 |
| Comentarios / autores | 406 / 332 |
| Integración comentario–video | 100 % |
| Videos con comentarios | 19 de 293 |
| Autores recurrentes | 9 |
| Autores puente estructurales | 7 |
| Videos articuladores | 5 |
| Comunidades / modularidad | 10 / 0.395 |
| Sentimiento NEG / NEU / POS | 61.3 % / 18.5 % / 20.2 % |
| Confianza media del modelo | 0.818 |

> Una arista representa coparticipación observada, no amistad, respuesta, aprobación ni coordinación. `reply_count` no identifica autores de respuesta y no se usa para inventar conexiones.

## Entrega principal

- [Notebook final ejecutado](notebooks/Lab6_Analitica_Redes_Sociales.ipynb)
- [Informe final en PDF](reports/informe_final.pdf)
- [Fuente LaTeX](reports/informe_final.tex)
- [Ficha del repositorio](entrega/Ficha_Repositorio_Laboratorio_6.pdf)
- [Auditoría de la rúbrica](outputs/tables/cumplimiento_rubrica_final.csv)
- [Evidencia por ejercicio](outputs/tables/evidencia_ejercicios.csv)

Repositorio oficial: [Laboratorio 6 · Grupo 1](https://github.com/DanielBarillasM/Laboratorio-6.-Analitica-de-Redes-Sociales_Grupo-1_DS_Sec-10)

El archivo `notebooks/Lab6_Avance_75.ipynb` se conserva únicamente como historial del avance. Para la entrega y reproducción debe utilizarse el notebook final enlazado arriba.

## Reproducción completa

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_final.py
python scripts/build_final_notebook.py
python -m pytest -q
```

También puede usarse `uv sync --extra test`. El notebook ejecuta `scripts/run_final.py` en su primera celda, así que **Run All** reconstruye el análisis antes de mostrarlo.

La primera inferencia descarga `pysentimiento/robertuito-sentiment-analysis`. Después se reutiliza `sentimiento_predicciones_cache.csv` únicamente si coinciden modelo, IDs y huellas de los textos. Para repetir la inferencia:

```powershell
$env:LAB6_FORCE_SENTIMENT = "1"
python scripts/run_final.py
```

## Metodología

- Carga robusta de CSV/XLSX basada en firma de archivo.
- IDs estables para nodos; nombres y handles solo como etiquetas.
- Texto limpio para frecuencias y texto original para sentimiento.
- Videos sin comentarios conservados como nodos aislados.
- Proyecciones ponderadas por participación compartida.
- Louvain ponderado con semilla fija 42.
- Grado, grado ponderado, intermediación, cercanía armónica y PageRank.
- Puentes verificados mediante puntos de articulación y remoción.
- Sentimiento NEG/NEU/POS con probabilidades, confianza y muestra mínima `n ≥ 5`.

El modelo empleado es [RoBERTuito para análisis de sentimiento](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis), diseñado para texto social en español. Sus etiquetas siguen siendo predicciones: ironía, jerga y contexto pueden causar errores.

## Estructura

```text
├── Data/                  # CSV originales y datos normalizados
├── Instructions/          # PDF oficial
├── config/                # Curso, equipo y parámetros
├── entrega/               # Ficha DOCX/PDF del repositorio
├── notebooks/             # Notebook narrativo final
├── outputs/
│   ├── figures/           # EDA, redes, remoción y sentimiento
│   └── tables/            # Evidencia auditable y resultados
├── reports/               # Informe final TeX/PDF
├── scripts/               # Pipelines y constructores
├── src/lab6_social/       # Implementación modular
└── tests/                 # 40 pruebas automatizadas
```

## Limitaciones

Solo 19 de 293 videos tienen comentarios. La muestra depende de las consultas y del momento de recolección; vistas, likes y respuestas cambian con el tiempo. Un canal no equivale necesariamente a una persona. Los resultados caracterizan este corpus y no representan a toda la audiencia de YouTube ni a la población guatemalteca.

## Material para Canvas

Según la guía, deben presentarse:

1. `reports/informe_final.pdf`.
2. `notebooks/Lab6_Analitica_Redes_Sociales.ipynb` como notebook ejecutado y reproducible.
3. Enlace al espacio colaborativo del grupo con historial.
4. Enlace a este repositorio de GitHub.
5. Este `README.md`, incluido dentro del repositorio, con dependencias e instrucciones de ejecución.

La ficha de repositorio en `entrega/` es un apoyo opcional; no sustituye los enlaces colaborativo y de GitHub solicitados por la guía.

## Equipo

| Integrante | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

---

<div align="center"><sub>Laboratorio 6 · Grupo 1 · Data Science, Sección 10</sub></div>
