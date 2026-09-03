<div align="center">

# Laboratorio 6 · Analítica de Redes Sociales

### Participación, copresencia y comunidades en una muestra de YouTube

**CC3084 · Data Science · Sección 10 · Grupo 1**<br>
Universidad del Valle de Guatemala · Segundo semestre 2026

![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)
![Avance](https://img.shields.io/badge/avance-74%2F100-0F9D91?style=flat-square)
![Tests](https://img.shields.io/badge/tests-12%20passed-16A34A?style=flat-square)
![Reproducible](https://img.shields.io/badge/análisis-reproducible-F59E0B?style=flat-square)

</div>

---

## Propósito

Este repositorio contiene el avance reproducible del Laboratorio 6. Estudia cómo se concentra la participación en una muestra de YouTube y cómo se conectan autores, videos, canales y temas mediante redes bipartitas, proyecciones y comunidades.

El avance cubre completamente los ejercicios 1–6 y la mayor parte del ejercicio 7: **74 de 100 puntos trazables de la rúbrica**. El PDF define como avance mínimo las actividades 1–4; este repositorio incorpora además proyecciones, topología y comunidades.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Videos | 293 |
| Canales | 97 |
| Comentarios | 406 |
| Autores | 332 |
| Comentarios integrados | 100 % |
| Nodos de la red bipartita | 625 |
| Aristas autor–video | 343 |
| Comunidades de autores | 10 |
| Modularidad | 0.395 |
| Participación reunida por los cinco videos principales | 75.4 % |

> Una arista autor–video significa que el autor comentó ese video. No representa amistad, respuesta directa, aprobación ni coordinación.

## Entregables del avance

- [Notebook ejecutado](notebooks/Lab6_Avance_75.ipynb)
- [Informe en PDF](reports/informe_avance_75.pdf)
- [Fuente LaTeX](reports/informe_avance_75.tex)
- [Script principal](scripts/run_advance.py)
- [Tablas de resultados](outputs/tables)
- [Visualizaciones](outputs/figures)
- [Código reutilizable](src/lab6_social)
- [Pruebas automatizadas](tests)

## Consideración especial de los datos

`youtube_videos_12808423.csv` tiene extensión `.csv`, pero actualmente su contenido es un libro Excel. El cargador inspecciona la firma binaria y acepta:

- CSV real en UTF-8, UTF-8 con BOM, Windows-1252 o Latin-1;
- archivos `.xlsx` normales;
- libros Excel descargados con extensión `.csv`.

Cuando se reemplace el archivo por el CSV verdadero, no será necesario modificar el código. Además, tres IDs que Excel interpretó como fórmulas se recuperan desde `video_url` y quedan registrados en `outputs/tables/video_ids_recuperados.csv`.

## Ejecución

### 1. Crear el entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

También puede utilizarse `uv`:

```powershell
uv sync --extra test
```

### 2. Generar el análisis

```powershell
python scripts/run_advance.py
```

El script localiza automáticamente los archivos cuyo nombre contiene `videos` y `comments` dentro de `/Data`.

### 3. Ejecutar directamente desde el notebook

Abra `notebooks/Lab6_Avance_75.ipynb` y seleccione **Run All / Ejecutar todo**. La primera celda ejecuta el pipeline completo y regenera automáticamente las tablas y figuras antes de mostrar los resultados.

El notebook debe mantenerse dentro del repositorio porque utiliza `/Data`, `/src` y `/scripts` para evitar duplicar la implementación.

### 4. Reconstruir el archivo del notebook

```powershell
python scripts/build_notebook.py
```

### 5. Compilar el informe

```powershell
cd reports
pdflatex -interaction=nonstopmode -halt-on-error informe_avance_75.tex
pdflatex -interaction=nonstopmode -halt-on-error informe_avance_75.tex
```

### 6. Ejecutar pruebas

```powershell
python -m pytest -q
```

## Estructura

```text
├── Data/                   # Datos originales proporcionados
├── Instructions/          # PDF oficial
├── config/                 # Metadatos del curso y equipo
├── data/processed/         # Resultados intermedios reconstruibles
├── notebooks/              # Notebook narrativo ejecutado
├── outputs/
│   ├── figures/            # EDA y visualizaciones de redes
│   └── tables/             # Calidad, métricas, nodos y aristas
├── reports/                # Informe TeX y PDF
├── scripts/                # Flujo y generador del notebook
├── src/lab6_social/        # Carga, limpieza, análisis y redes
└── tests/                  # Validación automatizada
```

## Decisiones metodológicas

- Los nodos se identifican con IDs estables; nombres y handles son etiquetas.
- Se conservan `texto_original` y `texto_limpio`.
- Los videos sin comentarios permanecen como nodos aislados.
- `reply_count` no genera conexiones entre usuarios porque no identifica quién respondió.
- La proyección autor–autor representa copresencia en videos, no relaciones sociales explícitas.
- Louvain usa como peso la cantidad de videos compartidos y una semilla fija de 42.
- El sentimiento formal se reserva para una herramienta validada para español en la fase final.

## Limitaciones esenciales

Solo 19 de 293 videos tienen comentarios. La muestra depende de consultas y canales utilizados durante la recolección; visualizaciones, likes y respuestas son conteos observados en un momento específico. Por ello, los resultados no deben generalizarse a todos los usuarios de YouTube ni a la población de Guatemala.

## Equipo

| Integrante | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

---

<div align="center"><sub>Laboratorio 6 · Grupo 1 · Data Science, Sección 10</sub></div>
