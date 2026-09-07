# Handoff · Redes finales (Persona 1 → Persona 2)

Laboratorio 6 · Analítica de Redes Sociales · CC3084 Data Science · Sección 10 · Grupo 1
Universidad del Valle de Guatemala · Segundo semestre 2026

Este documento describe el trabajo estructural agregado sobre `main` después del avance del 75 %:
**ejercicio 8 completo**, la parte estructural del **ejercicio 7** y la pregunta pendiente del
**inciso 3.5** sobre autores puente. No se tocaron el notebook, el informe de avance, el README,
la ficha DOCX ni nada relacionado con sentimiento.

---

## 1. Cómo reproducir

```bash
python3 -m pip install -r requirements.txt      # o: uv sync
python3 scripts/run_network_final.py            # ~8 s; escribe tablas y figuras nuevas
python3 -m pytest tests -q                      # 31 pruebas
```

El script no depende del avance: reconstruye la red bipartita y ambas proyecciones desde
`Data/` con los mismos módulos (`lab6_social.io`, `lab6_social.networks`) y la semilla 42.
Ejecutarlo dos veces produce archivos byte a byte idénticos, incluidas las figuras.

> **Advertencia para la Persona 2.** `scripts/run_advance.py` sigue funcionando, pero fue
> ejecutado originalmente en Windows: al correrlo en macOS o Linux reescribe los CSV del avance
> con finales de línea LF y genera un diff enorme aunque los números no cambien. Si solo se
> necesita el trabajo nuevo, basta con `run_network_final.py`. Si se vuelve a correr
> `run_advance.py`, conviene revisar con `git diff --stat` antes de hacer commit.

---

## 2. Metodología

### 2.1 Pesos y distancias

Los pesos de las proyecciones miden **fuerza de conexión**, no costo:

| Red | Arista | Peso |
|---|---|---|
| Autor–autor | dos autores comentaron el mismo video | número de videos compartidos |
| Video–video | dos videos comparten al menos un autor | número de autores compartidos |

Por eso `lab6_social.centrality.add_distance()` agrega a cada arista

```
distance = 1 / weight
```

y el resto del módulo respeta esta separación:

| Se calcula con `distance` | Se calcula con `weight` |
|---|---|
| caminos mínimos | grado ponderado |
| intermediación (betweenness) | PageRank |
| centralidad armónica | intensidad de conexión |

### 2.2 Elección de centralidades

`outputs/tables/justificacion_centralidades.csv` documenta cada decisión. En resumen:

- **Grado y grado ponderado**: alcance directo e intensidad de copresencia.
- **Intermediación**: es la medida central del ejercicio 8, porque identifica quién está en el
  paso entre grupos.
- **Centralidad armónica** en lugar de cercanía clásica: la red está fragmentada (10 componentes
  de autores, 284 de videos). La armónica suma `1/d` y asigna 0 a los pares inalcanzables, así
  que sigue siendo comparable entre componentes; la cercanía clásica no está definida.
- **PageRank** ponderado como complemento del grado: descuenta la centralidad heredada de vecinos
  muy conectados.
- **Vectores propios: descartada**. En un grafo con muchos componentes se concentra en la
  componente dominante y asigna cero al resto.

La centralidad armónica se acumula recorriendo los destinos en orden fijo
(`_harmonic_centrality`) en lugar de usar la utilidad de NetworkX, porque la suma en coma
flotante depende del orden y hacía que los CSV cambiaran en el decimal 15 entre corridas.

### 2.3 Definiciones operativas

- **Autor recurrente**: comentó en más de un video (`videos_comentados > 1`).
- **Diversidad de participación**: canales y categorías distintas, proporción
  `canales_distintos / videos_comentados` y entropía normalizada del reparto de sus comentarios
  entre videos.
- **Puente estructural**: es punto de articulación; al removerlo aparecen componentes nuevos.
- **Puente redundante**: tiene intermediación positiva pero rutas alternativas; conecta sin ser
  indispensable.
- **Video articulador**: el equivalente en la proyección video–video.
- **Fragmentación**: proporción de **pares desconectados**, medida siempre sobre el mismo conjunto
  de nodos (todos menos el removido), de modo que el incremento no se deba al simple hecho de
  tener un nodo menos. `incremento_fragmentacion = fragmentacion_despues − fragmentacion_antes`.

---

## 3. Resultados principales

### 3.1 Autores (proyección autor–autor: 332 nodos, 10 732 aristas)

- **9 de 332 autores (2.7 %)** comentaron en más de un video; el máximo observado es 3 videos.
- Solo esos **9 autores tienen intermediación positiva**: el resto vive dentro de una sola
  camarilla de video y no aparece en ningún camino mínimo.
- **7 son puntos de articulación** (puentes estructurales) y **2 son puentes redundantes**.
- 4 autores quedan aislados por ser el único comentarista de su video.

Los tres primeros por intermediación:

| Autor | Videos | Grado | Intermediación | ¿Articulación? |
|---|---:|---:|---:|:--:|
| @virgiliogarcia3039 | 2 | 145 | 0.2390 | sí |
| @inge_vergueta | 3 | 183 | 0.2203 | sí |
| @josegil3813 | 2 | 49 | 0.1827 | sí |

### 3.2 Videos (proyección video–video: 293 nodos, 11 aristas)

- Solo **10 videos** comparten autores con otro video y forman una única componente.
- **5 son articuladores**; "Qué rico come tu diputado" es el único cuya remoción parte la
  componente en tres.
- 283 videos quedan aislados: 274 sin comentarios recolectados y 9 con comentarios pero sin
  ningún autor compartido.

### 3.3 Recurrencia frente a intermediación

Coinciden como **condición necesaria**, no como jerarquía. Comentar en más de un video es la única
forma de obtener intermediación positiva, de modo que el Spearman sobre los 332 autores es
`rho = 1.000` por construcción; **entre los 9 autores recurrentes cae a `rho = 0.208`**. Ningún
autor con el máximo de 3 videos encabeza la intermediación: la lidera @virgiliogarcia3039 con
2 videos. Lo decisivo es **qué** videos se conectan, no cuántos.

### 3.4 Experimentos de remoción

Se evaluaron 25 nodos (todos los puntos de articulación más los líderes en intermediación y grado
ponderado). **12 fragmentan la red.** Caso extremo: al remover a @virgiliogarcia3039 la componente
mayor de autores cae de 276 a 214 nodos (−22.5 %) y la fragmentación sube 23.9 puntos
porcentuales. Fuera de los puntos de articulación, quitar nodos de grado ponderado alto **no
cambia** el número de componentes: la red es un conjunto de camarillas por video, muy redundante
hacia adentro y frágil solo en los pocos enlaces entre camarillas.

Una consecuencia útil para el informe: en esta red el incremento de fragmentación de un punto de
articulación **coincide numéricamente** con su intermediación normalizada, porque los pares que
quedan desconectados son exactamente los pares cuyos caminos mínimos pasaban por él.

### 3.5 Comunidades (parte estructural del ejercicio 7)

10 comunidades, modularidad 0.3949 (Louvain ponderado, semilla 42). **6 dependen de pocos nodos.**
Las tres principales:

| Comunidad | Autores | Densidad interna | Articulaciones internas | Caída de la componente mayor al quitar su nodo top |
|---:|---:|---:|---:|---:|
| 1 | 126 | 1.000 | 0 | 0.8 % |
| 2 | 61 | 0.397 | 2 | 29.5 % |
| 3 | 55 | 0.806 | 1 | 1.8 % |

La comunidad 1 es una camarilla perfecta (un solo video muy comentado); la comunidad 2 es la que
realmente depende de sus puentes.

### 3.6 Respuesta al inciso 3.5

`outputs/tables/respuesta_35_autores_puente.csv` contiene la respuesta con su matiz, lista para
integrarse a `preguntas_obligatorias.csv` o al informe final:

> 7 autores son puntos de articulación en la proyección autor–autor: al removerlos la red gana
> componentes. 5 videos cumplen el papel equivalente en la proyección video–video. De los 9 autores
> recurrentes, solo 7 son puentes estructurales; el resto comenta en videos que ya estaban
> conectados por otras personas.

---

## 4. Archivos creados

### Código

| Archivo | Contenido |
|---|---|
| `src/lab6_social/centrality.py` | distancias inversas, centralidades, perfiles de autor y video, puntos de articulación, puentes, remoción y estructura de comunidades |
| `src/lab6_social/visualization_network_final.py` | las cinco figuras nuevas |
| `scripts/run_network_final.py` | orquestador reproducible |
| `tests/test_centrality.py` | 19 pruebas nuevas (31 en total) |

### Tablas (`outputs/tables/`)

`centralidad_autores.csv` · `centralidad_videos.csv` · `autores_recurrentes.csv` ·
`autores_puente.csv` · `videos_articuladores.csv` · `puntos_articulacion.csv` ·
`impacto_remocion.csv` · `comunidades_principales_estructura.csv`

Complementarias: `justificacion_centralidades.csv` (ejercicio 8.1),
`interpretacion_estructural.csv` (las nueve preguntas de interpretación con evidencia),
`respuesta_35_autores_puente.csv` y `resumen_redes_finales.json`.

### Figuras (`outputs/figures/`)

`centralidad_autores.png` · `centralidad_videos.png` · `autores_puente.png` ·
`videos_articuladores.png` · `impacto_remocion.png`

---

## 5. Reglas metodológicas respetadas

- `reply_count` **no** genera aristas: no identifica a quién respondió a quién.
- Los nodos se identifican por `author_channel_id` y `video_id`; los nombres y handles son solo
  etiquetas.
- Los nodos aislados se conservan en tablas y métricas; solo se ocultan del dibujo del grafo, y la
  figura lo indica.
- Una arista significa copresencia de comentarios: no es amistad, conversación, respuesta ni
  aprobación.
- No se generaliza a todos los usuarios de YouTube ni a la población de Guatemala.
- Sin rutas absolutas, sin credenciales, semilla fija en 42.

---

## 6. Qué queda para la Persona 2

1. **Ejercicio 9 · sentimiento**: herramienta validada para español, comparación por video, canal,
   tema o comunidad, y explicación de hallazgos.
2. **Ejercicio 7.5**: agregar el sentimiento a la caracterización de comunidades; la parte
   estructural ya está en `comunidades_principales_estructura.csv`.
3. **Ejercicio 10**: interpretación, limitaciones y conclusiones integradas. Se puede partir de
   `interpretacion_estructural.csv`, sobre todo de las preguntas 8 y 9 (cobertura parcial y
   diferencia entre centralidad observada e influencia real).
4. **Integración final**: notebook ejecutable, informe final en TeX/PDF, README y ficha DOCX, e
   incorporar `respuesta_35_autores_puente.csv` a las preguntas obligatorias del inciso 3.5.
5. **Actualizar el alcance**: `alcance_avance_75.csv` todavía marca "Nodos centrales y
   participantes puente" con 0 de 7 puntos y `evidencia_ejercicios.csv` marca el ejercicio 8 como
   pendiente. Ambos se generan en `scripts/run_advance.py`; actualizarlos es parte de la
   integración final y se dejó deliberadamente para la Persona 2.

---

## 7. Limitación que atraviesa todo lo anterior

Solo **19 de 293 videos** tienen comentarios recolectados y el 75.4 % de los comentarios se
concentra en cinco videos. La proyección video–video parte de una fragmentación del 99.92 % de
pares desconectados. El bajo número de puentes mide la cobertura de la muestra tanto como el
comportamiento de las audiencias: es un resultado sobre **la red observada**, no sobre YouTube.
