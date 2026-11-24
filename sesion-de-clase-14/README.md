# Proyecto de Minería Web - Sesión de clase 14

Esta sesión cierra el curso construyendo un **asistente para la Tienda Virtual** con los datos que se
scrapearon y vectorizaron en las sesiones anteriores. Se trabajan cuatro temas: **RAG**, **ontologías**,
**RAG determinista** y **MCP** (*Model Context Protocol*). Cada ejercicio compara el resultado **con y
sin** la tecnología sobre un mismo banco de 10 preguntas.

Los notebooks trabajan sobre los CSV de la tienda que están **copiados en la carpeta `data/`** de este
proyecto. Las herramientas de datos en vivo consultan la API REST de la Tienda Virtual local. Todos los
modelos corren **localmente, en CPU**, y no se usa ninguna API externa.

El documento [`CONTENIDOS.md`](CONTENIDOS.md) describe todos los conceptos, el código y los resultados
de la sesión.

## Paso previo: levantar "Mi Tienda Virtual"

Los ejercicios 3, 4 y el propuesto consultan la API en `http://localhost:3000`.

```bash
cd ../tienda-virtual
cp .env.example .env.local        # solo la primera vez
npm install                       # solo la primera vez
npm run build && npm run start    # modo producción
```

Se recomienda el modo producción. `npm run dev` también funciona, pero el modo desarrollo vigila los
archivos del proyecto, y si detecta la carpeta superior como raíz del workspace también recorre los
`.venv` de las sesiones. Eso puede dejar la API sin responder.

Comprobar con `curl http://localhost:3000/api/health`. La base de la tienda está en memoria: al
reiniciarla vuelve a los datos originales.

## Instalación y kernel

Requiere **Python 3.11**. Todo se instala dentro de esta carpeta: el entorno virtual en `.venv/` y el
kernel de Jupyter dentro del mismo entorno (`--sys-prefix`), sin tocar la configuración global.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name s14-venv --display-name "Python (sesion-de-clase-14)"
jupyter notebook                  # usar el Jupyter del .venv para ver el kernel
```

Al abrir los notebooks, seleccionar el kernel **Python (sesion-de-clase-14)**.

La primera ejecución descarga los modelos de Hugging Face a su caché local (unos 1.5 GB en total).
Después funcionan sin internet.

| Modelo | Uso | Notebooks |
| --- | --- | --- |
| `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | embeddings para ChromaDB y para el router de intenciones | 01–05 |
| `Qwen/Qwen2.5-0.5B-Instruct` (transformers) | LLM para generar respuestas y llamar tools | 01–05 |

Para usar un modelo más capaz, cambiar `MODELO_LLM` a `Qwen/Qwen2.5-1.5B-Instruct`.

## Datos (`data/`)

| Archivo | Contenido | Origen |
| --- | --- | --- |
| `productos.csv` | 90 productos (9 categorías × 10) con descripción, precio, stock y calificación | scraping de las páginas de producto |
| `clientes.csv` | 120 clientes de 7 países. Contiene **datos personales**; en la ontología solo se usan `id`, ciudad y país | scraping del sitemap de clientes |
| `ordenes.csv` | 300 órdenes con cliente, país y totales | API REST `/api/ordenes` |
| `orden_items.csv` | 900 líneas de orden (orden, producto, cantidad) | API REST `/api/ordenes` (campo `items`) |
| `resenas_entrega.csv` | 110 reseñas post-compra (1–5★) ligadas a una orden y un producto | scraping con Playwright |
| `comentarios.csv` | 135 testimonios generales de clientes | scraping de `/testimonios` |
| `preguntas_prueba.csv` | banco de 10 preguntas con términos esperados y enfoque ideal | creado para la sesión |

## Notebooks (`notebooks/`)

Se ejecutan **en orden**: el 01 crea la base vectorial `bdv/` y el 02 crea la ontología `ontologia/`,
que usan los siguientes.

| Notebook | Técnica | Qué hace |
| --- | --- | --- |
| `01_ejercicio1_rag.ipynb` | ChromaDB, BM25, Qwen | **Ejercicio 1**: LLM sin contexto, documentos y metadatos, indexación persistente, búsqueda semántica con filtros, generación aumentada con citas, línea base por palabras clave. **Comparación**: LLM solo vs. RAG por palabras clave vs. RAG vectorial. |
| `02_ejercicio2_ontologia_rag_determinista.ipynb` | rdflib, SPARQL, pyshacl | **Ejercicio 2**: TBox y ABox de la tienda en Turtle, consultas SPARQL, inferencia por jerarquía y regla de negocio (`ProductoEnRiesgo`), validación SHACL. RAG determinista: catálogo de intenciones, router por embeddings, extracción de entidades desde la ontología, plantillas SPARQL, *fallback*. **Comparaciones**: pandas vs. RAG vs. SPARQL; RAG generativo (muestreo y greedy) vs. determinista, con repetibilidad. |
| `03_ejercicio3_mcp_server.ipynb` | SDK `mcp` (MCPServer, Client) | **Ejercicio 3**: conceptos de MCP, escritura de `mcp_tienda/herramientas.py` y `mcp_tienda/server.py`, cliente MCP (`tools/list`, `tools/call`, resources, prompts). **Comparación**: stock en vivo con LLM solo, RAG, ontología y LLM + MCP, cambiando el stock en la tienda. |
| `04_integracion_asistente.ipynb` | `mcp_tienda/agente.py` | **Integración**: agente Qwen + cliente MCP; modos de falla del tool-calling con un modelo pequeño y siete estrategias para hacerlo fiable. **Comparación**: tool-calling ingenuo vs. robusto y **tablero final** de las 5 configuraciones. Relación con CRISP-ML(Q). |
| `05_ejercicio_propuesto.ipynb` | todo lo anterior | **Ejercicio propuesto**: tres opciones (post-venta, ontología extendida, recomendador), esqueleto con `TODO`, entregable y rúbrica. |

## Asistente (`mcp_tienda/`)

| Archivo | Contenido |
| --- | --- |
| `herramientas.py` | lógica sin MCP: RAG semántico, RAG determinista, API en vivo, resources (lo escribe el notebook 03) |
| `server.py` | MCP server `tienda-virtual` por stdio: 5 tools, 2 resources, 1 prompt (lo escribe el notebook 03) |
| `agente.py` | cliente MCP + Qwen con estrategia `ingenua` o `robusta` |
| `server_extendido.py` | punto de partida del ejercicio propuesto (lo escribe el notebook 05) |

Ejecutar el server a mano (queda esperando mensajes JSON-RPC por stdin):

```bash
.venv/bin/python mcp_tienda/server.py
```

Cualquier host compatible con MCP puede registrarlo con
`{"command": ".venv/bin/python", "args": ["mcp_tienda/server.py"]}`.

## Otros archivos

| Ruta | Contenido |
| --- | --- |
| `comparacion.py` | utilidades para evaluar el banco de preguntas y mostrar las tablas "con vs. sin" |
| `ontologia/tienda.ttl` | TBox: clases y propiedades |
| `ontologia/tienda_poblada.ttl` | TBox + ABox + tripletas inferidas |
| `ontologia/intenciones.json` | catálogo de intenciones del RAG determinista |
| `bdv/` | base vectorial ChromaDB (colecciones `productos` y `resenas`) |
| `salidas/` | resultados por configuración (`resultados_*.csv`), tablero final, gráficos y log del MCP server |

## Agenda de la clase (3 h)

| Tiempo | Bloque |
| --- | --- |
| 0:00–0:10 | Arquitectura del día |
| 0:10–0:55 | Ejercicio 1 · RAG |
| 0:55–1:40 | Ejercicio 2 · Ontología + RAG determinista |
| 1:40–1:55 | Break |
| 1:55–2:35 | Ejercicio 3 · MCP server |
| 2:35–2:50 | Integración: agente + MCP |
| 2:50–3:00 | Ejercicio propuesto |
