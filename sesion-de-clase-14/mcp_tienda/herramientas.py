"""Lógica de las herramientas de la Tienda Virtual (sin nada de MCP).

Empaqueta lo construido en los ejercicios 1 y 2 para que el MCP server lo exponga:
- RAG semántico sobre la base vectorial `bdv/` (Ejercicio 1)
- RAG determinista sobre la ontología `ontologia/` (Ejercicio 2)
- Datos en vivo desde la API REST de la Tienda Virtual

Todo se carga de forma perezosa (la primera vez que se usa) para que el server arranque rápido.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]
BDV = RAIZ / "bdv"
ONTO = RAIZ / "ontologia"
MODELO_EMB = "paraphrase-multilingual-MiniLM-L12-v2"
TIENDA_URL = os.environ.get("TIENDA_URL", "http://localhost:3000")
API_KEY = os.environ.get("TIENDA_API_KEY", "sk_demo_000000000000000000000000000000")
UMBRAL = 0.55
PREFIJOS = ("PREFIX tv: <http://tienda.local/ontologia#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n")


# ---------------------------------------------------------------- recursos compartidos
@lru_cache
def _modelo_emb():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELO_EMB)


@lru_cache
def _colecciones():
    import chromadb
    from chromadb.utils import embedding_functions
    fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODELO_EMB)
    cliente = chromadb.PersistentClient(path=str(BDV))
    return (cliente.get_collection("productos", embedding_function=fn),
            cliente.get_collection("resenas", embedding_function=fn))


@lru_cache
def _grafo():
    from rdflib import Graph
    return Graph().parse(ONTO / "tienda_poblada.ttl")


def _sparql(consulta: str, **parametros) -> list[dict]:
    from rdflib import Literal
    enlaces = {k: Literal(v) for k, v in parametros.items()}
    r = _grafo().query(PREFIJOS + consulta, initBindings=enlaces)
    return [{str(var): (fila[i].toPython() if fila[i] is not None else None)
             for i, var in enumerate(r.vars)} for fila in r]


@lru_cache
def _intenciones():
    intenciones = json.loads((ONTO / "intenciones.json").read_text())
    ejemplos = [(i["nombre"], e) for i in intenciones for e in i["ejemplos"]]
    emb = _modelo_emb().encode([e for _, e in ejemplos], normalize_embeddings=True)
    return {i["nombre"]: i for i in intenciones}, [n for n, _ in ejemplos], emb


@lru_cache
def _diccionario():
    etiqueta = lambda clase, prop="rdfs:label": [
        f["l"] for f in _sparql(f"SELECT DISTINCT ?l WHERE {{ ?x a tv:{clase} ; {prop} ?l }}")]
    return {"producto": etiqueta("Producto", "tv:nombre"), "categoria": etiqueta("Categoria"),
            "subcategoria": etiqueta("Subcategoria"), "pais": etiqueta("Pais")}


# ---------------------------------------------------------------- utilidades de texto
def _norm(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return re.sub(r"[^a-z0-9 ]", " ", "".join(c for c in texto if not unicodedata.combining(c)))


def buscar_entidad(texto: str, candidatos: list[str]) -> str | None:
    """Candidato con mayor fracción de sus palabras presentes en el texto (≥ 0.6)."""
    palabras = set(_norm(texto).split())
    palabras |= {w[:-2] for w in palabras if w.endswith("es")} | {w[:-1] for w in palabras if w.endswith("s")}
    mejor, mejor_puntaje = None, 0.0
    for c in candidatos:
        tokens = [t for t in _norm(c).split() if t not in {"y", "de"}]
        puntaje = sum(t in palabras for t in tokens) / len(tokens)
        if puntaje > mejor_puntaje or (puntaje == mejor_puntaje and mejor
                                       and len(tokens) > len(_norm(mejor).split())):
            mejor, mejor_puntaje = c, puntaje
    return mejor if mejor_puntaje >= 0.6 else None


def resolver_producto(texto: str) -> dict | None:
    """Nombre aproximado → ficha del producto en la ontología (id, código, nombre)."""
    nombre = buscar_entidad(texto, _diccionario()["producto"])
    if nombre is None:
        return None
    fila = _sparql("SELECT ?p ?codigo WHERE { ?p tv:nombre ?nombre ; tv:codigo ?codigo }", nombre=nombre)[0]
    return {"id": int(str(fila["p"]).rsplit("_", 1)[-1]), "codigo": fila["codigo"], "nombre": nombre}


# ---------------------------------------------------------------- herramientas: RAG
def buscar_productos(consulta: str, categoria: str | None = None,
                     precio_max: float | None = None, k: int = 5) -> list[dict]:
    filtros = []
    if categoria:
        cat = buscar_entidad(categoria, _diccionario()["categoria"])
        if cat:
            filtros.append({"categoria": cat})
    if precio_max:
        filtros.append({"precio": {"$lte": float(precio_max)}})
    where = None if not filtros else filtros[0] if len(filtros) == 1 else {"$and": filtros}
    col_productos, _ = _colecciones()
    r = col_productos.query(query_texts=[consulta], n_results=k, where=where)
    return [{"nombre": m["nombre"], "categoria": m["categoria"], "subcategoria": m["subcategoria"],
             "precio": m["precio"], "calificacion": m["calificacion"],
             "similitud": round(1 - d, 3)}
            for m, d in zip(r["metadatas"][0], r["distances"][0])]


def buscar_resenas(consulta: str, solo_negativas: bool = False, k: int = 5) -> list[dict]:
    _, col_resenas = _colecciones()
    where = {"calificacion": {"$lte": 2}} if solo_negativas else None
    r = col_resenas.query(query_texts=[consulta], n_results=k, where=where)
    return [{"producto": m["producto"], "pais": m["pais"], "calificacion": m["calificacion"],
             "fecha": m["fecha"], "texto": doc.split(": ", 1)[-1]}
            for m, doc in zip(r["metadatas"][0], r["documents"][0])]


# ---------------------------------------------------------------- herramienta: RAG determinista
def _formatear(intencion: dict, parametros: dict, filas: list[dict]) -> str:
    if not filas or all(v is None for v in filas[0].values()):
        return "No hay resultados para esa consulta en la ontología."
    texto = intencion["encabezado"].format(**{**parametros, **filas[0], "n": len(filas)})
    if intencion["fila"]:
        texto += " " + "; ".join(intencion["fila"].format(**f) for f in filas) + "."
    return texto


def clasificar(pregunta: str) -> tuple[str, float]:
    _, nombres, emb = _intenciones()
    sim = emb @ _modelo_emb().encode(pregunta, normalize_embeddings=True)
    mejor = {}
    for n, s in zip(nombres, sim):
        mejor[n] = max(mejor.get(n, -1.0), float(s))
    nombre = max(mejor, key=mejor.get)
    return nombre, mejor[nombre]


def consultar_ontologia(pregunta: str) -> dict:
    catalogo, _, _ = _intenciones()
    nombre, confianza = clasificar(pregunta)
    intencion = catalogo[nombre]
    numeros = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", pregunta.replace(",", ""))]
    parametros = {}
    for p in intencion["parametros"]:
        if p in ("umbral", "precio_max"):
            parametros[p] = int(numeros[-1]) if numeros else None
        else:
            parametros[p] = buscar_entidad(pregunta, _diccionario()[p])
    faltan = [p for p, v in parametros.items() if v is None]
    if confianza < UMBRAL or faltan or intencion["sparql"] is None:
        motivo = (f"confianza {confianza:.2f} < {UMBRAL}" if confianza < UMBRAL
                  else f"faltan parámetros: {faltan}" if faltan else "pregunta abierta")
        return {"determinista": False, "intencion": nombre, "confianza": round(confianza, 2),
                "motivo": motivo,
                "sugerencia": "Usa buscar_productos o buscar_resenas para esta pregunta."}
    filas = _sparql(intencion["sparql"], **parametros)
    return {"determinista": True, "intencion": nombre, "confianza": round(confianza, 2),
            "parametros": parametros, "sparql": intencion["sparql"],
            "respuesta": _formatear(intencion, parametros, filas)}


# ---------------------------------------------------------------- herramientas: API en vivo
def _api(ruta: str, **params):
    try:
        r = requests.get(f"{TIENDA_URL}{ruta}", params=params, headers={"x-api-key": API_KEY}, timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        raise RuntimeError(f"La Tienda Virtual no responde en {TIENDA_URL}. "
                           "Levántala con `npm run dev` en la carpeta tienda-virtual.") from None
    if r.status_code >= 400:
        raise RuntimeError(f"La API respondió {r.status_code}: {r.text[:200]}")
    return r.json()


def stock_en_vivo(producto: str) -> dict:
    ficha = resolver_producto(producto)
    if ficha is None:
        return {"error": f"No reconozco el producto '{producto}' en el catálogo."}
    try:
        datos = _api(f"/api/productos/{ficha['id']}")
    except RuntimeError as e:
        return {"error": str(e)}
    return {"producto": datos["nombre"], "codigo": datos["codigo"], "stock": datos["stock"],
            "precio": datos["precio"], "fuente": f"{TIENDA_URL}/api/productos/{ficha['id']}",
            "consultado": datetime.now().isoformat(timespec="seconds")}


def ordenes_cliente(cliente_id: int, max_ordenes: int = 5) -> dict:
    try:
        datos = _api("/api/ordenes", clienteId=int(cliente_id), pageSize=max_ordenes)
    except RuntimeError as e:
        return {"error": str(e)}
    return {"cliente_id": int(cliente_id), "total_ordenes": datos["pageInfo"]["total"],
            "ordenes": [{"numero": o["numero"], "fecha": o["fecha"], "total": o["total"],
                         "productos": [it["producto"]["nombre"] for it in o["items"]]}
                        for o in datos["items"]]}


# ---------------------------------------------------------------- recursos (solo lectura)
def esquema_ontologia() -> str:
    return (ONTO / "tienda.ttl").read_text()


def categorias() -> dict:
    filas = _sparql("SELECT ?categoria ?subcategoria WHERE { ?s a tv:Subcategoria ; rdfs:label ?subcategoria ; "
                    "tv:subcategoriaDe/rdfs:label ?categoria } ORDER BY ?categoria ?subcategoria")
    resultado: dict[str, list[str]] = {}
    for f in filas:
        resultado.setdefault(f["categoria"], []).append(f["subcategoria"])
    return resultado
