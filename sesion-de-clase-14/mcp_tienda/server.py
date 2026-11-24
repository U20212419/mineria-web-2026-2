"""MCP server de la Tienda Virtual.

Expone como herramientas MCP el RAG semántico, el RAG determinista sobre la ontología y los datos
en vivo de la API REST. Se comunica por stdio (JSON-RPC sobre stdin/stdout):

    python mcp_tienda/server.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # para importar herramientas.py

from mcp.server.mcpserver import MCPServer

import herramientas as h

mcp = MCPServer(
    "tienda-virtual",
    instructions=("Herramientas de 'Mi Tienda Virtual'. Para conteos, promedios, rankings y precios usa "
                  "consultar_ontologia; para descripciones y opiniones usa buscar_productos o buscar_resenas; "
                  "para stock actual usa stock_en_vivo."),
)


# ------------------------------------------------------------------ tools
@mcp.tool()
def buscar_productos(consulta: str, categoria: str | None = None, precio_max: float | None = None) -> list[dict]:
    """Busca productos del catálogo por significado (RAG semántico). Útil para preguntas descriptivas
    como 'audífonos con cancelación de ruido'. Filtros opcionales: categoría y precio máximo en soles."""
    return h.buscar_productos(consulta, categoria, precio_max)


@mcp.tool()
def buscar_resenas(consulta: str, solo_negativas: bool = False) -> list[dict]:
    """Busca reseñas de entrega de clientes por significado. Útil para opiniones y quejas.
    Con solo_negativas=True devuelve solo reseñas de 1 o 2 estrellas."""
    return h.buscar_resenas(consulta, solo_negativas)


@mcp.tool()
def consultar_ontologia(pregunta: str) -> dict:
    """Responde de forma exacta y reproducible preguntas de precios, conteos, promedios, rankings y
    relaciones (ventas por país, reseñas negativas por país, productos en riesgo) con SPARQL sobre la
    ontología de la tienda. Devuelve la respuesta y la consulta usada como evidencia."""
    return h.consultar_ontologia(pregunta)


@mcp.tool()
def stock_en_vivo(producto: str) -> dict:
    """Consulta el stock y precio ACTUAL de un producto en la API de la Tienda Virtual."""
    return h.stock_en_vivo(producto)


@mcp.tool()
def ordenes_cliente(cliente_id: int) -> dict:
    """Devuelve las órdenes más recientes de un cliente (por su id) desde la API de la Tienda Virtual."""
    return h.ordenes_cliente(cliente_id)


# ------------------------------------------------------------------ resources
@mcp.resource("ontologia://esquema", mime_type="text/turtle")
def esquema() -> str:
    """TBox de la ontología (clases y propiedades) en Turtle."""
    return h.esquema_ontologia()


@mcp.resource("tienda://categorias", mime_type="application/json")
def categorias() -> str:
    """Categorías y subcategorías del catálogo."""
    return json.dumps(h.categorias(), ensure_ascii=False)


# ------------------------------------------------------------------ prompts
@mcp.prompt()
def analista_de_producto(producto: str) -> str:
    """Plantilla para analizar un producto combinando datos exactos, opiniones y stock en vivo."""
    return (f"Analiza el producto '{producto}' de Mi Tienda Virtual. Usa consultar_ontologia para su "
            f"precio, buscar_resenas para lo que opinan los clientes y stock_en_vivo para el stock actual. "
            f"Termina con una recomendación de una frase.")


if __name__ == "__main__":
    mcp.run()   # transporte stdio por defecto
