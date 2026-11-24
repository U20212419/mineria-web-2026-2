"""MCP server de la Tienda Virtual + las tools del ejercicio propuesto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import herramientas as h
from server import mcp          # el server base, con sus 5 tools, resources y prompt


@mcp.tool()
def mi_tool_nueva(parametro: str) -> dict:
    """TODO: describe qué hace y cuándo usarla (es lo que lee el LLM para decidir)."""
    # TODO: llamar aquí a tu función (copia su código a herramientas.py o a este archivo)
    return {"pendiente": parametro}


if __name__ == "__main__":
    mcp.run()
