"""Agente de la Tienda Virtual: cliente MCP + Qwen2.5 (local) con tool-calling.

Dos estrategias para comparar:

- "ingenua": se le pasan todas las tools a Qwen, se parsea su <tool_call> tal cual y Qwen redacta la
  respuesta final con el resultado. Es lo que haría un agente estándar con un modelo grande.
- "robusta": técnicas para que un modelo de 0.5B use tools de forma fiable:
    1. pre-router determinista (intenciones del Ejercicio 2): si reconoce la intención, llama la
       tool exacta sin pasar por el LLM
    2. pocas tools expuestas al LLM cuando decide él, con ejemplos (few-shot)
    3. parseo tolerante del JSON + normalización de nombres y tipos de argumentos
    4. anclaje: se descartan argumentos que no aparecen en la pregunta (números o categorías inventadas)
       y se valida contra el inputSchema que publica el MCP server
    5. reintento con el error de validación como retroalimentación (máx. 2)
    6. fallback a una tool segura si todo falla
    7. respuestas exactas (ontología, stock en vivo) con plantilla, sin reescritura del LLM

Uso (en un notebook, con `await`):

    agente = AgenteTienda(llm, tokenizer, estrategia="robusta")
    await agente.conectar()
    r = await agente.preguntar("¿Cuánto stock queda del DJI Air 3?")
    await agente.cerrar()
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
import asyncio
from pathlib import Path

import jsonschema
import torch
from mcp import Client, StdioServerParameters, stdio_client

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import herramientas as h  # noqa: E402  (router e intenciones del Ejercicio 2)

SISTEMA = ("Eres el asistente de 'Mi Tienda Virtual', una tienda peruana de tecnología. "
           "Responde en español, en pocas frases.")

# Qué tool atiende cada intención determinista del Ejercicio 2
INTENCION_A_TOOL = {"stock_producto": "stock_en_vivo"}   # el stock siempre se consulta en vivo

FEW_SHOT = [
    {"role": "user", "content": "¿Qué cámaras sirven para grabar deportes extremos?"},
    {"role": "assistant", "content": '<tool_call>\n{"name": "buscar_productos", "arguments": {"consulta": "cámara para grabar deportes extremos"}}\n</tool_call>'},
    {"role": "user", "content": "¿Qué opinan los clientes sobre el empaque?"},
    {"role": "assistant", "content": '<tool_call>\n{"name": "buscar_resenas", "arguments": {"consulta": "opiniones sobre el empaque"}}\n</tool_call>'},
]


def resultado_tool(r) -> object:
    """Contenido de un CallToolResult como objeto Python."""
    if r.structured_content is not None:
        return r.structured_content.get("result", r.structured_content)
    texto = "".join(getattr(c, "text", "") for c in r.content)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return texto


def _esquema_simple(schema: dict) -> dict:
    """Simplifica el JSON Schema de MCP para el chat template de Qwen (sin anyOf/null ni títulos)."""
    props = {}
    for nombre, p in schema.get("properties", {}).items():
        tipo = p.get("type") or next((a["type"] for a in p.get("anyOf", []) if a.get("type") != "null"), "string")
        props[nombre] = {"type": tipo}
    return {"type": "object", "properties": props, "required": schema.get("required", [])}


class AgenteTienda:
    def __init__(self, llm, tokenizer, estrategia: str = "robusta", servidor: Path | None = None,
                 log: Path | None = None, verbose: bool = False):
        assert estrategia in ("ingenua", "robusta")
        self.llm, self.tok, self.estrategia, self.verbose = llm, tokenizer, estrategia, verbose
        self.servidor = servidor or CARPETA / "server.py"
        self.log = log or CARPETA.parent / "salidas" / "mcp_server.log"
        self.tools: dict[str, dict] = {}

    # ------------------------------------------------------------------ conexión MCP
    async def conectar(self):
        """Lanza el server por stdio y descubre sus tools.

        La conexión vive en una tarea asyncio propia: en Jupyter cada celda corre en una tarea
        distinta y anyio exige cerrar la conexión en la misma tarea que la abrió.
        """
        params = StdioServerParameters(command=sys.executable, args=[str(self.servidor)])
        listo, self._cerrar = asyncio.get_running_loop().create_future(), asyncio.Event()

        async def mantener():
            with open(self.log, "a") as errlog:
                try:
                    async with Client(stdio_client(params, errlog=errlog)) as cliente:
                        listo.set_result(cliente)
                        await self._cerrar.wait()
                except BaseException as e:
                    if not listo.done():
                        listo.set_exception(e)
                    else:
                        raise

        self._tarea = asyncio.create_task(mantener())
        self.cliente = await listo
        self.tools = {t.name: {"descripcion": t.description, "schema": t.input_schema}
                      for t in (await self.cliente.list_tools()).tools}
        return list(self.tools)

    async def cerrar(self):
        self._cerrar.set()
        await self._tarea

    def _tools_qwen(self, nombres):
        return [{"type": "function", "function": {"name": n, "description": self.tools[n]["descripcion"],
                                                  "parameters": _esquema_simple(self.tools[n]["schema"])}}
                for n in nombres]

    # ------------------------------------------------------------------ LLM
    def _generar(self, mensajes, tools=None, max_new_tokens=200) -> str:
        entrada = self.tok.apply_chat_template(mensajes, tools=tools, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        with torch.no_grad():
            salida = self.llm.generate(**entrada, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ parseo y validación
    @staticmethod
    def _parsear_estricto(texto):
        m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", texto, re.S)
        if not m:
            return None
        llamada = json.loads(m.group(1))
        return llamada["name"], llamada.get("arguments", {})

    def _parsear_tolerante(self, texto):
        bloque = re.search(r"\{.*\}", texto, re.S)          # primer objeto JSON, con o sin <tool_call>
        if not bloque:
            return None
        crudo = bloque.group(0)
        for intento in (crudo, crudo.replace("'", '"'), re.sub(r",\s*}", "}", crudo)):
            try:
                llamada = json.loads(intento)
                break
            except json.JSONDecodeError:
                llamada = None
        if not isinstance(llamada, dict):
            return None
        nombre = llamada.get("name") or llamada.get("tool") or ""
        args = llamada.get("arguments") or llamada.get("parameters") or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip().startswith("{") else {}
        nombre = (difflib.get_close_matches(nombre, list(self.tools), n=1, cutoff=0.6) or [nombre])[0]
        return nombre, args

    def _normalizar_args(self, nombre, args):
        props = self.tools[nombre]["schema"].get("properties", {})
        limpio = {}
        for clave, valor in args.items():
            clave = (difflib.get_close_matches(clave, list(props), n=1, cutoff=0.6) or [None])[0]
            if clave is None or valor in (None, ""):
                continue
            tipo = _esquema_simple({"properties": {clave: props[clave]}})["properties"][clave]["type"]
            try:
                if tipo == "number":
                    valor = float(re.sub(r"[^\d.]", "", str(valor)))
                elif tipo == "integer":
                    valor = int(float(re.sub(r"[^\d.]", "", str(valor))))
                elif tipo == "boolean" and isinstance(valor, str):
                    valor = valor.lower() in ("true", "si", "sí", "1")
            except ValueError:
                continue
            limpio[clave] = valor
        return limpio

    @staticmethod
    def _anclar_args(args, pregunta):
        """Descarta argumentos que el modelo inventó: números o categorías que no están en la pregunta."""
        numeros = set(re.findall(r"\d+", pregunta.replace(",", "")))
        anclados = {}
        for clave, valor in args.items():
            if isinstance(valor, (int, float)) and not isinstance(valor, bool):
                if str(int(valor)) not in numeros:
                    continue
            if clave == "categoria":
                valor = h.buscar_entidad(pregunta, h._diccionario()["categoria"])
                if valor is None:
                    continue
            anclados[clave] = valor
        return anclados

    def _validar(self, nombre, args):
        if nombre not in self.tools:
            return f"La herramienta '{nombre}' no existe. Opciones: {list(self.tools)}"
        try:
            jsonschema.validate(args, self.tools[nombre]["schema"])
        except jsonschema.ValidationError as e:
            return f"Argumentos inválidos para {nombre}: {e.message}"
        return None

    # ------------------------------------------------------------------ redacción de la respuesta
    def _redactar(self, pregunta, nombre, datos) -> str:
        """Respuestas exactas con plantilla; el LLM solo redacta sobre resultados de búsqueda."""
        if isinstance(datos, dict) and "error" in datos:
            return f"No pude completar la consulta: {datos['error']}"
        if nombre == "consultar_ontologia" and datos.get("determinista"):
            return datos["respuesta"]
        if nombre == "stock_en_vivo":
            return (f"Stock actual de {datos['producto']}: {datos['stock']} unidades "
                    f"(precio S/ {datos['precio']}, consultado en la API a las {datos['consultado'][-8:]}).")
        contexto = json.dumps(datos, ensure_ascii=False)[:2500]
        return self._generar([{"role": "system", "content": SISTEMA},
                              {"role": "user", "content": f"Datos obtenidos de la tienda:\n{contexto}\n\n"
                               f"Con esos datos, responde en una o dos frases y menciona los nombres tal "
                               f"como aparecen.\n\nPregunta: {pregunta}"}])

    async def _llamar(self, nombre, args, traza):
        r = await self.cliente.call_tool(nombre, args)
        datos = resultado_tool(r)
        traza.append({"paso": "call_tool", "tool": nombre, "args": args, "error_mcp": r.is_error,
                      "resultado": str(datos)[:300]})
        return datos

    # ------------------------------------------------------------------ estrategias
    async def _ingenua(self, pregunta, traza):
        mensajes = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": pregunta}]
        salida = self._generar(mensajes, tools=self._tools_qwen(self.tools))
        traza.append({"paso": "llm", "salida": salida})
        try:
            llamada = self._parsear_estricto(salida)
        except (json.JSONDecodeError, KeyError):
            llamada = None
        if llamada is None:
            traza.append({"paso": "sin_tool"})
            return None, salida
        nombre, args = llamada
        if self._validar(nombre, args):
            traza.append({"paso": "invalida", "detalle": self._validar(nombre, args)})
            return nombre, "No pude usar la herramienta correctamente."
        datos = await self._llamar(nombre, args, traza)
        mensajes += [{"role": "assistant", "content": salida},
                     {"role": "tool", "content": json.dumps(datos, ensure_ascii=False)[:2500]}]
        return nombre, self._generar(mensajes)

    async def _robusta(self, pregunta, traza):
        # 1. Pre-router determinista
        intencion, confianza = h.clasificar(pregunta)
        traza.append({"paso": "router", "intencion": intencion, "confianza": round(confianza, 2)})
        if intencion != "pregunta_abierta" and confianza >= h.UMBRAL:
            tool = INTENCION_A_TOOL.get(intencion, "consultar_ontologia")
            args = {"producto": pregunta} if tool == "stock_en_vivo" else {"pregunta": pregunta}
            datos = await self._llamar(tool, args, traza)
            if not (tool == "consultar_ontologia" and not datos.get("determinista")):
                return tool, self._redactar(pregunta, tool, datos)
            traza.append({"paso": "router_sin_resultado", "motivo": datos.get("motivo")})

        # 2. El LLM elige solo entre las tools de búsqueda, con few-shot
        opciones = ["buscar_productos", "buscar_resenas"]
        mensajes = [{"role": "system", "content": SISTEMA}, *FEW_SHOT, {"role": "user", "content": pregunta}]
        for intento in range(3):
            salida = self._generar(mensajes, tools=self._tools_qwen(opciones), max_new_tokens=120)
            traza.append({"paso": f"llm_intento_{intento + 1}", "salida": salida})
            llamada = self._parsear_tolerante(salida)                       # 3. parseo tolerante
            if llamada is None:
                error = "No encontré un bloque <tool_call> con JSON válido."
            else:
                nombre, args = llamada
                if nombre in self.tools:
                    args = self._anclar_args(self._normalizar_args(nombre, args), pregunta)
                error = self._validar(nombre, args)                          # 4. anclaje + validación
                if error is None and nombre not in opciones:
                    error = f"Usa una de estas herramientas: {opciones}"
            if error is None:
                datos = await self._llamar(nombre, args, traza)
                return nombre, self._redactar(pregunta, nombre, datos)
            traza.append({"paso": "error", "detalle": error})
            mensajes += [{"role": "assistant", "content": salida},             # 5. reintento
                         {"role": "user", "content": f"Error: {error} Responde solo con <tool_call>"
                          '{"name": ..., "arguments": {...}}</tool_call>.'}]

        # 6. Fallback seguro
        nombre = "buscar_resenas" if re.search(r"rese|queja|opin", pregunta, re.I) else "buscar_productos"
        traza.append({"paso": "fallback", "tool": nombre})
        datos = await self._llamar(nombre, {"consulta": pregunta}, traza)
        return nombre, self._redactar(pregunta, nombre, datos)

    async def preguntar(self, pregunta: str) -> dict:
        traza, t0 = [], time.perf_counter()
        metodo = self._robusta if self.estrategia == "robusta" else self._ingenua
        tool, respuesta = await metodo(pregunta, traza)
        resultado = {"pregunta": pregunta, "tool": tool, "respuesta": respuesta, "traza": traza,
                     "latencia_s": round(time.perf_counter() - t0, 2)}
        if self.verbose:
            mostrar_traza(resultado)
        return resultado


def mostrar_traza(r: dict):
    print(f"❓ {r['pregunta']}")
    for paso in r["traza"]:
        detalle = {k: v for k, v in paso.items() if k != "paso"}
        print(f"   · {paso['paso']:<20} {json.dumps(detalle, ensure_ascii=False)[:180]}")
    print(f"🤖 {r['respuesta']}  ({r['latencia_s']} s)\n")
