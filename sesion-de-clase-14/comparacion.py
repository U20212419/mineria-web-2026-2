"""Utilidades para comparar respuestas "con vs. sin" una tecnología.

Todos los notebooks de la sesión responden el mismo banco de preguntas
(`data/preguntas_prueba.csv`). Cada pregunta trae una lista de términos que
una respuesta correcta debe contener (`esperado`, separados por `|`).

La evaluación es deliberadamente simple y transparente:
- correcta   -> contiene todos los términos esperados
- parcial    -> contiene al menos uno
- incorrecta -> no contiene ninguno
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
DATA = RAIZ / "data"
SALIDAS = RAIZ / "salidas"
SALIDAS.mkdir(exist_ok=True)

COLORES = {"correcta": "#d4edda", "parcial": "#fff3cd", "incorrecta": "#f8d7da"}


def cargar_banco() -> pd.DataFrame:
    return pd.read_csv(DATA / "preguntas_prueba.csv")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin separadores de miles (5,599 -> 5599)."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c)).lower()
    texto = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", texto)
    texto = re.sub(r"(?<=\d),(?=\d{1,2}\b)", ".", texto)  # coma decimal: 3,44 -> 3.44
    return texto


def _contiene(texto: str, termino: str) -> bool:
    # Los números se comparan como número completo: "4" no coincide con "46".
    if re.fullmatch(r"\d+(\.\d+)?", termino):
        return re.search(rf"(?<![\d.]){re.escape(termino)}(?![\d])", texto) is not None
    return termino in texto


def evaluar(respuesta: str, esperado: str) -> str:
    terminos = [normalizar(t) for t in str(esperado).split("|") if t.strip()]
    r = normalizar(respuesta)
    encontrados = sum(_contiene(r, t) for t in terminos)
    if encontrados == len(terminos):
        return "correcta"
    return "parcial" if encontrados else "incorrecta"


def responder_banco(nombre: str, funcion, banco: pd.DataFrame | None = None,
                    repeticiones: int = 1) -> pd.DataFrame:
    """Ejecuta `funcion(pregunta) -> str` sobre el banco y evalúa cada respuesta.

    Con `repeticiones > 1` mide además si la respuesta es idéntica en cada
    ejecución (repetibilidad).
    """
    banco = cargar_banco() if banco is None else banco
    filas = []
    for _, q in banco.iterrows():
        salidas, t0 = [], time.perf_counter()
        for _ in range(repeticiones):
            salidas.append(str(funcion(q.pregunta)))
        latencia = (time.perf_counter() - t0) / repeticiones
        filas.append({
            "id": q.id, "tipo": q.tipo, "pregunta": q.pregunta,
            "configuracion": nombre, "respuesta": salidas[0],
            "resultado": evaluar(salidas[0], q.esperado),
            "repetible": len(set(salidas)) == 1 if repeticiones > 1 else None,
            "latencia_s": round(latencia, 2),
        })
    df = pd.DataFrame(filas)
    df.to_csv(SALIDAS / f"resultados_{nombre}.csv", index=False)
    return df


def registrar(nombre: str, respuestas: list[str], latencias: list[float],
              banco: pd.DataFrame | None = None, extra: dict | None = None) -> pd.DataFrame:
    """Igual que `responder_banco`, pero con respuestas ya calculadas (p. ej. con `await`)."""
    banco = cargar_banco() if banco is None else banco
    df = pd.DataFrame({
        "id": banco.id, "tipo": banco.tipo, "pregunta": banco.pregunta, "configuracion": nombre,
        "respuesta": respuestas,
        "resultado": [evaluar(r, e) for r, e in zip(respuestas, banco.esperado)],
        "repetible": None, "latencia_s": [round(t, 2) for t in latencias],
        **(extra or {}),
    })
    df.to_csv(SALIDAS / f"resultados_{nombre}.csv", index=False)
    return df


def lado_a_lado(*resultados: pd.DataFrame, ancho: int = 220):
    """Tabla pregunta x configuración, coloreada según el resultado."""
    todos = pd.concat(resultados)
    respuestas = todos.pivot(index=["id", "pregunta"], columns="configuracion",
                             values="respuesta")
    estados = todos.pivot(index=["id", "pregunta"], columns="configuracion",
                          values="resultado")
    orden = [r.configuracion.iloc[0] for r in resultados]
    respuestas, estados = respuestas[orden], estados[orden]
    recortar = lambda s: s if len(s) <= ancho else s[:ancho] + "…"
    tabla = respuestas.map(recortar)
    estilo = estados.map(lambda e: f"background-color: {COLORES[e]}; color: #222")
    return (tabla.style.apply(lambda _: estilo, axis=None)
            .set_properties(**{"text-align": "left", "white-space": "pre-wrap",
                               "vertical-align": "top"}))


def resumen(*resultados: pd.DataFrame) -> pd.DataFrame:
    """Conteo de correctas / parciales / incorrectas por configuración."""
    todos = pd.concat(resultados)
    tabla = (todos.groupby(["configuracion", "resultado"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["correcta", "parcial", "incorrecta"], fill_value=0))
    tabla["latencia_media_s"] = todos.groupby("configuracion").latencia_s.mean().round(2)
    if todos.repetible.notna().any():
        tabla["repetibles"] = todos.groupby("configuracion").repetible.sum()
    return tabla.reindex([r.configuracion.iloc[0] for r in resultados])
