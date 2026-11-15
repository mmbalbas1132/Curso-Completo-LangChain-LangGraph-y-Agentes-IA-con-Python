"""
Introducción a LCEL y RunnableLambda.

Demuestra cómo envolver funciones de Python en RunnableLambda y encadenarlas
con el operador `|`, aprovechando el contrato Runnable (.invoke, .batch).
"""

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda

load_dotenv()  # Patrón estándar del curso (aquí no consumimos variables).

# ── Paso 1: función lambda ────────────────────────────────────────────────
paso1 = RunnableLambda(lambda x: f"Numero {x}")

# ── Paso 2: función con nombre (más legible y reutilizable) ───────────────
def duplicar_texto(texto: str) -> list[str]:
    """Devuelve el texto recibido duplicado en una lista."""
    return [texto] * 2

paso2 = RunnableLambda(duplicar_texto)

# ── Cadena LCEL ───────────────────────────────────────────────────────────
cadena = paso1 | paso2

# ── Demostración: invoke y batch sobre la MISMA cadena ───────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("invoke(43)")
    print("=" * 50)
    resultado = cadena.invoke(43)
    print(resultado)  # ['Numero 43', 'Numero 43']

    print("\n" + "=" * 50)
    print("batch([1, 2, 3])  ← la misma cadena, otra forma de invocarla")
    print("=" * 50)
    resultados = cadena.batch([1, 2, 3])
    for entrada, salida in zip([1, 2, 3], resultados):
        print(f"  {entrada} → {salida}")