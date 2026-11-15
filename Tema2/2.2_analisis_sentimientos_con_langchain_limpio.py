"""
Análisis de sentimientos con LangChain (LCEL) — Parte 2: batch.
"""

import os
import json

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud."""
    return text.strip()[:500]


preprocessor = RunnableLambda(preprocess_text)


def generate_summary(text):
    """Genera un resumen conciso del texto."""
    prompt = f"Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content


summary_branch = RunnableLambda(generate_summary)


def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado."""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}

    Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}


sentiment_branch = RunnableLambda(analyze_sentiment)


def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado."""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"],
    }


merger = RunnableLambda(merge_results)

parallel_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch,
})

chain = preprocessor | parallel_analysis | merger

reviews_batch = [
    "Excelente producto, muy satisfecho con la compra",
    "Terrible calidad, no lo recomiendo para nada",
    "Está bien, cumple su función básica pero nada especial",
]

resultado_batch = chain.batch(reviews_batch)

print(resultado_batch)