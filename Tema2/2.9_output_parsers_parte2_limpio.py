"""
Extracción de datos estructurados con Pydantic + LangChain (structured output).
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )


class AnalisisTexto(BaseModel):
    """Esquema de salida para el análisis de un texto."""

    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(
        description="Sentimiento del texto (Positivo, neutro o negativo)"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
structured_llm = llm.with_structured_output(AnalisisTexto)


if __name__ == "__main__":
    texto_prueba = (
        "Me encantó la nueva película de acción, tiene muchos efectos "
        "especiales y emoción."
    )

    resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

    print(resultado.model_dump_json())