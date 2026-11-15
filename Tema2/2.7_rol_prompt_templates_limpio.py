"""
Ejemplo de ChatPromptTemplate con sub-plantillas System y Human.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

plantilla_sistema = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
)

plantilla_humano = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    plantilla_sistema,
    plantilla_humano,
])

chain = chat_prompt | llm
respuesta = chain.invoke({
    "rol": "nutricionista",
    "especialidad": "dietas veganas",
    "tono": "profesional pero accesible",
    "tema": "proteínas vegetales",
    "pregunta": "¿Cuáles son las mejores fuentes de proteína vegana para un atleta profesional?",
})

print(respuesta.content)