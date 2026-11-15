"""
Ejemplo de ChatPromptTemplate con MessagesPlaceholder + invocación real al LLM.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])

historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha."),
]

if __name__ == "__main__":
    # 1) Ver los mensajes exactos que se enviarán al modelo
    print("=" * 60)
    print("Mensajes de entrada (prompt formateado)")
    print("=" * 60)
    mensajes = chat_prompt.format_messages(
        historial=historial_conversacion,
        pregunta_actual="¿Puedes decirme algo interesante de su arquitectura?",
    )
    for m in mensajes:
        print(f"[{type(m).__name__}] {m.content}")

    # 2) Invocar de verdad al LLM vía LCEL
    print("\n" + "=" * 60)
    print("Respuesta del modelo")
    print("=" * 60)
    chain = chat_prompt | llm
    respuesta = chain.invoke({
        "historial": historial_conversacion,
        "pregunta_actual": "¿Puedes decirme algo interesante de su arquitectura?",
    })
    print(respuesta.content)