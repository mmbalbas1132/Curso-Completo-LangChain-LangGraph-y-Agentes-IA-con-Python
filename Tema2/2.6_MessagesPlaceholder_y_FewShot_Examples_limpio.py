"""
Lección: MessagesPlaceholder y Few-Shot Examples — versión funcional.
"""

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "❌ Falta DEEPSEEK_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


# ==============================================================================
# 2. CÓDIGO BASE: Historial de Conversación
# ==============================================================================
print("\n" + "=" * 70)
print("2. CÓDIGO BASE: Historial de Conversación")
print("=" * 70)

chat_prompt_historial = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}"),
])

historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha."),
]

chain_historial = chat_prompt_historial | llm
respuesta_historial = chain_historial.invoke({
    "historial": historial_conversacion,
    "pregunta_actual": "¿Puedes decirme algo interesante de su arquitectura?",
})
print("\nRespuesta del modelo:")
print(respuesta_historial.content)


# ==============================================================================
# 3. CÓDIGO EVOLUCIONADO: Few-Shot Examples
# ==============================================================================
print("\n" + "=" * 70)
print("3. CÓDIGO EVOLUCIONADO: Few-Shot Examples")
print("=" * 70)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO. Responde SOLO con la etiqueta."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto a analizar: {texto_usuario}"),
])

ejemplos_sentimientos = [
    HumanMessage(content="Texto a analizar: Me encanta este producto, es increíble"),
    AIMessage(content="POSITIVO"),
    HumanMessage(content="Texto a analizar: El servicio fue terrible, muy decepcionante"),
    AIMessage(content="NEGATIVO"),
    HumanMessage(content="Texto a analizar: El clima está nublado hoy"),
    AIMessage(content="NEUTRO"),
]

chain_sentimientos = chat_prompt | llm

textos_a_clasificar = [
    "¡Qué día tan maravilloso!",
    "No pienso volver a comprar aquí nunca más.",
    "El tren sale a las 15:30.",
]

for texto in textos_a_clasificar:
    respuesta = chain_sentimientos.invoke({
        "ejemplos": ejemplos_sentimientos,
        "texto_usuario": texto,
    })
    print(f"  {texto!r:55s} → {respuesta.content.strip()}")


# ==============================================================================
# 4. OTROS EJEMPLOS DE USO
# ==============================================================================
print("\n" + "=" * 70)
print("4. OTROS EJEMPLOS DE USO")
print("=" * 70)

# --- Ejemplo A: Extracción de información (NER) ---
ejemplos_extraccion = [
    HumanMessage(content="Texto: Juan Pérez trabaja en Google como ingeniero desde 2020"),
    AIMessage(content="Nombre: Juan Pérez, Empresa: Google, Puesto: ingeniero, Año: 2020"),
    HumanMessage(content="Texto: María Silva es doctora en el Hospital Central"),
    AIMessage(content="Nombre: María Silva, Empresa: Hospital Central, Puesto: doctora, Año: N/A"),
]

prompt_extraccion = ChatPromptTemplate.from_messages([
    ("system", "Extrae entidades del texto. Formato: Nombre, Empresa, Puesto, Año."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto: {texto}"),
])

chain_extraccion = prompt_extraccion | llm
resultado_extraccion = chain_extraccion.invoke({
    "ejemplos": ejemplos_extraccion,
    "texto": "Carlos Ruiz es profesor en la Universidad de Sevilla desde 2015",
})
print("Ejemplo A — Extracción de datos estructurados:")
print(" ", resultado_extraccion.content.strip())

# --- Ejemplo B: Traducción con cambio de tono ---
ejemplos_traduccion = [
    HumanMessage(content="Formal en inglés: Good morning, how are you today?"),
    AIMessage(content="Casual en español: ¡Hola! ¿Qué tal?"),
    HumanMessage(content="Formal en inglés: I would like to schedule a meeting"),
    AIMessage(content="Casual en español: ¿Podemos quedar?"),
]

prompt_traduccion = ChatPromptTemplate.from_messages([
    ("system", "Traduce del inglés formal al español casual. Responde solo con la traducción."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Formal en inglés: {texto}"),
])

chain_traduccion = prompt_traduccion | llm
resultado_traduccion = chain_traduccion.invoke({
    "ejemplos": ejemplos_traduccion,
    "texto": "Would you be available for a brief call tomorrow afternoon?",
})
print("\nEjemplo B — Traducción con cambio de tono:")
print(" ", resultado_traduccion.content.strip())