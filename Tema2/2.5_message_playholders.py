# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ➕ Añadido: `MessagesPlaceholder` es una pieza especial. No es un texto,
# es un "hueco" que le dice al template: "aquí va una lista entera de mensajes".


# ==============================================================================
# 2. CONSTRUCCIÓN DEL PROMPT TEMPLATE
# ==============================================================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])

# ➕ Añadido: el orden es CRÍTICO y tiene lógica conversacional:
#   1. "system": le dice al modelo cómo debe comportarse.
#   2. "historial": los turnos previos de la conversación (el contexto).
#   3. "human": el nuevo mensaje que acaba de escribir el usuario.


# ==============================================================================
# 3. SIMULACIÓN DE DATOS
# ==============================================================================
# Simulamos un historial de conversación
historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.")
]

# ➕ Añadido: fíjate en que el historial es una LISTA de objetos, no un string.
# Cada objeto tiene un TIPO (HumanMessage o AIMessage) que indica quién habló.
# Esto es exactamente lo que el `MessagesPlaceholder` espera recibir.
#Usuario, IA no son roles mágicos, son simplemente nombres de variables que hemos usado en la simulación para que el ejemplo sea más claro. En la práctica, el LLM no necesita que los mensajes empiecen con "Usuario:" o "IA:".


# ==============================================================================
# 4. FORMATEO Y EJECUCIÓN
# ==============================================================================
mensajes = chat_prompt.format_messages(
    historial=historial_conversacion,
    pregunta_actual="¿Puedes decirme algo interesante de su arquitectura?"
)

# ➕ Añadido: el diccionario que pasamos a `format_messages` tiene dos claves:
#   - "historial": coincide con el `variable_name` del MessagesPlaceholder.
#   - "pregunta_actual": coincide con el hueco {pregunta_actual} del texto.
# El resultado, `mensajes`, es una lista de 6 objetos BaseMessage:
# 1 System + 4 del historial + 1 Human (la nueva pregunta).


# ==============================================================================
# 5. VISUALIZACIÓN DEL RESULTADO
# ==============================================================================
for m in mensajes:
    print(m.content)

# ➕ Añadido: este bucle imprime el texto de cada mensaje generado.
# Si hiciéramos `print(type(m))`, veríamos SystemMessage, HumanMessage y AIMessage.
# Esto demuestra que el LLM recibirá la conversación perfectamente estructurada.