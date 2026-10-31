"""
================================================================================
MATERIAL DE ESTUDIO: MESSAGESPLACEHOLDER Y FEW-SHOT EXAMPLES
================================================================================
En esta lección exploramos cómo evolucionar el uso de `MessagesPlaceholder`.
Si en la lección anterior lo usábamos para inyectar el HISTORIAL de una
conversación, ahora lo usaremos para algo mucho más potente: inyectar
EJEMPLOS (Few-Shot Examples) para enseñarle al modelo "en tiempo real".

CONCEPTOS CLAVE DE ESTA LECCIÓN:
    1. In-Context Learning (ICL) → Enseñar al modelo mediante ejemplos en el prompt.
    2. Few-Shot Examples       → Dar de 2 a 5 ejemplos (Input → Output) para guiar al modelo.
    3. MessagesPlaceholder     → La herramienta ideal para inyectar estos ejemplos manteniendo
                                 la estructura nativa de roles (Human/AI).

ESTRUCTURA TÍPICA DE UN PROMPT FEW-SHOT:
    - Sistema: Instrucciones generales (el "trabajo" a realizar).
    - Ejemplo 1: Input → Output esperado
    - Ejemplo 2: Input → Output esperado
    - Pregunta real: Input actual → (El modelo debe predecir el Output)

FILOSOFÍA DEL ESTUDIO:
    • Encabezados conceptuales ("¿por qué?").
    • Código comentado ("¿cómo?").
    • Notas "🔎 Para profundizar" con matices y errores comunes.
================================================================================
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ==============================================================================
# 2. CÓDIGO BASE: HISTORIAL DE CONVERSACIÓN (Lección anterior)
# ==============================================================================
# Este es el patrón que ya conocemos. Lo dejamos aquí como referencia para
# comparar cómo el MISMO concepto (MessagesPlaceholder) sirve para dos
# propósitos distintos: memoria y ejemplos.
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("2. CÓDIGO BASE: Historial de Conversación")
print("="*70)

chat_prompt_historial = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])

historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.")
]

mensajes_historial = chat_prompt_historial.format_messages(
    historial=historial_conversacion,
    pregunta_actual="¿Puedes decirme algo interesante de su arquitectura?"
)

# 🔎 Para profundizar:
#   Aquí inyectamos una "conversación pasada". El modelo usa esa memoria
#   para entender que "su arquitectura" se refiere a París.


# ==============================================================================
# 3. CÓDIGO EVOLUCIONADO: FEW-SHOT EXAMPLES (Lo nuevo)
# ==============================================================================
# El mismo mecanismo, pero ahora los mensajes inyectados no son una
# conversación real, sino un "entrenamiento" rápido con ejemplos.
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("3. CÓDIGO EVOLUCIONADO: Few-Shot Examples")
print("="*70)

# Template para clasificación de sentimientos con few-shot examples
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto a analizar: {texto_usuario}")
])

# Few-shot examples para análisis de sentimientos
# ¿POR QUÉ ASÍ? Porque el modelo "ve" la estructura exacta de una conversación.
# Ve que cuando un humano dice "Texto a analizar: X", la IA responde con
# una única palabra en mayúsculas. Es un molde perfecto para imitar.
ejemplos_sentimientos = [
    HumanMessage(content="Texto a analizar: Me encanta este producto, es increíble"),
    AIMessage(content="POSITIVO"),
    HumanMessage(content="Texto a analizar: El servicio fue terrible, muy decepcionante"),
    AIMessage(content="NEGATIVO"),
    HumanMessage(content="Texto a analizar: El clima está nublado hoy"),
    AIMessage(content="NEUTRO")
]

# Generar el prompt con los ejemplos
mensajes = chat_prompt.format_messages(
    ejemplos=ejemplos_sentimientos,
    texto_usuario="¡Qué día tan maravilloso!"
)

# Ver el resultado
# Observa cómo la lista resultante es una mezcla de instrucciones, ejemplos
# y la pregunta real, todo perfectamente estructurado en roles.
for i, m in enumerate(mensajes):
    print(f"Mensaje {i+1} ({m.__class__.__name__}):")
    print(m.content)
    print("-" * 40)


# ==============================================================================
# 4. OTROS EJEMPLOS DE USO (Casos prácticos en la industria)
# ==============================================================================
# Aquí tienes otros patrones comunes donde brillan los Few-Shot Examples
# inyectados con MessagesPlaceholder.

print("\n" + "="*70)
print("4. OTROS EJEMPLOS DE USO (Impresos en código)")
print("="*70)

# --- Ejemplo A: Extracción de Información (NER - Named Entity Recognition) ---
ejemplos_extraccion = [
    HumanMessage(content="Texto: Juan Pérez trabaja en Google como ingeniero desde 2020"),
    AIMessage(content="Nombre: Juan Pérez, Empresa: Google, Puesto: ingeniero, Año: 2020"),
    HumanMessage(content="Texto: María Silva es doctora en el Hospital Central"),
    AIMessage(content="Nombre: María Silva, Empresa: Hospital Central, Puesto: doctora, Año: N/A")
]
print("Ejemplo A: Extracción de datos estructurados de texto libre.")

# --- Ejemplo B: Traducción con Estilo ---
ejemplos_traduccion = [
    HumanMessage(content="Formal en inglés: Good morning, how are you today?"),
    AIMessage(content="Casual en español: ¡Hola! ¿Qué tal?"),
    HumanMessage(content="Formal en inglés: I would like to schedule a meeting"),
    AIMessage(content="Casual en español: ¿Podemos quedar?")
]
print("Ejemplo B: Traducción cambiando el tono (de formal a casual).")


# ==============================================================================
# 5. RESUMEN DE VENTAJAS: ¿POR QUÉ USAR MessagesPlaceholder PARA ESTO?
# ==============================================================================
#
# CON MessagesPlaceholder:
#   ✅ Estructura clara: Cada ejemplo mantiene su rol (Human/AI).
#   ✅ Escalable: Es fácil añadir o quitar ejemplos de la lista de Python.
#   ✅ Reutilizable: Cambiar la lista de ejemplos = crear una tarea nueva.
#   ✅ Natural: El modelo entiende el formato conversacional de los LLMs modernos.
#
# SIN MessagesPlaceholder (usando texto plano concatenado):
#   ❌ Menos claro: Todo mezclado en un solo bloque de texto gigante.
#   ❌ Más errores: El modelo puede confundirse y alucinar el formato.
#   ❌ Difícil mantenimiento: Cambiar los ejemplos requiere reescribir el string.
#   ❌ Menos efectivo: Pierde la separación explícita de turnos que la API de chat espera.
#
# ==============================================================================
# 6. MEJORES PRÁCTICAS PARA FEW-SHOT EXAMPLES
# ==============================================================================
#
# 1. Cantidad (2-5 ejemplos): Es suficiente para mostrar el patrón sin
#    reventar la ventana de contexto ni disparar el coste en tokens.
# 2. Diversidad: Cubre diferentes casos o variaciones del problema
#    (ej. un positivo muy claro, un negativo sutil, un neutro).
# 3. Formato consistente: El mismo patrón exacto en TODOS los ejemplos
#    (si un Output no tiene punto final, ninguno debería tenerlo).
# 4. Ejemplos de calidad: Los Outputs deben ser perfectos, porque el
#    modelo tenderá a copiar exactamente ese estilo y formato.
#
# ==============================================================================