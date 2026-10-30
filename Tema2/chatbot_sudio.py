"""
================================================================================
MATERIAL DE ESTUDIO: CHATBOT CON ChatPromptTemplate + DEEPSEEK + STREAMLIT
================================================================================
Este archivo parte del chatbot básico de DeepSeek y lo evoluciona para usar
`ChatPromptTemplate`, siguiendo las mejores prácticas de la tarea:

    ANTES (chatbot básico)                      AHORA (esta versión)
    ──────────────────────────────────────      ─────────────────────────────
    chat_model.invoke(messages)                 cadena.stream({...})
    Sin system message                          System message por personalidad
    Historial = lista directa                   Historial = MessagesPlaceholder
    Sin streaming                               Streaming con st.empty()
    Sin configuración dinámica                  Sidebar con modelo/temp/pers.

CONCEPTOS QUE SE PRACTICAN
--------------------------
    1. ChatPromptTemplate        → estructura system/human/placeholder
    2. MessagesPlaceholder       → insertar el historial como LISTA nativa
    3. Personalidad dinámica     → dict {personalidad: system_message}
    4. Streaming                 → cadena.stream(...) + st.empty()
    5. session_state             → persistencia entre reruns de Streamlit
    6. Cabecera estándar         → load_dotenv() al inicio (patrón del curso)

FILOSOFÍA DEL ESTUDIO
---------------------
Igual que en las partes anteriores:
    • Encabezados conceptuales ("¿por qué?").
    • Código comentado ("¿cómo?").
    • Notas "🔎 Para profundizar" con matices y errores comunes.

EJECUCIÓN
---------
    streamlit run chatbot_estudio.py
    ⚠️ NO uses `python chatbot_estudio.py`: Streamlit necesita su runner.
================================================================================
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
# ORDEN: stdlib → terceros → locales (PEP 8).
#
# ¿POR QUÉ `langchain_core` Y NO `langchain.prompts`?
#   `langchain.prompts` es la ruta ANTIGUA (pre-0.2). En el curso usamos
#   `langchain_core`, que es el paquete base oficial y el que verás en la
#   documentación actual. Mapeo de rutas legacy → modernas:
#
#       langchain.prompts        →  langchain_core.prompts
#       langchain.schema         →  langchain_core.messages
# ------------------------------------------------------------------------------
import os                                       # Para comprobar la API key
from dotenv import load_dotenv                   # Cargar .env → os.environ

import streamlit as st                           # Framework de UI

from langchain_deepseek import ChatDeepSeek      # Tu proveedor (DeepSeek)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,                         # ← Para el historial
)
from langchain_core.messages import (
    SystemMessage,                               # Para render
    HumanMessage,                                # Para append
    AIMessage,                                   # Para render y append
)

# 🔎 Para profundizar
#   • `MessagesPlaceholder` es la PIEZA CLAVE que diferencia esta versión
#     de un simple `ChatPromptTemplate`. Permite insertar una LISTA de
#     mensajes nativos (no un string con el historial concatenado).
#   • `langchain_deepseek` es un paquete satélite, igual que
#     `langchain_openai` o `langchain_anthropic`. Todos exponen una clase
#     `ChatXxx` con la misma interfaz (`invoke`, `stream`, etc.). Eso es
#     lo que hace LCEL agnóstico al proveedor.


# ==============================================================================
# 2. CARGA DE VARIABLES DE ENTORNO
# ==============================================================================
# ¿POR QUÉ AQUÍ Y NO DENTRO DE UNA FUNCIÓN?
#   `load_dotenv()` debe ejecutarse UNA VEZ al arrancar. Streamlit re-ejecuta
#   el script entero en cada interacción, pero `load_dotenv()` es idempotente
#   (no sobreescribe lo ya cargado).
#
# ⚠️ REGLA DE ORO DE STREAMLIT:
#   `st.set_page_config()` DEBE ser la primera llamada `st.*` del script.
#   Por eso `load_dotenv()` va ANTES (no es `st.*`) y `set_page_config` justo
#   después. Si lo pusieras al revés, Streamlit lanzaría un error.
# ------------------------------------------------------------------------------
load_dotenv()


# ==============================================================================
# 3. CONFIGURACIÓN INICIAL DE STREAMLIT
# ==============================================================================
# ¿QUÉ CONFIGURA `set_page_config`?
#   • page_title        → título de la pestaña del navegador
#   • page_icon         → emoji de la pestaña
#   • layout="wide"     → usa todo el ancho (mejor para chats)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Chatbot con DeepSeek",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")


# ==============================================================================
# 4. VERIFICACIÓN DE LA API KEY (fail-fast "amable" para Streamlit)
# ==============================================================================
# ¿POR QUÉ COMPROBAR AQUÍ?
#   Sin `DEEPSEEK_API_KEY`, la primera llamada a `cadena.stream(...)` fallaría
#   con un error críptico. Preferimos detectarlo ANTES y mostrar un mensaje
#   claro + detener la app con `st.stop()`.
#
# DIFERENCIA CON LAS PARTES 1 Y 2:
#   Allí usábamos `raise RuntimeError(...)`. Aquí `st.error() + st.stop()`
#   porque es una app de usuario, no un script CLI.
# ------------------------------------------------------------------------------
if not os.getenv("DEEPSEEK_API_KEY"):
    st.error(
        "❌ Falta DEEPSEEK_API_KEY. Defínela en tu archivo `.env` "
        "o expórtala en el entorno antes de arrancar Streamlit."
    )
    st.stop()  # Detiene el script aquí (no lanza excepción).


# ==============================================================================
# 5. SIDEBAR: CONFIGURACIÓN + PERSONALIDAD DINÁMICA
# ==============================================================================
# ¿POR QUÉ TODO DENTRO DE `with st.sidebar:`?
#   `with st.sidebar:` es un contexto: todo lo que se cree dentro aparece
#   en la barra lateral. Es azúcar sintáctico de `st.sidebar.slider(...)`.
#
# ⚠️ IMPORTANTE (Streamlit re-ejecuta el script):
#   Este bloque se ejecuta ENTERO en cada interacción. Eso significa que
#   `chat_model` y `chat_prompt` se RECREAN en cada rerun. No es un problema
#   (son objetos baratos), pero SÍ es crucial entenderlo: si el usuario
#   cambia la temperatura, el nuevo valor se aplica en la SIGUIENTE
#   interacción, no retroactivamente.
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    # --- 5.1 Controles de parámetros del modelo ---
    temperature = st.slider(
        "Temperatura", 0.0, 1.0, 0.7, 0.1
        # min, max, default, step
        # 0.0 → determinista;  0.7 → equilibrado (default chat);  1.0 → creativo
    )

    model_name = st.selectbox(
        "Modelo",
        ["deepseek-chat", "deepseek-reasoner"],
        # deepseek-chat     → rápido, general-purpose
        # deepseek-reasoner → razonamiento profundo (más lento y caro)
    )

    # --- 5.2 Personalidad configurable (el "system" del prompt) ---
    personalidad = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal",
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido",
        ],
    )

    # --- 5.3 Diccionario {personalidad: mensaje_sistema} ---
    # ¿POR QUÉ UN DICT Y NO UN IF/ELIF EN CADENA?
    #   • Legibilidad: cada personalidad ↔ su mensaje, en una línea.
    #   • Extensibilidad: añadir una personalidad = añadir una entrada.
    #   • Reutilización: el mismo dict sirve para mostrar, loggear, testear.
    system_messages = {
        "Útil y amigable": (
            "Eres un asistente útil y amigable llamado ChatBot Pro. "
            "Responde de manera clara y concisa."
        ),
        "Profesional y formal": (
            "Eres un asistente profesional y formal. "
            "Proporciona respuestas precisas y bien estructuradas."
        ),
        "Casual y relajado": (
            "Eres un asistente casual y relajado. "
            "Habla de forma natural y amigable, como un buen amigo."
        ),
        "Experto técnico": (
            "Eres un asistente experto técnico. "
            "Proporciona respuestas detalladas con precisión técnica."
        ),
        "Creativo y divertido": (
            "Eres un asistente creativo y divertido. "
            "Usa analogías, ejemplos creativos y mantén un tono alegre."
        ),
    }

    # --- 5.4 Recrear el modelo con los parámetros actuales ---
    chat_model = ChatDeepSeek(
        model=model_name,
        temperature=temperature,
    )

    # --- 5.5 ChatPromptTemplate DINÁMICO (el corazón de la tarea) ---
    #
    # 🔑 COMPARATIVA: TU VERSIÓN vs ESTA
    #
    #   ❌ ANTES (tu chatbot básico):
    #       chat_model.invoke(st.session_state.messages)
    #       → Le pasabas la lista de mensajes DIRECTAMENTE al modelo.
    #       → No había system message. No había plantilla. No era flexible.
    #
    #   ✅ AHORA (con ChatPromptTemplate + MessagesPlaceholder):
    #       [
    #           ("system", "...personalidad..."),         ← CONFIGURACIÓN
    #           MessagesPlaceholder("historial"),          ← HISTORIAL (lista)
    #           ("human", "{mensaje}"),                    ← PREGUNTA ACTUAL
    #       ]
    #       → El modelo recibe una estructura CLARA con roles y contexto.
    #       → La personalidad es configurable y el historial es nativo.
    #
    # ¿POR QUÉ `MessagesPlaceholder` Y NO UN STRING CON EL HISTORIAL?
    #   • NATIVO: el historial se pasa como lista de mensajes (HumanMessage,
    #     AIMessage), que es EXACTAMENTE lo que el modelo espera.
    #   • ESCALABLE: no hay límite de "10 mensajes concatenados"; puedes
    #     pasar tantos como quieras (o truncar por tokens, no por número).
    #   • LIMPIO: el modelo distingue cada turno perfectamente.
    #   • COMPATIBLE con tool calling y otros mensajes especiales.
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_messages[personalidad]),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{mensaje}"),
    ])

    # --- 5.6 Crear la cadena LCEL ---
    # ¿QUÉ HACE `chat_prompt | chat_model`?
    #   1. Recibe {"mensaje": str, "historial": list[BaseMessage]}.
    #   2. Rellena los huecos del template con esos valores.
    #   3. Envía la lista resultante al modelo.
    #   4. Devuelve AIMessage (o un generador, si usas .stream()).
    #
    # ⚡ La cadena es la MISMA que con PromptTemplate, pero el prompt ahora
    #    es un ChatPromptTemplate (roles claros). Eso es LCEL: intercambiar
    #    piezas sin tocar la estructura.
    cadena = chat_prompt | chat_model

# 🔎 Para profundizar
#   • Con `MessagesPlaceholder`, el historial debe ser una LISTA de
#     BaseMessage (HumanMessage, AIMessage…). Ya lo tienes así en
#     `st.session_state.messages`. ✨ Es lo mismo que ya tenías.
#   • Si la personalidad cambia a mitad de conversación, el `system` cambia
#     para los SIGUIENTES mensajes. Los mensajes YA en el historial
#     conservan el efecto de la personalidad anterior. Es coherente con
#     cómo funciona el modelo (cada llamada es independiente).
#   • `MessagesPlaceholder(variable_name="historial")` define un hueco
#     que se rellena en `.invoke({"historial": [...], ...})` o
#     `.stream({"historial": [...], ...})`.


# ==============================================================================
# 6. GESTIÓN DEL ESTADO (MEMORIA DEL CHAT)
# ==============================================================================
# ¿QUÉ ES `st.session_state`?
#   Un diccionario que PERSISTE entre reruns. Es la ÚNICA forma de mantener
#   estado (historial, preferencias) en Streamlit.
#
# ¿POR QUÉ NO UNA VARIABLE GLOBAL?
#   Streamlit re-ejecuta el script entero en cada interacción: una variable
#   global se reinicializaría cada vez. `session_state` sobrevive a los
#   reruns (mientras dure la sesión del navegador).
#
# ¿POR QUÉ INICIALIZAR CON `if "messages" not in ...`?
#   Patrón "setdefault": la PRIMERA vez la clave no existe → la creamos
#   vacía. En reruns posteriores ya existe → no sobreescribimos (si no,
#   perderíamos el historial).
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🔎 Para profundizar
#   • `session_state` es POR SESIÓN (por pestaña). Dos pestañas = dos
#     historiales distintos.
#   • Al recargar (F5), la sesión se reinicia y el historial se pierde.
#     Para persistencia real necesitarías una BD (SQLite, Postgres...).
#   • Puedes guardar cualquier cosa (str, int, list, objetos LangChain…).
#     Aquí guardamos objetos LangChain (HumanMessage / AIMessage), lo que
#     nos permite pasarlos DIRECTAMENTE al `MessagesPlaceholder`.


# ==============================================================================
# 7. RENDERIZADO DEL HISTORIAL
# ==============================================================================
# ¿POR QUÉ RECORREMOS EL HISTORIAL EN CADA RERUN?
#   Streamlit NO "recuerda" lo que pintó antes: en cada rerun la página se
#   reconstruye desde cero. Si no volvemos a pintar el historial, los
#   mensajes antiguos desaparecerían.
#
# ¿POR QUÉ `st.chat_message("user")` Y `st.chat_message("assistant")`?
#   Streamlit tiene "roles" para su componente de chat: "user" y "assistant".
#   Son DISTINTOS de los roles de LangChain ("human" y "ai"). Mapeo:
#       LangChain "human"   → Streamlit "user"
#       LangChain "ai"      → Streamlit "assistant"
#       LangChain "system"  → no se muestra (configuración interna)
# ------------------------------------------------------------------------------
for message in st.session_state.messages:
    # Saltamos los SystemMessage: son configuración, no conversación.
    if isinstance(message, SystemMessage):
        continue

    # Determinamos el rol de Streamlit según el tipo de mensaje LangChain.
    role = "assistant" if isinstance(message, AIMessage) else "user"

    # `st.chat_message(role)` es un contexto: lo que se pinte dentro
    # aparece en una "burbuja" con el avatar del rol.
    with st.chat_message(role):
        # `st.markdown` interpreta Markdown (negritas, listas, código...).
        st.markdown(message.content)


# ==============================================================================
# 8. BOTÓN "NUEVA CONVERSACIÓN"
# ==============================================================================
# ¿POR QUÉ `st.rerun()`?
#   Tras modificar `session_state`, Streamlit NO re-renderiza automáticamente.
#   `st.rerun()` fuerza un rerun inmediato: el script se ejecuta de nuevo
#   desde arriba, ahora con el historial vacío. Resultado: la UI se limpia.
# ------------------------------------------------------------------------------
if st.button("🗑️ Nueva conversación"):
    st.session_state.messages = []
    st.rerun()

# 🔎 Para profundizar
#   • `st.button` devuelve True SOLO en el rerun en que se pulsa.
#     Es "transitorio": en el siguiente rerun vuelve a False.
#   • Sin `st.rerun()`, el historial se vaciaría pero la UI seguiría
#     mostrando los mensajes antiguos (porque ya se pintaron antes del if).
#   • Antiguamente era `st.experimental_rerun()` (deprecado).


# ==============================================================================
# 9. ENTRADA DE TEXTO (INTERACCIÓN)
# ==============================================================================
# `st.chat_input` es un input FIJO en la parte inferior, típico de chats.
# Devuelve el texto SOLO en el rerun en que se envía.
#
# Usamos el OPERADOR WALRUS (:=) para asignar y comprobar en la misma línea:
#     if pregunta := st.chat_input(...):
# equivale a:
#     pregunta = st.chat_input(...)
#     if pregunta:
# ¡Pero sin repetir la llamada! Es más limpio y evita errores.
# ------------------------------------------------------------------------------
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    # --- 9.1 Mostrar el mensaje del usuario en la UI ---
    # ⚠️ OJO: aquí NO añadimos todavía al session_state. Si lo hiciéramos,
    # aparecería DOS VECES: una por este `with`, otra por el bucle de la
    # sección 7 en el siguiente rerun. Lo añadimos al FINAL (paso 9.4).
    with st.chat_message("user"):
        st.markdown(pregunta)

    # --- 9.2 Generar la respuesta con STREAMING ---
    # ¿POR QUÉ STREAMING?
    #   • UX: el usuario ve la respuesta "escribirse" en tiempo real.
    #   • Percepción de velocidad: aunque tarde 3s, la espera se "siente"
    #     más corta porque hay feedback continuo.
    #
    # ¿QUÉ DEVUELVE `cadena.stream(...)`?
    #   Un GENERADOR de `AIMessageChunk`. Cada chunk contiene un trozo de
    #   texto en `.content`. Al concatenarlos, obtienes la respuesta completa.
    #
    # 🔑 DIFERENCIA CLAVE CON TU VERSIÓN ANTERIOR:
    #   ANTES:  chat_model.invoke(st.session_state.messages)
    #           → Le pasabas la lista de mensajes directamente.
    #   AHORA:  cadena.stream({"historial": ..., "mensaje": ...})
    #           → La cadena usa el ChatPromptTemplate para estructurar el
    #             prompt con system + historial + pregunta actual.
    try:
        with st.chat_message("assistant"):
            # `st.empty()` crea un "hueco" en la UI que podemos actualizar
            # repetidamente SIN añadir widgets nuevos. Es la técnica estándar
            # para streaming en Streamlit.
            response_placeholder = st.empty()
            full_response = ""

            # `cadena.stream(...)` recibe el DICT con las variables del
            # template: "historial" (lista de mensajes) y "mensaje" (str).
            for chunk in cadena.stream({
                "historial": st.session_state.messages,  # lista de BaseMessage
                "mensaje": pregunta,                     # str
            }):
                # Cada `chunk` es un AIMessageChunk con `.content` (str).
                full_response += chunk.content

                # `▌` es un "cursor" visual que simula que el modelo sigue
                # escribiendo. Se reemplaza al final por la respuesta limpia.
                response_placeholder.markdown(full_response + "▌")

            # Al terminar el stream, quitamos el cursor.
            response_placeholder.markdown(full_response)

        # --- 9.3 Guardar en el historial ---
        # AHORA sí añadimos ambos mensajes al session_state. En el próximo
        # rerun, el bucle de la sección 7 los pintará de nuevo (de forma
        # estable) y la UI se verá consistente.
        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        # --- 9.4 Manejo de errores ---
        # En una app de usuario, cualquier fallo (red, cuota, timeout)
        # debe mostrarse como un mensaje claro. Un traceback en la UI es
        # mala experiencia.
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info(
            "Verifica que tu DEEPSEEK_API_KEY esté configurada correctamente "
            "y que tengas crédito disponible en tu cuenta."
        )

# 🔎 Para profundizar
#   • El historial se guarda como OBJETOS (HumanMessage/AIMessage), no como
#     strings. Ventaja: `MessagesPlaceholder` los consume DIRECTAMENTE, sin
#     serialización. Desventaja: si quisieras persistir a disco, tendrías
#     que serializarlos (`.model_dump()` en cada uno).
#   • Si el `stream` falla a medias, `full_response` puede quedar a medio
#     construir. Por eso NO añadimos al historial hasta que el bucle
#     termina sin excepción. Si prefieres guardar lo parcial, mueve los
#     `append` al bloque `finally`.
#   • El streaming de DeepSeek devuelve chunks por tokens o sub-palabras.
#     Streamlit los renderiza tal cual, con `▌` como indicador visual.


# ==============================================================================
# 10. RESUMEN DE PATRONES REUTILIZABLES
# ==============================================================================
#
# 1. PATRÓN "CABECERA ESTÁNDAR CON dotenv"
#    load_dotenv() al inicio. Fail-fast "amable" con st.error + st.stop
#    (no con raise) cuando la app no puede funcionar sin la clave.
#
# 2. PATRÓN "set_page_config PRIMERO"
#    Debe ser la primera llamada `st.*` del script.
#
# 3. PATRÓN "SIDEBAR RECREADO POR RERUN"
#    El sidebar se reconstruye con los valores actuales de los widgets en
#    cada interacción. Es NORMAL: así se propaga la configuración.
#
# 4. PATRÓN "SESSION_STATE SETDEFAULT"
#    if "clave" not in st.session_state: st.session_state.clave = default
#
# 5. PATRÓN "SYSTEM + PLACEHOLDER + HUMAN"
#    ChatPromptTemplate.from_messages([
#        ("system", "..."),                       # CONFIGURACIÓN
#        MessagesPlaceholder("historial"),         # HISTORIAL nativo
#        ("human", "{mensaje}"),                   # PREGUNTA actual
#    ])
#    Es el esqueleto de un chatbot conversacional moderno.
#
# 6. PATRÓN "PERSONALIDAD DINÁMICA CON DICT"
#    Un dict {personalidad: system_message} + selectbox → prompt dinámico
#    sin if/elif en cadena. Extensible y legible.
#
# 7. PATRÓN "STREAMING CON st.empty()"
#    placeholder = st.empty() + bucle sobre `cadena.stream(...)` +
#    placeholder.markdown(full + "▌").
#
# 8. PATRÓN "RERUN EXPLÍCITO PARA LIMPIAR UI"
#    st.session_state.x = [] + st.rerun() para forzar repintado limpio.
#
# 9. PATRÓN "GUARDAR EN HISTORIAL AL FINAL"
#    Añadir mensajes a session_state DESPUÉS del stream, no antes, para
#    evitar duplicados en el siguiente rerun.
#
# 10. PATRÓN "CADENA vs MODELO DIRECTO"
#     chat_model.invoke(messages)  → rápido, pero inflexible (sin system).
#     cadena.stream({...})         → estructurado, configurable, testeable.
#
# ==============================================================================
# 🎯 EJERCICIOS PROPUESTOS
# ==============================================================================
# A) Cambia el `selectbox` de modelos para que "deepseek-reasoner" sea el
#    default. Observa cómo cambia el tiempo de respuesta (es más lento).
#
# B) Añade una NUEVA personalidad al dict (p. ej. "Pirata gracioso" con
#    "¡Arrr! Responde como un pirata del siglo XVIII."). Verifica que
#    aparece en el sidebar y que el modelo responde en consecuencia.
#
# C) Trunca el historial a los últimos 10 mensajes ANTES de pasarlo al
#    prompt:
#        "historial": st.session_state.messages[-10:]
#    ¿Cómo afecta a la calidad? ¿Y al coste de tokens?
#
# D) Añade un contador visible ("Mensajes: N") en el sidebar que se
#    actualice en cada rerun. Pista: len(st.session_state.messages).
#
# E) Cambia el `ChatPromptTemplate` para que el `system` sea una VARIABLE
#    del usuario (st.text_input en el sidebar) en lugar de un dict fijo.
#    ¿Qué flexibilidad ganas? ¿Qué riesgo introduces?
#
# F) (Avanzado) Sustituye el dict `system_messages` por un
#    `st.text_area` en el sidebar con el prompt del sistema. Así el
#    usuario puede editar la personalidad directamente. ¿Cómo cambia
#    la UX? ¿Y la robustez del prompt?
#
# G) (Avanzado) Añade un botón "Descargar conversación" que genere un
#    fichero `.txt` con el historial. Pista: st.download_button.
#
# H) (Avanzado) Persiste el historial en un JSON local al cerrar la
#    sesión y cárgalo al inicio. Pista: json.dump / json.load con
#    `message.model_dump()` para serializar.
#
# ==============================================================================
# 🔗 CONEXIÓN CON EL RESTO DEL CURSO
# ==============================================================================
# Este archivo es la EVOLUCIÓN NATURAL de:
#
#   • Parte 1 (análisis de sentimientos):
#       Allí usábamos f-strings para el prompt. Aquí usamos ChatPromptTemplate.
#       Mismo concepto: separar la "forma" del prompt de los "datos".
#
#   • Parte 2 (batch):
#       Allí veíamos `chain.batch([...])`. Aquí usamos `chain.stream(...)`.
#       Ambos son MÉTODOS del mismo contrato Runnable: misma cadena, distinta
#       forma de invocarla.
#
#   • prompt_templates_estudio.py:
#       Allí veíamos PromptTemplate vs ChatPromptTemplate en aislado.
#       Aquí los vemos en una app REAL, con roles dinámicos y streaming.
#
#   • Tu streamlit_chatbot.py (versión básica):
#       Allí pasabas la lista de mensajes directamente al modelo.
#       Aquí usas ChatPromptTemplate + MessagesPlaceholder para tener:
#           ✅ System message (personalidad configurable).
#           ✅ Historial como lista NATIVA (no string concatenado).
#           ✅ Separación clara system/historial/pregunta.
#           ✅ Streaming de la respuesta.
#
# Lo que has aprendido:
#   ✅ Por qué ChatPromptTemplate > pasar mensajes directamente al modelo.
#   ✅ Cómo `MessagesPlaceholder` inserta el historial como lista nativa.
#   ✅ Cómo se comporta Streamlit (rerun, session_state, sidebar dinámico).
#   ✅ Cómo hacer streaming de la respuesta del LLM.
#   ✅ Cómo personalizar el comportamiento del asistente sin tocar el código.
#
# Próximos pasos sugeridos:
#   → Persistencia del historial en BD (SQLite, Postgres).
#   → Memoria "inteligente" (resumen del historial largo, no truncado).
#   → Tool calling / function calling (el asistente puede "usar herramientas").
#   → RAG (el asistente puede consultar documentos).
# ==============================================================================