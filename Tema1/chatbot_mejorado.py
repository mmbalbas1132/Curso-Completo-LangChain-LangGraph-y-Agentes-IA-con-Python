# ═══════════════════════════════════════════════════════════════════════════
# CHATBOT CON LANGCHAIN + DEEPSEEK + STREAMLIT
# Lección: LCEL, PromptTemplate, streaming, configuración dinámica y
#          manejo de errores.
#
# Ejecutar:   streamlit run chatbot_mejorado.py
# Requiere:   archivo .env con DEEPSEEK_API_KEY=tu_clave
# ═══════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────
# 1) IMPORTACIONES
# ───────────────────────────────────────────────────────────────────────────

# Streamlit: framework para apps web interactivas en Python puro.
# IMPORTANTE: en cada interacción del usuario, Streamlit RE-EJECUTA este
# archivo completo de arriba abajo. Esa es la razón por la que recreamos
# el modelo más abajo y por la que usamos session_state para la memoria.
import streamlit as st

# dotenv: carga variables del archivo .env en os.environ.
# Lo usamos para no hardcodear la API key (mala práctica de seguridad).
from dotenv import load_dotenv

# ChatDeepSeek: wrapper LangChain de la API de DeepSeek.
# Implementa la interfaz BaseChatModel, por lo que expone .invoke() y
# .stream(), igual que ChatOpenAI, ChatAnthropic, etc. → portabilidad.
from langchain_deepseek import ChatDeepSeek

# Clases de mensajes de LangChain. Estructura tipada por rol:
#   - HumanMessage   → turno del usuario
#   - AIMessage      → turno del modelo
#   - SystemMessage  → instrucciones internas (no visibles al usuario)
# Gracias a esta tipificación puedes cambiar de proveedor sin tocar la
# lógica de la app: el wrapper traduce estos objetos al formato nativo.
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# PromptTemplate: plantilla con variables {entre_llaves}.
# Separa ESTRUCTURA (el template) de DATOS (los valores inyectados).
# Es un Runnable: se puede encadenar con "|" dentro de LCEL.
from langchain_core.prompts import PromptTemplate


# ───────────────────────────────────────────────────────────────────────────
# 2) CONFIGURACIÓN INICIAL DE LA PÁGINA
# ───────────────────────────────────────────────────────────────────────────

# Carga .env ANTES de crear ChatDeepSeek: el modelo lee DEEPSEEK_API_KEY
# del entorno al instanciarse. Si lo movemos después, falla la autenticación.
load_dotenv()

# set_page_config debe ser una de las PRIMERAS llamadas a Streamlit.
# layout="wide" usa todo el ancho de la ventana (mejor para chats).
st.set_page_config(page_title="Chatbot con DeepSeek", page_icon="🤖", layout="wide")

# Cabecera de la app.
st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")


# ───────────────────────────────────────────────────────────────────────────
# 3) SIDEBAR DE CONFIGURACIÓN DINÁMICA
# ───────────────────────────────────────────────────────────────────────────
# CONCEPTO: "configuración dinámica" = el usuario ajusta parámetros en
# caliente. Cada widget devuelve su valor actual y dispara un rerun al
# cambiar. Como el script se re-ejecuta, los nuevos valores se aplican
# automáticamente al re-instanciar el modelo (sección 4).

with st.sidebar:
    st.header("⚙️ Configuración")

    # Selectbox: menú desplegable. DeepSeek ofrece dos modelos con
    # comportamientos distintos (ver help).
    model_name = st.selectbox(
        "Modelo",
        ["deepseek-chat", "deepseek-reasoner"],
        index=0,   # Opción por defecto: la primera de la lista.
        help="deepseek-chat es rápido y versátil; deepseek-reasoner razona paso a paso."
    )

    # Slider: control deslizante con min/max/step.
    # temperature=0.0 → respuestas muy deterministas.
    # temperature=1.0 → respuestas más creativas y variadas.
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Solo aplica a deepseek-chat. Mayor valor = más creatividad."
    )

    st.divider()   # Línea visual separadora.

    # Botón "Nueva conversación".
    # st.rerun() es NECESARIO: aunque vaciemos 'messages', Streamlit está
    # pintando el historial antiguo en este mismo instante. Forzamos un
    # ciclo nuevo para que el bucle de render (sección 8) vea la lista
    # vacía y la UI refleje el borrado de inmediato.
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ───────────────────────────────────────────────────────────────────────────
# 4) INICIALIZACIÓN DINÁMICA DEL MODELO
# ───────────────────────────────────────────────────────────────────────────
# CONCEPTO: los parámetros del modelo son INMUTABLES una vez creado el
# objeto. Como Streamlit re-ejecuta el script al mover un slider,
# recrear el modelo aquí es la forma natural de aplicar los nuevos valores.
#
# deepseek-reasoner NO acepta 'temperature' (lanza error), por eso el if/else.

if model_name == "deepseek-reasoner":
    chat_model = ChatDeepSeek(model=model_name)
else:
    chat_model = ChatDeepSeek(model=model_name, temperature=temperature)


# ───────────────────────────────────────────────────────────────────────────
# 5) PROMPT TEMPLATE
# ───────────────────────────────────────────────────────────────────────────
# CONCEPTO: PromptTemplate = "molde" con huecos {variable}.
# Al invocarlo con un dict {var: valor} devuelve un PromptValue listo
# para enviarse al modelo.
#
# Piezas del template:
#   - Personalidad del bot: primeras líneas (fijas).
#   - {historial}: contexto conversacional (string plano).
#   - {mensaje}:   pregunta actual del usuario.
#   - "Asistente:": cue que empuja al modelo a completar la respuesta.

prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres ChatBot Pro, un asistente útil, amigable y conciso.
Responde en español de forma clara y directa.

Historial de conversación:
{historial}

Usuario: {mensaje}
Asistente:"""
)


# ───────────────────────────────────────────────────────────────────────────
# 6) CADENA LCEL
# ───────────────────────────────────────────────────────────────────────────
# CONCEPTO: LCEL = composición con "|". El flujo:
#
#     prompt_template → chat_model
#     (rellena huecos)   (genera respuesta)
#
# 'cadena' es un Runnable: expone .invoke(), .stream(), .batch(), .ainvoke().
# Ventajas frente a LLMChain (obsoleta):
#   - Declarativo y legible.
#   - Composable (puedes añadir parsers, validadores, fallbacks, etc.).
#   - Streaming y async de primera clase.

cadena = prompt_template | chat_model


# ───────────────────────────────────────────────────────────────────────────
# 7) GESTIÓN DEL ESTADO (MEMORIA)
# ───────────────────────────────────────────────────────────────────────────
# session_state persiste entre reruns. Sin esto, el historial se borraría
# en cada interacción (variables normales se reinician con el script).
if "messages" not in st.session_state:
    st.session_state.messages = []


# ───────────────────────────────────────────────────────────────────────────
# 8) RENDERIZADO DEL HISTORIAL
# ───────────────────────────────────────────────────────────────────────────
# Recorremos los mensajes guardados y los pintamos como burbujas.
# Es lo que hace que la conversación "no desaparezca" al recargar.
for message in st.session_state.messages:
    # Los SystemMessage son instrucciones internas: no se muestran.
    if isinstance(message, SystemMessage):
        continue

    # Traducimos el tipo al rol que entiende Streamlit.
    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)


# ───────────────────────────────────────────────────────────────────────────
# 9) ENTRADA DEL USUARIO + STREAMING + MANEJO DE ERRORES
# ───────────────────────────────────────────────────────────────────────────
# Walrus (:=): asigna y comprueba en una sola expresión.
# Equivale a:  pregunta = st.chat_input(...); if pregunta: ...
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    # ── 9.1 Pintar la pregunta inmediatamente ──────────────────────────
    # Feedback instantáneo → la UI se siente responsiva aunque el modelo
    # tarde segundos en responder.
    with st.chat_message("user"):
        st.markdown(pregunta)

    # ── 9.2 Serializar el historial a texto plano ──────────────────────
    # El PromptTemplate espera un STRING en {historial}, no una lista.
    # Formato: "Usuario: ..." / "Asistente: ...".
    # Filtramos SystemMessage para no exponer instrucciones internas.
    #
    # Nota de estudio: en producción se usaría MessagesPlaceholder para
    # preservar los roles nativos. Aquí se serializa a texto porque el
    # template así lo requiere.
    historial_texto = "\n".join(
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
        for m in st.session_state.messages
        if not isinstance(m, SystemMessage)
    )

    # ── 9.3 Manejo de errores ──────────────────────────────────────────
    # try/except envuelve la llamada a la API: puede fallar por clave
    # inválida (401), cuota (429), timeout, contexto demasiado largo,
    # respuesta vacía/bloqueada, etc.
    try:
        with st.chat_message("assistant"):
            # st.empty(): contenedor "hueco" que se sobrescribe sin duplicar.
            # Es la pieza clave para el streaming en la UI.
            response_placeholder = st.empty()
            full_response = ""

            # ── 9.4 Streaming ──────────────────────────────────────────
            # .stream() devuelve un GENERADOR de AIMessageChunk.
            # Acumulamos el texto y repintamos con un cursor "▌" para
            # simular escritura en vivo.
            #
            # Ventajas: menor latencia percibida, detección temprana de
            # errores, sensación de interactividad tipo ChatGPT.
            for chunk in cadena.stream({
                "mensaje": pregunta,
                "historial": historial_texto
            }):
                # Algunos chunks no traen .content (metadatos, eventos).
                # hasattr + truthiness → filtrado defensivo.
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")

            # Fin del stream: quitamos el cursor y dejamos texto limpio.
            response_placeholder.markdown(full_response)

        # ── 9.5 Persistir la conversación ──────────────────────────────
        # Guardamos AMBOS mensajes SOLO si el try tuvo éxito.
        # Así el historial no queda corrupto si la API falla a mitad.
        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        # Mensaje amigable: nada de tracebacks crudos al usuario final.
        st.error(f"❌ Error al generar respuesta: {str(e)}")
        st.info(
            "Verifica que tu API Key de DeepSeek esté configurada correctamente "
            "en el archivo .env y que tengas conexión a internet."
        )