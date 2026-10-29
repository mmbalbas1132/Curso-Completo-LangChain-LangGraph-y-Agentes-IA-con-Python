# ═══════════════════════════════════════════════════════════════════════════
# CHATBOT CON LANGCHAIN + DEEPSEEK + STREAMLIT
# Material de estudio: LCEL, PromptTemplate, streaming, configuración
# dinámica y manejo de errores.
#
# Ejecutar con:  streamlit run streamlit_chatbot.py
# Requiere:      archivo .env con DEEPSEEK_API_KEY=tu_clave
# ═══════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ───────────────────────────────────────────────────────────────────────────

# Streamlit: framework para crear apps web interactivas con Python puro.
# Cada vez que el usuario interactúa con un widget (botón, slider, input),
# Streamlit RE-EJECUTA todo el script de arriba a abajo.
# Esto es clave para entender por qué recreamos el modelo más abajo.
import streamlit as st

# dotenv: lee el archivo .env y carga sus variables en os.environ.
# Así no escribimos la API Key en el código (mala práctica de seguridad).
from dotenv import load_dotenv

# ChatDeepSeek: clase de LangChain que envuelve la API de DeepSeek.
# Actúa como un "ChatModel": recibe mensajes o texto y devuelve un AIMessage.
from langchain_deepseek import ChatDeepSeek

# Clases de mensajes de LangChain.
# 💡 CONCEPTO: LangChain estandariza los mensajes en clases:
#   - HumanMessage   → lo que escribe el usuario
#   - AIMessage      → lo que responde el modelo
#   - SystemMessage  → instrucciones "de sistema" (no visibles al usuario)
# Esta estandarización permite intercambiar modelos (OpenAI, DeepSeek, etc.)
# sin cambiar la lógica de la app.
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# PromptTemplate: "molde" de texto con variables {entre_llaves}.
# 💡 CONCEPTO: separa la ESTRUCTURA del prompt de los DATOS que se inyectan.
# Beneficios: reutilización, validación de variables, integración con LCEL.
from langchain_core.prompts import PromptTemplate


# ───────────────────────────────────────────────────────────────────────────
# 2. CONFIGURACIÓN INICIAL DE LA PÁGINA
# ───────────────────────────────────────────────────────────────────────────

# load_dotenv() busca un archivo .env en el directorio actual y carga
# sus variables. DEBE ejecutarse antes de crear ChatDeepSeek, porque el
# modelo lee DEEPSEEK_API_KEY del entorno al instanciarse.
load_dotenv()

# set_page_config DEBE ser la primera llamada a Streamlit (o una de las
# primeras). Configura el título de la pestaña, el icono y el layout.
# layout="wide" usa todo el ancho de la ventana (útil para chats).
st.set_page_config(page_title="Chatbot con DeepSeek", page_icon="🤖", layout="wide")

# Elementos visuales de la cabecera.
st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")


# ───────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR DE CONFIGURACIÓN DINÁMICA
# ───────────────────────────────────────────────────────────────────────────
# 💡 CONCEPTO: Configuración dinámica = el usuario ajusta parámetros en
# tiempo de ejecución sin tocar el código. En Streamlit, cada widget
# devuelve su valor actual y dispara un rerun al cambiar.

with st.sidebar:  # Todo lo que esté aquí dentro va a la barra lateral.
    st.header("⚙️ Configuración")

    # Selectbox: menú desplegable de opciones.
    # DeepSeek ofrece dos modelos con comportamientos distintos:
    #   - deepseek-chat:      generalista, rápido, acepta "temperature"
    #   - deepseek-reasoner:  razona paso a paso, NO acepta "temperature"
    model_name = st.selectbox(
        "Modelo",
        ["deepseek-chat", "deepseek-reasoner"],
        index=0,  # Opción por defecto (primera de la lista).
        help="deepseek-chat es rápido y versátil; deepseek-reasoner razona paso a paso."
    )

    # Slider: control deslizante entre min y max.
    #   temperature=0.0 → respuestas muy deterministas y repetitivas.
    #   temperature=1.0 → respuestas muy creativas y variadas.
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.7,   # Valor inicial.
        step=0.1,    # Incremento entre valores.
        help="Solo aplica a deepseek-chat. Mayor valor = más creatividad."
    )

    st.divider()  # Línea separadora visual.

    # ── Botón "Nueva conversación" ───────────────────────────────────────
    # 💡 CONCEPTO: st.session_state es un diccionario persistente entre
    # reruns de Streamlit. Vaciar 'messages' borra el historial lógico.
    # st.rerun() fuerza un nuevo ciclo de ejecución para reflejar el cambio
    # en la interfaz (sin él, la pantalla seguiría mostrando el historial
    # anterior hasta la siguiente interacción).
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []  # Limpia el historial.
        st.rerun()                      # Refresca la app inmediatamente.


# ───────────────────────────────────────────────────────────────────────────
# 4. INICIALIZACIÓN DINÁMICA DEL MODELO
# ───────────────────────────────────────────────────────────────────────────
# 💡 CONCEPTO: Recreamos el modelo en CADA ejecución del script.
# ¿Por qué? Porque los parámetros como "model" o "temperature" son
# INMUTABLES una vez creado el objeto ChatDeepSeek. Streamlit re-ejecuta
# el script completo al mover un slider, así que recrear el modelo es la
# forma natural de aplicar los nuevos valores.

if model_name == "deepseek-reasoner":
    # deepseek-reasoner lanza un error si le pasas "temperature",
    # así que lo instanciamos sin ese parámetro.
    chat_model = ChatDeepSeek(model=model_name)
else:
    # deepseek-chat sí acepta temperature.
    chat_model = ChatDeepSeek(model=model_name, temperature=temperature)


# ───────────────────────────────────────────────────────────────────────────
# 5. PROMPT TEMPLATE
# ───────────────────────────────────────────────────────────────────────────
# 💡 CONCEPTO: Un PromptTemplate tiene:
#   - input_variables: lista de "huecos" que el usuario debe rellenar.
#   - template:         el texto con los huecos entre {llaves}.
# Al invocarlo con un dict {var: valor}, LangChain sustituye los huecos
# y devuelve un PromptValue listo para enviarse al modelo.

prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres ChatBot Pro, un asistente útil, amigable y conciso.
Responde en español de forma clara y directa.

Historial de conversación:
{historial}

Usuario: {mensaje}
Asistente:"""
)

# Nota de estudio:
#   - La "personalidad" del bot va en las primeras líneas del template.
#   - El "{historial}" da contexto al modelo (memoria conversacional).
#   - El "{mensaje}" es la pregunta actual.
#   - "Asistente:" al final es un "cue" que empuja al modelo a responder.


# ───────────────────────────────────────────────────────────────────────────
# 6. CADENA LCEL
# ───────────────────────────────────────────────────────────────────────────
# 💡 CONCEPTO: LCEL (LangChain Expression Language) usa el operador "|"
# para encadenar componentes. El flujo es:
#
#     prompt_template  →  chat_model
#     (rellena huecos)     (genera respuesta)
#
# "cadena" es un Runnable: tiene .invoke(), .stream(), .batch(), etc.
# Ventajas frente a LLMChain (obsoleto):
#   - Más declarativo y legible.
#   - Composable: puedes añadir parsers, validadores, etc.
#   - Soporta streaming y async de forma nativa.
cadena = prompt_template | chat_model


# ───────────────────────────────────────────────────────────────────────────
# 7. GESTIÓN DEL ESTADO (MEMORIA DEL CHAT)
# ───────────────────────────────────────────────────────────────────────────
# 💡 CONCEPTO: st.session_state persiste entre reruns dentro de la misma
# sesión de navegador. Sin esto, cada interacción borraría el historial
# porque Streamlit re-ejecuta el script desde cero.
if "messages" not in st.session_state:
    st.session_state.messages = []


# ───────────────────────────────────────────────────────────────────────────
# 8. RENDERIZADO DEL HISTORIAL
# ───────────────────────────────────────────────────────────────────────────
# Recorremos los mensajes guardados y los pintamos con el estilo adecuado
# (burbuja de usuario o de asistente).
for message in st.session_state.messages:
    # Los SystemMessage son instrucciones internas: no se muestran.
    if isinstance(message, SystemMessage):
        continue

    # Detectamos el rol para que Streamlit use el estilo correcto.
    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)


# ───────────────────────────────────────────────────────────────────────────
# 9. ENTRADA DEL USUARIO + STREAMING + MANEJO DE ERRORES
# ───────────────────────────────────────────────────────────────────────────
# 💡 CONCEPTO: El operador walrus (:=) asigna y evalúa en una sola línea.
# Equivale a:
#     pregunta = st.chat_input(...)
#     if pregunta:
# pero más compacto.
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    # ── 9.1. Mostrar el mensaje del usuario inmediatamente ─────────────
    # Lo mostramos ANTES de llamar al modelo para que la interfaz se
    # sienta responsiva (feedback instantáneo).
    with st.chat_message("user"):
        st.markdown(pregunta)

    # ── 9.2. Convertir el historial a texto plano ──────────────────────
    # El PromptTemplate espera un STRING en {historial}, no una lista
    # de mensajes. Por eso formateamos cada mensaje como:
    #     "Usuario: ..."  /  "Asistente: ..."
    # Nota de estudio: para un enfoque más idiomático, LangChain ofrece
    # MessagesPlaceholder, que conserva los roles nativos. Aquí usamos
    # texto plano porque así lo pide el enunciado del ejercicio.
    historial_texto = "\n".join(
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
        for m in st.session_state.messages
        if not isinstance(m, SystemMessage)
    )

    # ── 9.3. Manejo de errores ─────────────────────────────────────────
    # 💡 CONCEPTO: try/except envuelve la llamada a la API, que puede
    # fallar por múltiples motivos:
    #   - API Key inválida o ausente (401).
    #   - Cuota agotada (429).
    #   - Fallo de red / timeout.
    #   - Contexto demasiado largo (límite de tokens).
    #   - Respuesta vacía o bloqueada por filtros.
    try:
        with st.chat_message("assistant"):
            # Placeholder vacío: lo iremos rellenando con cada chunk.
            # 💡 CONCEPTO: st.empty() crea un contenedor que podemos
            # sobrescribir múltiples veces sin duplicar contenido.
            response_placeholder = st.empty()
            full_response = ""

            # ── 9.4. Streaming ────────────────────────────────────────
            # 💡 CONCEPTO: .stream() devuelve un generador de chunks
            # (trozos de la respuesta). Cada chunk es un AIMessageChunk
            # con .content parcial. Acumulamos el texto y lo pintamos
            # con un cursor "▌" al final para simular escritura.
            #
            # Ventajas del streaming:
            #   - Menor latencia percibida (el usuario ve progreso).
            #   - Permite detectar errores antes de que termine.
            #   - Da sensación de interactividad tipo ChatGPT.
            for chunk in cadena.stream({
                "mensaje": pregunta,
                "historial": historial_texto
            }):
                # Algunos chunks pueden venir vacíos o sin .content
                # (por ejemplo, metadatos). Los filtramos con hasattr.
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    # Sobrescribimos el placeholder con el texto acumulado
                    # + el cursor parpadeante "▌".
                    response_placeholder.markdown(full_response + "▌")

            # Al terminar el stream, quitamos el cursor para dejar
            # la respuesta limpia.
            response_placeholder.markdown(full_response)

        # ── 9.5. Persistir la conversación ────────────────────────────
        # Guardamos AMBOS mensajes SOLO si el try tuvo éxito.
        # Esto evita historiales corruptos si la API falla a mitad.
        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        # Mostramos el error de forma amigable (sin traceback crudo).
        st.error(f"❌ Error al generar respuesta: {str(e)}")
        # Sugerimos causas probables al usuario.
        st.info(
            "Verifica que tu API Key de DeepSeek esté configurada correctamente "
            "en el archivo .env y que tengas conexión a internet."
        )