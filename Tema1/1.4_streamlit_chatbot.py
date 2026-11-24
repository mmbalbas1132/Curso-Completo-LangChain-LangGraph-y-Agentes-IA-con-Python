# ─────────────────────────────────────────────────────────────
# 1) IMPORTACIONES
# ─────────────────────────────────────────────────────────────
import streamlit as st                      # Framework de UI web interactiva.
from dotenv import load_dotenv             # Carga variables del archivo .env.
from langchain_deepseek import ChatDeepSeek                    # Wrapper LangChain para DeepSeek.
from langchain_core.messages import (
    AIMessage,                             # Representa un turno del ASISTENTE (el modelo).
    HumanMessage,                          # Representa un turno del USUARIO.
    SystemMessage,                         # Representa una instrucción del SISTEMA (oculta al usuario).
)

# ─────────────────────────────────────────────────────────────
# 2) CONSTANTES DE CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
# Nombre del modelo servido por DeepSeek. 'deepseek-chat' es el modelo conversacional.
MODEL_NAME = "deepseek-chat"

# 'temperature' controla la creatividad (0 = determinista, 1 = creativo).
# 0.7 es un buen punto medio para un chatbot generalista.
TEMPERATURE = 0.7

# ─────────────────────────────────────────────────────────────
# 3) CONFIGURACIÓN INICIAL DE LA APP
# ─────────────────────────────────────────────────────────────
# Carga DEEPSEEK_API_KEY (y cualquier otra variable) desde el archivo .env
# ubicado en la raíz del proyecto. Sin esto, ChatDeepSeek fallaría al autenticarse.
load_dotenv()

# Metadatos de la pestaña del navegador y del layout de la página.
# layout="wide" hace que la app use todo el ancho en lugar de una columna central.
st.set_page_config(page_title="Chatbot con DeepSeek", page_icon="🤖", layout="wide")

st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")

# Instanciamos el modelo una sola vez.
# En Streamlit, esta línea se ejecuta en CADA re-ejecución del script,
# así que evitar crear el modelo dentro de un bucle es importante para el rendimiento.
chat_model = ChatDeepSeek(model=MODEL_NAME, temperature=TEMPERATURE)

# ─────────────────────────────────────────────────────────────
# 4) GESTIÓN DEL ESTADO (LA "MEMORIA" DEL CHAT)
# ─────────────────────────────────────────────────────────────
# Cada vez que el usuario interactúa, Streamlit re-ejecuta TODO este archivo.
# Las variables normales se reinician; solo 'st.session_state' persiste.
# Por eso inicializamos la lista de mensajes AQUÍ la primera vez.
if "messages" not in st.session_state:
    st.session_state.messages = []         # Lista que contendrá HumanMessage y AIMessage.

# ─────────────────────────────────────────────────────────────
# 5) RENDERIZADO DEL HISTORIAL
# ─────────────────────────────────────────────────────────────
# Recorremos los mensajes guardados y los pintamos como burbujas en la UI.
# Esto es lo que hace que la conversación "no desaparezca" al recargar.
for message in st.session_state.messages:
    # Los SystemMessage son instrucciones internas del sistema (no se muestran al usuario).
    if isinstance(message, SystemMessage):
        continue

    # Traducimos el tipo de mensaje a un rol que Streamlit entiende:
    # "assistant" para AIMessage, "user" para cualquier otro (HumanMessage).
    role = "assistant" if isinstance(message, AIMessage) else "user"

    # st.chat_message dibuja una burbuja con el avatar y estilo del rol indicado.
    with st.chat_message(role):
        st.markdown(message.content)       # message.content es el texto plano del turno.

# ─────────────────────────────────────────────────────────────
# 6) ENTRADA DE TEXTO E INTERACCIÓN
# ─────────────────────────────────────────────────────────────
# st.chat_input dibuja la caja de texto fija abajo.
# Devuelve None si el usuario aún no ha enviado nada, o un str si ha enviado.
# El operador walrus (:=) permite asignar y comprobar en la misma expresión.
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    # A) Pintamos la pregunta del usuario inmediatamente en la interfaz.
    with st.chat_message("user"):
        st.markdown(pregunta)

    # B) Añadimos la pregunta al historial como HumanMessage.
    # Así, en la siguiente llamada al modelo, esta pregunta formará parte del contexto.
    st.session_state.messages.append(HumanMessage(content=pregunta))

    # C) Generamos la respuesta.
    # Le pasamos TODA la lista de mensajes (no solo la última pregunta),
    # porque el LLM no tiene memoria por sí mismo: el contexto se lo damos nosotros.
    respuesta_ia = chat_model.invoke(st.session_state.messages)

    # D) Mostramos la respuesta en la interfaz como burbuja del asistente.
    with st.chat_message("assistant"):
        st.markdown(respuesta_ia.content)

    # E) Persistimos la respuesta en el historial.
    # 'respuesta_ia' YA es un AIMessage (lo devuelve ChatDeepSeek), así que
    # no hace falta envolverla: la guardamos tal cual.
    st.session_state.messages.append(respuesta_ia)