# ─────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ─────────────────────────────────────────────────────────────
import streamlit as st
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ─────────────────────────────────────────────────────────────
# 2. CONFIGURACIÓN INICIAL
# ─────────────────────────────────────────────────────────────
# Carga las variables del archivo .env (DEEPSEEK_API_KEY)
load_dotenv()

# Configuración de la pestaña del navegador y el diseño de la página
st.set_page_config(page_title="Chatbot con DeepSeek", page_icon="🤖", layout="wide")

st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")

# Inicializamos el modelo de lenguaje
chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# ─────────────────────────────────────────────────────────────
# 3. GESTIÓN DEL ESTADO (MEMORIA DEL CHAT)
# ─────────────────────────────────────────────────────────────
# Streamlit reinicia el script cada vez que interactúas.
# Usamos 'session_state' para que los mensajes no se borren al recargar.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────
# 4. RENDERIZADO DEL HISTORIAL
# ─────────────────────────────────────────────────────────────
# Recorremos la lista de mensajes guardados y los mostramos en pantalla.
for message in st.session_state.messages:
    # Los mensajes de sistema (instrucciones ocultas para la IA) no se muestran
    if isinstance(message, SystemMessage):
        continue

    # Identificamos si el mensaje es del usuario o de la IA para darle el estilo correcto
    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)

# ─────────────────────────────────────────────────────────────
# 5. ENTRADA DE TEXTO (INTERACCIÓN)
# ─────────────────────────────────────────────────────────────
# st.chat_input muestra la caja de texto fija en la parte inferior.
# Usamos el operador 'walrus' (:=) para asignar y comprobar en la misma línea.
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):
    # A) Mostrar inmediatamente el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(pregunta)

    # B) Guardar el mensaje del usuario en el historial
    st.session_state.messages.append(HumanMessage(content=pregunta))

    # C) Generar la respuesta
    # OJO: Aquí le pasamos TODO el historial (st.session_state.messages)
    # para que el modelo tenga contexto de lo que se habló antes.
    respuesta_ia = chat_model.invoke(st.session_state.messages)

    # D) Mostrar la respuesta de la IA en la interfaz
    with st.chat_message("assistant"):
        st.markdown(respuesta_ia.content)

    # E) Guardar la respuesta de la IA en el historial (ya es un AIMessage)
    st.session_state.messages.append(respuesta_ia)