"""
Chatbot con ChatPromptTemplate, DeepSeek y Streamlit.

Ejecución:
    streamlit run chatbot_estudio.py
"""

import os
from dotenv import load_dotenv

import streamlit as st

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

load_dotenv()

st.set_page_config(
    page_title="Chatbot con DeepSeek",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")

if not os.getenv("DEEPSEEK_API_KEY"):
    st.error(
        "❌ Falta DEEPSEEK_API_KEY. Defínela en tu archivo `.env` "
        "o expórtala en el entorno antes de arrancar Streamlit."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuración")

    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7, 0.1)

    model_name = st.selectbox(
        "Modelo",
        ["deepseek-chat", "deepseek-reasoner"],
    )

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

    chat_model = ChatDeepSeek(
        model=model_name,
        temperature=temperature,
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_messages[personalidad]),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{mensaje}"),
    ])

    cadena = chat_prompt | chat_model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)

if st.button("🗑️ Nueva conversación"):
    st.session_state.messages = []
    st.rerun()

if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    with st.chat_message("user"):
        st.markdown(pregunta)

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in cadena.stream({
                "historial": st.session_state.messages,
                "mensaje": pregunta,
            }):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info(
            "Verifica que tu DEEPSEEK_API_KEY esté configurada correctamente "
            "y que tengas crédito disponible en tu cuenta."
        )