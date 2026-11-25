# Chat web (Streamlit) sobre un sistema RAG; la lógica RAG está en rag_system.py.
import streamlit as st
# query_rag(pregunta) -> (respuesta, docs_info); get_retriever_info() -> dict (aquí solo se usa 'tipo').
from rag_system import query_rag, get_retriever_info

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG - Asistente Legal",
    page_icon="⚖️",
    layout="wide"
)

# Título
st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

# Inicializar el historial de chat
# Streamlit reejecuta el script entero en cada interacción; session_state conserva el historial.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar simplificado
with st.sidebar:
    st.header("📋 Información del Sistema")
    
    # Información del retriever
    retriever_info = get_retriever_info()
    
    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {retriever_info['tipo']}")
    
    st.markdown("**🤖 Modelos:**")
    st.info("Consultas: GPT-4o-mini\nRespuestas: GPT-4o")
    
    st.divider()
    
    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Layout principal con columnas
col1, col2 = st.columns([2, 1])

# Dos columnas (proporción 2:1): chat a la izquierda, fragmentos de la última respuesta a la derecha.
with col1:
    st.markdown("### 💬 Chat")
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.markdown("### 📄 Documentos Relevantes")
    
    # Mostrar documentos de la última consulta
    if st.session_state.messages:
        # Mensajes: dict con 'role' y 'content'; los del asistente llevan además 'docs'.
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant" and "docs" in last_message:
            docs = last_message["docs"]
            
            if docs:
                for doc in docs:
                    with st.expander(f"📄 Fragmento {doc['fragmento']}", expanded=False):
                        st.markdown(f"**Fuente:** {doc['fuente']}")
                        st.markdown(f"**Página:** {doc['pagina']}")
                        st.markdown("**Contenido:**")
                        st.text(doc['contenido'])

# Input del usuario
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    # Añadir mensaje del usuario al historial
    # Se guarda la pregunta antes de llamar al RAG (query_rag captura los errores y los devuelve como texto).
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta
    with st.spinner("🔍 Analizando..."):
        response, docs = query_rag(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})
    
    # Recargar para mostrar los nuevos mensajes
    # El historial ya se pintó más arriba: el rerun redibuja la pantalla con los mensajes nuevos.
    st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con MMR Retriever</div>", 
    unsafe_allow_html=True
)