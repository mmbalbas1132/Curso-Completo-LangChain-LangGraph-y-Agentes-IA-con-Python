# ============================================================
# BLOQUE 0 — IMPORTS
# ------------------------------------------------------------
# - streamlit as st: biblioteca que convierte un script de Python en una
#   web interactiva. Casi todo lo que se ve (título, chat, botones) son
#   llamadas del tipo `st.algo(...)`.
# - rag_system: módulo del propio proyecto (archivo rag_system.py, en la
#   misma carpeta). Aquí solo se importan dos funciones: este archivo NO
#   contiene la lógica RAG, solo la interfaz.
#     · query_rag(pregunta): por cómo se usa más abajo, devuelve DOS
#       valores: el texto de la respuesta y una lista de fragmentos (docs).
#     · get_retriever_info(): devuelve un diccionario del que solo se lee
#       la clave 'tipo'.
#   ✅ Contrato confirmado al leer rag_system.py (ver 1.3): query_rag devuelve
#   (respuesta, docs_info) y get_retriever_info() un dict con 5 claves.
# ============================================================

import streamlit as st
from rag_system import query_rag, get_retriever_info

# ============================================================
# BLOQUE 1 — CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------
# Se ejecuta en cada reejecución del script.
# ============================================================
# Configuración de la página
# - page_title: texto de la pestaña del navegador.
# - page_icon: icono de la pestaña.
# - layout="wide": usa todo el ancho de la ventana en lugar de una
#   columna central.
st.set_page_config(
    page_title="Sistema RAG - Asistente Legal",
    page_icon="⚖️",
    layout="wide"
)

# ============================================================
# BLOQUE 2 — TÍTULO
# ------------------------------------------------------------
# `st.title` pinta el título grande; `st.divider` dibuja una línea
# horizontal de separación.
# ============================================================
# Título
st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

# ============================================================
# BLOQUE 3 — ESTADO DE LA SESIÓN (la "memoria" del chat)
# ------------------------------------------------------------
# Streamlit reejecuta este archivo ENTERO, de arriba abajo, cada vez que
# el usuario interactúa (escribe, pulsa un botón...). Las variables
# normales se reinician en cada pasada; `st.session_state` es el almacén
# que se conserva durante la sesión del navegador.
# Por eso el historial vive ahí, y solo se crea vacío la primera vez
# (`"messages" not in st.session_state`).
# ============================================================
# Inicializar el historial de chat
# Estructura de cada mensaje (diccionario):
#   {"role": "user" | "assistant", "content": texto}
# y, solo en los del asistente, una clave extra "docs" con los
# fragmentos usados. ✅ Comprobado ejecutando la app con un
# rag_system simulado (Streamlit 1.65.0).
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# BLOQUE 4 — BARRA LATERAL
# ------------------------------------------------------------
# `with st.sidebar:` coloca todo lo que haya dentro en la barra lateral.
# Se ejecuta ANTES de pintar el historial (esto importa en el botón de
# limpiar, más abajo).
# ============================================================
# Sidebar simplificado
with st.sidebar:
    st.header("📋 Información del Sistema")
    
    # Información del retriever
    # Se llama en cada reejecución. Más abajo solo se muestra la clave
    # 'tipo' del diccionario (✅ con la config actual: "MMR + MultiQuery + Hybrid").
    retriever_info = get_retriever_info()
    
    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {retriever_info['tipo']}")
    
    st.markdown("**🤖 Modelos:**")
    # Texto escrito a mano en este archivo. ✅ Coincide con config.py
    # (QUERY_MODEL = gpt-4o-mini, GENERATION_MODEL = gpt-4o), pero si se
    # cambian allí, este texto no se actualiza solo.
    st.info("Consultas: GPT-4o-mini\nRespuestas: GPT-4o")
    
    st.divider()
    
    # `st.button` devuelve True solo en la pasada en la que se pulsa.
    # type="secondary" es el estilo normal; use_container_width=True
    # hace que el botón ocupe todo el ancho de la barra.
    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        # Observación ✅ (Streamlit 1.65.0): este botón vacía la lista ANTES
        # de que se pinte el historial (la barra lateral va antes), así
        # que probado con y sin `st.rerun()` la pantalla queda igualmente
        # vacía. El rerun aquí no es imprescindible, pero no hace daño.
        st.rerun()

# ============================================================
# BLOQUE 5 — DISTRIBUCIÓN PRINCIPAL EN COLUMNAS
# ------------------------------------------------------------
# `st.columns([2, 1])` crea dos columnas con anchos relativos 2:1:
# la primera (chat) es el doble de ancha que la segunda (documentos).
# ============================================================
# Layout principal con columnas
col1, col2 = st.columns([2, 1])

# ============================================================
# BLOQUE 6 — COLUMNA 1 — HISTORIAL DEL CHAT
# ------------------------------------------------------------
# Se recorre la lista guardada y se pinta cada mensaje con la burbuja
# de su rol (`st.chat_message(rol)` acepta "user" y "assistant").
# Es lo que hace que la conversación siga visible tras cada rerun.
# ============================================================
with col1:
    st.markdown("### 💬 Chat")
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ============================================================
# BLOQUE 7 — COLUMNA 2 — FRAGMENTOS DE LA ÚLTIMA RESPUESTA
# ------------------------------------------------------------
# Solo se muestran si el último mensaje es del asistente y tiene la
# clave "docs". Cada fragmento es un diccionario con las claves
# 'fragmento', 'fuente', 'pagina' y 'contenido' (✅ formato generado por
# query_rag en rag_system.py).
# `st.expander` crea un panel plegable; `expanded=False` lo deja cerrado.
# `st.text` muestra el texto tal cual, sin interpretar Markdown.
# ============================================================
with col2:
    st.markdown("### 📄 Documentos Relevantes")
    
    # Mostrar documentos de la última consulta
    if st.session_state.messages:
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

# ============================================================
# BLOQUE 8 — ENTRADA DEL USUARIO Y GENERACIÓN DE LA RESPUESTA
# ------------------------------------------------------------
# `st.chat_input` dibuja la caja de texto de chat y devuelve el texto
# enviado (o None si no se ha enviado nada). El operador walrus `:=`
# asigna y comprueba en la misma línea.
# Orden de lo que ocurre al enviar una pregunta:
#   1. Se guarda la pregunta en el historial.
#   2. Se llama a `query_rag(prompt)` dentro de un spinner.
#   3. Se guarda la respuesta con sus fragmentos.
#   4. `st.rerun()` reejecuta el script para pintar todo.
# ============================================================
# Input del usuario
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    # Añadir mensaje del usuario al historial
    # Desde este archivo solo se le pasa a `query_rag` la pregunta actual
    # (no el historial) y ✅ query_rag tampoco guarda memoria: cada pregunta
    # es independiente.
    # La pregunta se guarda ANTES de llamar al RAG.
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta
    # `st.spinner` muestra el mensaje mientras se ejecuta el bloque.
    with st.spinner("🔍 Analizando..."):
        response, docs = query_rag(prompt)
        # `app.py` no tiene try/except, pero ✅ `query_rag` (rag_system.py)
        # captura TODAS las excepciones y devuelve (texto_de_error, []): el error
        # se guarda y se muestra como si fuera la respuesta del asistente, sin
        # fragmentos. (Con un rag_system que sí lanzara, la ejecución se
        # interrumpiría y la pregunta quedaría sin respuesta: comprobado ✅.)
        st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})
    
    # Recargar para mostrar los nuevos mensajes
    # El historial y el panel de fragmentos ya se pintaron más arriba
    # con los datos antiguos; el rerun los vuelve a pintar con los nuevos.
    # ✅ Sin él, en esa pasada no se pintaría ninguna burbuja ni fragmento
    # (comprobado con Streamlit 1.65.0); aparecerían en la siguiente.
    st.rerun()

# ============================================================
# BLOQUE 9 — PIE DE PÁGINA
# ------------------------------------------------------------
# `unsafe_allow_html=True` permite que `st.markdown` interprete HTML.
# Aquí el HTML es un texto fijo escrito en este archivo (no procede de
# datos del usuario).
# ============================================================
# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con MMR Retriever</div>", 
    unsafe_allow_html=True
)


# ============================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ============================================================
# 1) Los errores llegan como texto de respuesta normal (query_rag los
#    captura): el historial guarda un "error" como si fuera una respuesta.
#    Se podría distinguir y mostrar un `st.error`. No lo aplico.
#
# 2) El texto de los modelos ("GPT-4o-mini / GPT-4o") está escrito a
#    mano. Mejor leerlo de config.py (coincide hoy). No lo aplico.
#
# 3) `get_retriever_info()` se llama en cada reejecución. Si fuera
#    costoso, se podría guardar en `st.session_state`. No lo aplico.
#
# 4) El botón "Limpiar Chat" funciona sin `st.rerun()` por el orden
#    de ejecución (ver observación en BLOQUE 4). No lo aplico:
#    cambiaría el código original.
#
# 5) El historial no se envía al RAG (✅ ni query_rag ni la cadena tienen
#    memoria): cada pregunta es independiente. No lo aplico.
#
# 6) Los nombres de claves ('fragmento', 'fuente', 'pagina',
#    'contenido') son un contrato implícito con rag_system.py. Una
#    constante o un tipo compartido evitaría desajustes. No lo aplico.
#
# 7) ✅ Con LangChain 1.x, rag_system.py falla (imports y get_relevant_documents,
#    ver 1.3) y esta interfaz mostraría solo el texto de error. No lo aplico.