# ============================================================
# BLOQUE 0 — IMPORTS Y COMPATIBILIDAD
# ------------------------------------------------------------
# Este módulo contiene toda la lógica RAG; app.py solo pinta la interfaz.
# - Chroma: base vectorial. Se importa de `langchain_chroma` (pip install -U langchain-chroma),
#   como en la documentación oficial. Cambio respecto al original, que la importaba de
#   `langchain_community`, donde la clase está deprecada ✅. Con `langchain_chroma` el
#   retriever MMR/similarity funciona igual y abre bases creadas con la clase antigua ✅.
# - OpenAIEmbeddings / ChatOpenAI: embeddings y modelo de chat de OpenAI.
# - PromptTemplate, MultiQueryRetriever, EnsembleRetriever: se importan de `langchain_core.prompts`
#   y `langchain_classic.retrievers` (pip install -U langchain-classic). Cambio respecto al
#   original, que los importaba de `langchain.prompts` / `langchain.retrievers`: 🔀 con
#   LangChain 1.x esas rutas dan ModuleNotFoundError ✅.
# - RunnablePassthrough / StrOutputParser: piezas de LCEL (entrada sin cambios / AIMessage a str).
# - streamlit: solo para `st.cache_resource`.
# ============================================================
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st

# `from config import *` y `from prompts import *` traen TODAS las constantes de esos
# archivos (ver 1.1 y 1.2).
from config import *
from prompts import *

# ============================================================
# BLOQUE 1 — INICIALIZACIÓN DEL SISTEMA (se ejecuta una vez)
# ------------------------------------------------------------
# `@st.cache_resource` hace que la función se ejecute una sola vez y devuelva el
# mismo objeto en las llamadas siguientes ✅ (comprobado: `a is b` → True). Así no se
# recrean la base ni los modelos en cada reejecución de Streamlit.
# ============================================================
@st.cache_resource
def initialize_rag_system():

    # Vector Store
    # Chroma(embedding_function=..., persist_directory=...) abre la base guardada en disco.
    # El `\` al final de la línea de `embedding_function` es innecesario (dentro de paréntesis); compila igual ✅.
    # La variable se llama `vectorestore` (falta una r): solo es un nombre.
    vectorestore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),\
        persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    # Dos modelos con temperature=0: uno barato para reescribir consultas y otro (gpt-4o)
    # para redactar la respuesta (nombres en config.py).
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)

    # 🔀 El nombre correcto es Maximal MARGINAL Relevance (no "Margin") ✅. MMR equilibra
    # parecido y diversidad: k = finales, fetch_k = candidatos, lambda_mult = diversidad
    # (0 máxima, 1 mínima).
    # Retriever MMR (Maximal Margin Relevance)
    base_retriever = vectorestore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    # Retriever adicional con similarity para comparar
    similarity_retriever = vectorestore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K}
    )

    # MULTI_QUERY_PROMPT convertido en plantilla (variable `question` ✅).
    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever.from_llm(retriever, llm, prompt): el LLM genera variantes de la
    # consulta y se recupera con cada una (firma oficial ✅).
    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    # EnsembleRetriever combina retrievers con weighted Reciprocal Rank Fusion (referencia
    # oficial ✅). Parámetros oficiales: retrievers, weights, c, id_key.
    # Ensemble Retriever que combinar MMR y similarity
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # mayor peso a MMR
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        # 🔀 `similarity_threshold` (arriba) NO existe en EnsembleRetriever: la clase lo
        # ignora sin error ✅. Los pesos 0.7/0.3 están escritos a mano aquí.
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

    # RAG_TEMPLATE como plantilla: variables context y question ✅.
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # format_docs: une los fragmentos en un texto con encabezado `[Fragmento N]`, y
    # `- Fuente: archivo` / `- Pagina: n` si hay metadatos. ✅ Ejemplo observado:
    #   '[Fragmento 1] - Fuente: c.pdf - Pagina: 0\ntexto'. La fuente se reduce al nombre de
    #   archivo solo si la ruta contiene `\` (rutas de Windows).
    # Funcion para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"
        
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted)

    # Cadena LCEL. El diccionario crea dos ramas en paralelo: "context" (pregunta ->
    # retriever -> format_docs) y "question" (RunnablePassthrough: la pregunta sin cambios).
    # Después: prompt -> gpt-4o -> StrOutputParser (devuelve str ✅).
    rag_chain = (
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    # ⚠️ Devuelve mmr_multi_retriever, no final_retriever: con búsqueda híbrida la respuesta
    # usa el ensemble, pero los fragmentos que se muestran salen de este otro retriever.
    return rag_chain, mmr_multi_retriever


# ============================================================
# BLOQUE 2 — CONSULTA: query_rag
# ------------------------------------------------------------
# Devuelve SIEMPRE una tupla (texto, lista_de_fragmentos); app.py la desempaqueta.
# Contrato con app.py ✅: cada fragmento es un dict con 'fragmento', 'contenido',
# 'fuente' y 'pagina'.
# ============================================================
def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener respuesta
        # Primera recuperación (dentro de la cadena) + generación de la respuesta.
        response = rag_chain.invoke(question)

        # Obtener documentos para mostrarlos
        # Segunda recuperación, solo para mostrar fragmentos: repite la llamada de MultiQuery
        # (otra llamada al modelo de consultas).
        # Cambio respecto al original: se usa `retriever.invoke(question)` ✅ (la forma actual).
        # El original llamaba a `get_relevant_documents`, marcado obsoleto desde langchain-core
        # 0.1.46 con eliminación en 1.0; en langchain-core 1.8.4 NO existe ✅ (AttributeError),
        # así que con 1.x se saltaba al `except` y la "respuesta" que veía el usuario era el
        # texto de error.
        docs = retriever.invoke(question)

        # Formatear los documentos para mostrar
        docs_info = []
        # Se muestran como máximo SEARCH_K fragmentos; el contenido se recorta a 1000 caracteres;
        # la fuente se reduce al nombre de archivo; la página es 'No especificada' si falta.
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)
        
        return response, docs_info
    
    # Se captura CUALQUIER excepción y se devuelve como texto de respuesta con fragmentos
    # vacíos: app.py nunca recibe una excepción (el error aparece como respuesta del asistente).
    except Exception as e:
        # (Errata "cosulta" en el mensaje original.)
        error_msg = f"Error al procesar la cosulta: {str(e)}"
        return error_msg, []
    
# ============================================================
# BLOQUE 3 — INFORMACIÓN DEL RETRIEVER
# ------------------------------------------------------------
# Devuelve un diccionario para la barra lateral de app.py, que solo usa 'tipo'.
# ✅ Con la configuración actual: {'tipo': 'MMR + MultiQuery + Hybrid', 'documentos': 2,
# 'diversidad': 0.7, 'candidatos': 20, 'umbral': 0.7}. El umbral se muestra aunque no
# tenga efecto real en la búsqueda.
# ============================================================
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }


# ============================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ============================================================
# 1) Devolver `final_retriever` para mostrar los fragmentos realmente usados y evitar la
#    segunda llamada de MultiQuery. No lo aplico.
#
# 2) Quitar `similarity_threshold` de EnsembleRetriever (se ignora) y mover los pesos a
#    config.py. No lo aplico.
#
# 3) En `query_rag`, no ocultar los errores como si fueran respuestas: devolver un
#    indicador de error o relanzar. No lo aplico.
#
# 4) Evitar `from config import *` / `from prompts import *`. No lo aplico.
#
# 5) Corregir erratas (`vectorestore`, "Margin", "cosulta", la `\` sobrante tras `embedding_function`).
#    No lo aplico.
#
# 6) Añadir tipos y docstrings a `query_rag`. No lo aplico.