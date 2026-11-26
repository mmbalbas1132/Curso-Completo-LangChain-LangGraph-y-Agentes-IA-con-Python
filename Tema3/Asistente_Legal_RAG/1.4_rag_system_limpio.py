# ⚠️ Escrito para LangChain 0.x: con langchain 1.x fallan los imports de `langchain.prompts` y
# `langchain.retrievers...` (ahora en langchain_core.prompts / langchain_classic.retrievers).
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st
from dotenv import load_dotenv

# Constantes (config.py) y plantillas (prompts.py), importadas con `*`.
# Registra config -> 1.1_config_limpio.py, prompts -> 1.2_..., rag_system -> 1.4_... (ver modulos_limpios.py).
import modulos_limpios
from config import *
from prompts import *

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()


# cache_resource: la función se ejecuta una vez y se reutiliza entre reejecuciones de Streamlit.
@st.cache_resource
def initialize_rag_system():
    # Vector Store
    vectorestore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL), \
        persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)

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

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    # Ensemble Retriever que combinar MMR y similarity
    # Híbrido: ensemble de MMR+MultiQuery y similitud (pesos 0.7/0.3).
    # `similarity_threshold` no es un parámetro de EnsembleRetriever: se ignora.
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3],  # mayor peso a MMR
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Funcion para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"

            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata[
                        'source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")

        return "\n\n".join(formatted)

    # Cadena RAG: {context: retriever -> format_docs, question: pregunta} -> prompt -> modelo -> texto.
    rag_chain = (
            {
                "context": final_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm_generation
            | StrOutputParser()
    )

    # Devuelve la cadena y el retriever MMR+MultiQuery (NO el ensemble).
    return rag_chain, mmr_multi_retriever


def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener respuesta
        response = rag_chain.invoke(question)

        # Obtener documentos para mostrarlos
        # Ojo: retriever sin ensemble. Se usa .invoke (`get_relevant_documents` ya no existe en langchain-core 1.x).
        docs = retriever.invoke(question)

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)

        return response, docs_info

    # Cualquier error se devuelve como texto de respuesta con lista de fragmentos vacía.
    except Exception as e:
        error_msg = f"Error al procesar la cosulta: {str(e)}"
        return error_msg, []


# Datos de configuración del retriever para la barra lateral (app.py solo usa 'tipo').
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }