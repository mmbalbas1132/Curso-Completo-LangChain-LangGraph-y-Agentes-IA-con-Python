# Configuración de modelos
# Modelos de OpenAI: embeddings, reescritura de consultas y respuesta final (se usan en rag_system.py).
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# Configuración del vector store
# Ruta ABSOLUTA de la base Chroma en tu WSL (la misma que crean 3.6 y 1.3); cámbiala si mueves el proyecto.
CHROMA_DB_PATH = "/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"

# Configuración del retriever
# MMR: SEARCH_K = fragmentos finales; MMR_FETCH_K = candidatos previos;
# MMR_DIVERSITY_LAMBDA: 0 = máxima diversidad, 1 = mínima.
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

# Configuracion alternativa para retriever hibrido
# Activa el retriever híbrido. SIMILARITY_THRESHOLD se pasa a EnsembleRetriever,
# que no tiene ese parámetro: se ignora.
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70