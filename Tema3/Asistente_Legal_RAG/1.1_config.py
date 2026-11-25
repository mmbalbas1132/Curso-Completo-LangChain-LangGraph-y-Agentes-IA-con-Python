# ============================================================
# BLOQUE 0 — MODELOS
# ------------------------------------------------------------
# Este archivo SOLO define constantes (no ejecuta nada). Lo importa
# rag_system.py con `from config import *`, que trae todos los nombres.
# El comentario original de la línea siguiente se conserva.
# ============================================================
# Configuración de modelos
# - EMBEDDING_MODEL: modelo de embeddings (convierte texto en vectores); se pasa a
#   OpenAIEmbeddings(model=...). Debe ser el MISMO con el que se creó la base Chroma.
#   ⚠️ (esta última condición es una regla general; no se puede comprobar con estos archivos).
# - QUERY_MODEL: modelo barato que reescribe la consulta (MultiQueryRetriever).
# - GENERATION_MODEL: modelo que redacta la respuesta final.
# ⚠️ Los nombres de modelo de OpenAI no se han contrastado con documentación oficial.
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# ============================================================
# BLOQUE 1 — BASE VECTORIAL
# ------------------------------------------------------------
# ============================================================
# Configuración del vector store
# Carpeta donde Chroma guarda/lee sus datos (persist_directory).
# Es una ruta absoluta de Windows de OTRO usuario/equipo: en tu máquina habrá que
# cambiarla. No se copia aquí para no repetir un nombre de usuario ajeno.
CHROMA_DB_PATH = "C:\\Users\\santiago\\curso_langchain\\Tema 3\\chroma_db"

# ============================================================
# BLOQUE 2 — RETRIEVER MMR
# ------------------------------------------------------------
# MMR (Maximal Marginal Relevance) equilibra parecido con la consulta y diversidad.
# ============================================================
# Configuración del retriever
# - SEARCH_TYPE: tipo de búsqueda que se pasa a as_retriever(search_type=...).
# - MMR_DIVERSITY_LAMBDA: se pasa como `lambda_mult` (0 a 1: 0 = máxima diversidad,
#   1 = mínima) ✅ (docstring de Chroma.max_marginal_relevance_search).
# - MMR_FETCH_K: candidatos que se recuperan antes de elegir los finales (`fetch_k`).
# - SEARCH_K: fragmentos finales (`k`); también recorta los que muestra la interfaz.
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

# ============================================================
# BLOQUE 3 — BÚSQUEDA HÍBRIDA
# ------------------------------------------------------------
# ============================================================
# Configuracion alternativa para retriever hibrido
# - ENABLE_HYBRID_SEARCH: si es True, rag_system.py combina el retriever MMR+MultiQuery
#   con uno de similitud (EnsembleRetriever); añade "+ Hybrid" al texto de la barra lateral.
# - SIMILARITY_THRESHOLD: 🔀 se pasa como `similarity_threshold=` a EnsembleRetriever, pero
#   ese parámetro NO existe (la referencia oficial no lo lista y la clase instalada ignora
#   los extras ✅): no tiene efecto sobre la búsqueda; solo se muestra en la interfaz.
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70


# ============================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ============================================================
# 1) Las constantes podrían leerse de un `.env` (sobre todo la ruta y el modelo). No lo aplico.
#
# 2) `CHROMA_DB_PATH` es absoluta y de otro equipo: mejor una ruta relativa al proyecto.
#    No lo aplico.
#
# 3) `SIMILARITY_THRESHOLD` no tiene efecto; conviene quitarlo o aplicarlo de verdad con
#    otro mecanismo. No lo aplico.
#
# 4) Los pesos del ensemble (0.7 y 0.3) están en rag_system.py: moverlos aquí.
#    No lo aplico.
#
# 5) `from config import *` oculta de dónde viene cada nombre; importar solo los
#    necesarios. No lo aplico.