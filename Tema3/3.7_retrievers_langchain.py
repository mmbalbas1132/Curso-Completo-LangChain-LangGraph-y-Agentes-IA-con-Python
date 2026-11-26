# ==============================================================================
# 3.7 — RETRIEVERS: CONSULTAR LA BASE VECTORIAL CON UNA INTERFAZ ESTÁNDAR
# ==============================================================================
# Este script NO crea la base de datos: abre la base Chroma que creó 3.6, la convierte en un
# "retriever" y le hace una pregunta. Un retriever es un objeto que recibe una pregunta (texto) y
# devuelve una lista de Document relacionados con ella. Es la pieza que usan los sistemas RAG
# (ver el proyecto Asistente_Legal_RAG) para buscar los fragmentos que luego se le pasan al modelo.
#
# Requisitos: pip install -U langchain-chroma langchain-openai chromadb python-dotenv
# Antes de ejecutarlo: ejecuta 3.6 (crea la carpeta chroma_db con los vectores).
# Clave de API: OPENAI_API_KEY, leída del archivo .env con load_dotenv() (BLOQUE 1).
# Coste: cada ejecución hace UNA llamada a la API de embeddings (la de la consulta); los
# vectores de los documentos ya están guardados.
# Comprobado (✅): con embeddings simulados y una base temporal, y con la API real en la ejecución
# del alumno (código de salida 0).
#
# Cambios sobre el original (deliberados, pedidos para actualizar el script):
#   1. Chroma se importa de langchain_chroma (el original usaba langchain_community).
#   2. Se añade load_dotenv() para leer la clave del .env.
#   3. La ruta de la base es la de 3.6 (el original apuntaba a una carpeta de Windows de otro
#      usuario). El original completo se conserva en 3.7_retrievers_langchain_v1.py.

# ==============================================================================
# BLOQUE 1 — IMPORTACIONES Y CARGA DEL .env
# ------------------------------------------------------------------------------
# Chroma: la base de datos vectorial (ver 3.6). Se importa de langchain_chroma (paquete
#   independiente, el que usa la documentación oficial). La clase de langchain_community está
#   obsoleta (✅ aviso "deprecated in LangChain 0.2.9" al activar todos los avisos de Python).
#   Al no importar nada de langchain_community, este script no muestra el aviso de retirada
#   de ese paquete (✅ comprobado).
# OpenAIEmbeddings: modelo de embeddings de OpenAI (ver 3.5).
# load_dotenv: lee el archivo .env y vuelca sus variables en el entorno (ver 3.3), para que
#   OpenAIEmbeddings encuentre OPENAI_API_KEY.
# ==============================================================================
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# ==============================================================================
# BLOQUE 2 — ABRIR LA BASE VECTORIAL EXISTENTE
# ------------------------------------------------------------------------------
# Chroma(embedding_function=..., persist_directory=...) ABRE la carpeta indicada; no añade
# documentos (eso lo hacía from_documents en 3.6). Detalles:
#   - Se usa la colección por defecto, que se llama "langchain" (✅), la misma que crea 3.6.
#   - El parámetro se llama embedding_function aquí (en from_documents de 3.6 se llamaba
#     embedding). La página oficial de Chroma usa embedding_function en este constructor.
#   - El modelo de embeddings debe ser el MISMO con el que se creó la base (aquí,
#     text-embedding-3-large, como en 3.6). Si no, Chroma falla por tamaño de vector distinto
#     (✅ "expecting embedding with dimension of 64, got 32" en la prueba con vectores simulados).
#   - Sin persist_directory la base sería solo en memoria y estaría vacía (la página oficial
#     de Chroma lo indica).
# Ojo — carpeta equivocada (✅ comprobado): si la ruta no existe, Chroma la CREA vacía y el
#   retriever devuelve una lista vacía, sin ningún error. Lo mismo ocurre con una ruta de
#   Windows ("C:\\Users\\...") en Linux: se crea una carpeta con ese nombre literal en el
#   directorio actual. Por eso la ruta debe ser la de la base de 3.6.
# Ojo: es una ruta absoluta de Linux de tu equipo (ver la sugerencia 1).
# ==============================================================================
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)

# ==============================================================================
# BLOQUE 3 — CREAR EL RETRIEVER
# ------------------------------------------------------------------------------
# vectorstore.as_retriever(...) convierte la base en un retriever (un VectorStoreRetriever ✅).
#   - search_type="similarity": busca los documentos más parecidos a la consulta. Los valores
#     admitidos, según el código instalado (✅), son "similarity", "similarity_score_threshold"
#     y "mmr"; "similarity" es el valor por defecto (✅). La página oficial de Chroma muestra
#     el uso con "mmr" y no lista los demás (⚠️).
#   - search_kwargs={"k": 2}: argumentos para la búsqueda; k es cuántos documentos devolver.
#     Sin k devuelve 4 (✅).
# Diferencia con 3.6: allí se llamaba directamente a vectorstore.similarity_search(consulta, k=3).
# Con "similarity" y el mismo k el resultado es el mismo (✅ comprobado con embeddings simulados);
# lo que cambia es la interfaz: un retriever es un componente estándar de LangChain (✅ es un
# Runnable, igual que los modelos y las cadenas), por lo que se puede usar con .invoke y
# conectar en cadenas, como hace el proyecto Asistente_Legal_RAG.
# ==============================================================================
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ==============================================================================
# BLOQUE 4 — LA CONSULTA Y LA BÚSQUEDA
# ------------------------------------------------------------------------------
# consulta: pregunta en lenguaje natural. Aquí sí cierra la interrogación (en 3.6 no).
# retriever.invoke(consulta): ejecuta la búsqueda con la consulta (texto) y devuelve una lista
#   de Document (✅ con k=2, 2 Document). Es la forma actual; el método antiguo
#   get_relevant_documents quedó obsoleto en LangChain (ver el cambio equivalente en 1.4).
#   Aquí se hace una llamada a la API de embeddings para convertir la consulta en un vector.
# ==============================================================================
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

# ==============================================================================
# BLOQUE 5 — MOSTRAR LOS RESULTADOS
# ------------------------------------------------------------------------------
# enumerate(resultados, start=1) numera desde 1 (la variable i no se imprime).
# doc.page_content es el texto del fragmento; doc.metadata, sus metadatos (source, page, etc.).
# El texto "Top 2" está escrito a mano: si cambias k, el mensaje no se actualiza (sugerencia 4).
# Salida real (✅ ejecución del alumno con la API de OpenAI, sin reproducir el contenido de los
# contratos): se imprimen 2 bloques. El 1.º es la primera página de un contrato de local de
# negocio y contiene la descripción del local; el 2.º es la última página (firmas) del mismo
# contrato. Son los mismos dos primeros resultados que dio 3.6 con k=3; el tercero de 3.6 no
# aparece porque aquí k=2.
# ==============================================================================
print("Top 2 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")

# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ------------------------------------------------------------------------------
# 1. Evitar rutas absolutas de un equipo concreto: ruta relativa o pathlib. No lo aplico.
# 2. Comprobar que la base no está vacía antes de buscar (por ejemplo, con
#    len(vectorstore.get()["ids"]), ✅ función pública), para detectar una ruta equivocada. No lo aplico.
# 3. Probar otros tipos de búsqueda: search_type="mmr" (con k y fetch_k en search_kwargs, como
#    en la página oficial) o "similarity_score_threshold" (exige score_threshold como float en
#    search_kwargs, ✅ lo valida el código instalado). No lo aplico.
# 4. Poner el número de resultados en una variable y usarla en search_kwargs y en el mensaje. No lo aplico.
# 5. Quitar la variable i (no se usa). No lo aplico.
# 6. Pasar el ejemplo a una función y envolver en "if __name__ == '__main__':". No lo aplico.
# 7. Incoherencia del original: el mensaje dice "mas" sin tilde. No lo aplico.
