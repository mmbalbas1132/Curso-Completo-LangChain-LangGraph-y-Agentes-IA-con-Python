# ==============================================================================
# 3.6 — VECTOR STORES: GUARDAR EMBEDDINGS EN CHROMA Y BUSCAR POR SIGNIFICADO
# ==============================================================================
# Este script junta lo visto en 3.1-3.5: carga PDFs (3.1), los trocea en chunks (3.3-3.4),
# convierte cada chunk en un vector con embeddings (3.5) y los guarda en una base de datos
# vectorial (Chroma). Después hace una búsqueda por significado: dada una pregunta, devuelve
# los 3 chunks cuyo vector se parece más al de la pregunta.
#
# Requisitos: pip install -U langchain-community langchain-chroma langchain-openai langchain-text-splitters
#             pypdf chromadb python-dotenv
# Clave de API: OPENAI_API_KEY, leída del archivo .env con load_dotenv() (BLOQUE 1).
# Cada ejecución llama a la API de OpenAI para convertir en vectores todos los chunks y
# la consulta (coste), y añade datos a la base en disco (ver BLOQUE 4).

# ==============================================================================
# BLOQUE 1 — IMPORTACIONES Y CARGA DEL .env
# ------------------------------------------------------------------------------
# Chroma: base de datos vectorial; guarda textos junto a sus vectores y busca por similitud.
#   Se importa de langchain_chroma (paquete independiente, el que usa la documentación oficial).
#   Cambio respecto al original, que la importaba de langchain_community, donde la clase está
#   obsoleta (✅ aviso "deprecated in LangChain 0.2.9" al activar todos los avisos). Esta base es
#   compatible con la creada por la clase antigua (✅ se abrió una carpeta antigua con la nueva).
#   Ojo: seguirá saliendo el aviso de retirada de langchain-community, porque
#   PyPDFDirectoryLoader y el resto de cargadores aún se importan de ahí.
# OpenAIEmbeddings: modelo de embeddings de OpenAI (ver 3.5).
# PyPDFDirectoryLoader: cargador que lee todos los PDF de una carpeta.
# RecursiveCharacterTextSplitter: divide textos en chunks (ver 3.4).
# os: se importa pero el script no lo usa (✅ no aparece en ninguna otra línea).
# load_dotenv: lee el archivo .env y vuelca sus variables en el entorno (ver 3.3).
# ==============================================================================
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# ==============================================================================
# BLOQUE 2 — CARGAR LOS PDF DE LA CARPETA
# ------------------------------------------------------------------------------
# PyPDFDirectoryLoader(ruta).load() lee todos los PDF de la carpeta y devuelve una lista de
# Document, uno por página (✅: 5 PDF de 3 páginas cada uno dieron 15 Document, ninguno vacío).
# Cada Document trae en metadata: author, creationdate, creator, moddate, page, page_label,
# producer, source (ruta del PDF) y total_pages (✅).
# La página oficial consultada solo muestra el uso con la ruta; el resto de parámetros del
# cargador no figura ahí (⚠️).
# Ojo: la ruta es una ruta absoluta de Linux de tu equipo; en otro equipo no existirá
# (ver la sugerencia 1). No se copia aquí el contenido de los contratos.
# ==============================================================================
loader = PyPDFDirectoryLoader("/home/mmbal/proyectos/curso-langchain/Tema3/contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio contratos.")

# ==============================================================================
# BLOQUE 3 — TROCEAR EN CHUNKS
# ------------------------------------------------------------------------------
# chunk_size=1000: máximo de 1000 caracteres por chunk; chunk_overlap=200: solape de 200 (ver 3.4).
# split_documents trocea cada página por separado y conserva sus metadatos.
# Resultado (✅): 15 páginas (94 a 2.569 caracteres cada una) → 39 chunks (94 a 998 caracteres,
# ninguno de más de 1000).
# ==============================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")

# ==============================================================================
# BLOQUE 4 — CREAR LA BASE VECTORIAL (Y GUARDARLA EN DISCO)
# ------------------------------------------------------------------------------
# Chroma.from_documents(docs, embedding=..., persist_directory=...):
#   1) usa el modelo de embeddings para convertir cada chunk en un vector (llamadas a la API de
#      OpenAI, de pago);
#   2) guarda textos, metadatos y vectores en una colección de Chroma (por defecto se llama
#      'langchain', ✅ firma de from_documents);
#   3) persist_directory: carpeta donde se guarda en disco; al terminar el programa los datos
#      siguen ahí (✅ comprobado con una carpeta temporal y embeddings simulados).
# Sin persist_directory la base sería solo en memoria (así lo indica la página oficial de Chroma).
# Ojo — duplicados (✅): si ejecutas el script otra vez con la misma carpeta, los 39 chunks se
# añaden de nuevo (39 → 78 vectores), porque no se indican ids ni se borra la colección.
# Ojo — errores: con una clave no válida la API responde 401 (AuthenticationError) en esta
# llamada (✅ observado con una clave falsa); los dos print anteriores ya se habrán mostrado.
# El nombre del modelo de embeddings debe ser el mismo al crear la base y al consultarla; si no,
# las dimensiones y los vectores no serían comparables (explicación general, ⚠️).
# ==============================================================================
vectorstore = Chroma.from_documents(
    docs_split,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)

# ==============================================================================
# BLOQUE 5 — LA CONSULTA
# ------------------------------------------------------------------------------
# Es una pregunta en lenguaje natural sobre los contratos. La cadena abre con "¿" pero no
# cierra con "?" (se respeta el código).
# ==============================================================================
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos"

# ==============================================================================
# BLOQUE 6 — BUSCAR Y MOSTRAR
# ------------------------------------------------------------------------------
# similarity_search(consulta, k=3): convierte la consulta en un vector (otra llamada a la API) y
# devuelve una lista con los 3 chunks (Document) cuyos vectores más se parecen al de la consulta
# (✅ con embeddings simulados devuelve 3 Document).
# enumerate(resultados, start=1): numera los resultados desde 1 (la variable i no se imprime).
# doc.page_content es el texto del chunk; doc.metadata, sus metadatos (source, page, etc.).
# Resultado real (✅ salida de una ejecución con la API de OpenAI, sin reproducir el
# contenido de los contratos): se imprimen 3 bloques. El 1.º es la primera página de un
# contrato de local de negocio y contiene la descripción del local, es decir, responde a la
# pregunta; el 2.º es la última página (firmas) del mismo contrato, donde aparece el nombre
# pero no la ubicación del local; el 3.º es la primera página de OTRO contrato de local de
# negocio, sin relación con la persona de la consulta. Conclusión: la búsqueda por similitud
# devuelve lo más parecido, no solo lo relevante; con k=3 pueden colarse resultados flojos.
# (Con embeddings simulados el orden no tiene ningún significado.)
# ==============================================================================
resultados = vectorstore.similarity_search(consulta, k=3)

print("Tres documentos más coincidentes con la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")

# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ------------------------------------------------------------------------------
# 1. Evitar rutas absolutas de un equipo concreto: ruta relativa o pathlib. No lo aplico.
# 2. Evitar duplicados al ejecutar varias veces: borrar la carpeta de la base antes, pasar
#    ids estables a from_documents (acepta el parámetro ids, ✅ firma) o abrir la base
#    existente en vez de recrearla. No lo aplico.
# 3. Quitar "import os" porque no se usa. No lo aplico.
# 4. Cerrar la consulta con "?" y poner el nombre completo como dato de la consulta solo si
#    hace falta. No lo aplico.
# 5. Pasar el ejemplo a una función y envolver en "if __name__ == '__main__':". No lo aplico.
