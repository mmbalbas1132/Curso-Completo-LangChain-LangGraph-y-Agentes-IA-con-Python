# ==============================================================================
# 1.3 — BUILD_VECTORSTORE: CONSTRUIR LA BASE VECTORIAL DEL ASISTENTE LEGAL
# ==============================================================================
# Este script crea (una sola vez, o cuando cambien los documentos) la base de datos vectorial
# Chroma que luego consulta rag_system.py: carga los PDF, los trocea, calcula los embeddings
# con el modelo de config.py y los guarda en disco en CHROMA_DB_PATH.
#
# Nota: es un script NUEVO (no es del curso original): reconstruye el procedimiento de
# 3.6_vector_stores.py aplicado al proyecto, con dos mejoras respecto a 3.6: ids estables (no
# duplica al re-ejecutar) y la opción --reconstruir.
#
# Uso:
#   python build_vectorstore.py                      -> usa la carpeta ../contratos
#   python build_vectorstore.py ruta/a/los/pdf       -> usa otra carpeta
#   python build_vectorstore.py --reconstruir        -> borra la colección y la crea de nuevo
# Requisitos: pip install -U langchain-community langchain-chroma langchain-openai
#             langchain-text-splitters pypdf python-dotenv
# Clave de API: OPENAI_API_KEY en el archivo .env (load_dotenv, BLOQUE 1).
# Coste: cada ejecución calcula los embeddings de TODOS los chunks (llamadas de pago a la
# API), aunque los ids eviten duplicados (✅ comprobado: 39 chunks → 39 textos embebidos
# en cada ejecución).

# ==============================================================================
# BLOQUE 1 — IMPORTACIONES, CONFIGURACIÓN Y CARGA DEL .env
# ------------------------------------------------------------------------------
# argparse: lee los argumentos de la línea de comandos (carpeta de PDF y --reconstruir).
# hashlib: calcula una huella (hash SHA-1) del texto de cada chunk para formar su id.
# Path (pathlib): manejo de rutas independiente del sistema operativo.
# load_dotenv: lee el archivo .env y vuelca sus variables en el entorno (ver 3.3).
# Chroma (langchain_chroma): la base vectorial (ver 3.6). PyPDFDirectoryLoader,
# OpenAIEmbeddings y RecursiveCharacterTextSplitter: ver 3.1, 3.5 y 3.4.
# from config import EMBEDDING_MODEL, CHROMA_DB_PATH: se toman de config.py (1.1) el modelo de
# embeddings y la carpeta de la base, los MISMOS que usa rag_system.py. Es importante: el
# modelo con el que se construye la base debe ser el mismo con el que se consulta.
# Aquí se importan solo los dos nombres necesarios (a diferencia de rag_system.py, que usa
# "from config import *").
# Ojo: CHROMA_DB_PATH es una ruta absoluta de config.py; ajústala a tu equipo.
# ==============================================================================
import argparse
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, CHROMA_DB_PATH


# load_dotenv() busca el .env empezando por la carpeta del script y subiendo por las
# superiores (✅ ver 3.3) y no sobrescribe variables ya definidas en el entorno.
load_dotenv()

# ==============================================================================
# BLOQUE 2 — PARÁMETROS
# ------------------------------------------------------------------------------
# CHUNK_SIZE=1000 y CHUNK_OVERLAP=200: tamaño máximo de cada chunk y solape, en caracteres
# (ver 3.4). Son los valores de 3.6; confírmalos si la base real se creó con otros.
# Con la carpeta de contratos de prueba (✅): 15 páginas → 39 chunks.
# CARPETA_PDF_POR_DEFECTO: carpeta "contratos" situada junto a la carpeta del proyecto.
#   Path(__file__).resolve().parent es la carpeta de este script; .parent.parent sube un nivel
#   más. Se puede cambiar pasando otra carpeta como argumento (ver la cabecera).
# ==============================================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CARPETA_PDF_POR_DEFECTO = Path(__file__).resolve().parent.parent / "contratos"


# ==============================================================================
# BLOQUE 3 — IDS ESTABLES PARA CADA CHUNK
# ------------------------------------------------------------------------------
# Problema (✅ visto en 3.6): sin ids, cada ejecución añade los mismos chunks otra vez y las
# búsquedas devuelven copias. Solución: darle a cada chunk un id que sea siempre el mismo.
# El id es "archivo|página|huella|n":
#   - archivo: solo el nombre del PDF (no la ruta completa), para que el id no cambie si mueves
#     la carpeta (✅ comprobado con otra ruta);
#   - página: número de página (metadata 'page');
#   - huella: los 12 primeros caracteres del SHA-1 del texto del chunk;
#   - n: contador por si dos chunks de la misma página tuvieran exactamente el mismo texto
#     (así nunca hay ids repetidos dentro de una misma llamada).
# Consecuencia: si el texto de un chunk cambia, su id cambia, y el chunk antiguo se queda en
# la base. En ese caso (PDF modificados o troceado distinto) usa --reconstruir.
# ==============================================================================
def crear_ids(chunks):
    ids = []
    vistos = {}
    for chunk in chunks:
        archivo = Path(chunk.metadata.get("source", "")).name
        pagina = chunk.metadata.get("page", "")
        huella = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        base = f"{archivo}|{pagina}|{huella}"
        n = vistos.get(base, 0)
        vistos[base] = n + 1
        ids.append(f"{base}|{n}")
    return ids


# ==============================================================================
# BLOQUE 4 — CONSTRUIR LA BASE
# ------------------------------------------------------------------------------
# 1) Se crea el modelo de embeddings con el modelo de config.py.
# 2) Se cargan los PDF (una Document por página). Si no hay ninguno, el script termina con un
#    mensaje claro (✅ comprobado) en vez de fallar más adelante.
# 3) Se trocea con RecursiveCharacterTextSplitter (ver 3.4).
# 4) Con --reconstruir (reconstruir=True) se borra la colección existente ANTES de crearla:
#    delete_collection() elimina la colección de Chroma. Se hace después de cargar y trocear, para
#    no perder la base antigua si la carga de los PDF falla.
# 5) Chroma.from_documents(chunks, embedding, ids, persist_directory) calcula los vectores y los
#    guarda en disco. Con ids ya existentes, Chroma actualiza en vez de duplicar (✅ 39 vectores
#    tras la 1.ª ejecución, tras repetirla, tras ejecutarla desde otra ruta y tras --reconstruir).
#    La colección se llama "langchain" (la de por defecto), que es la que abre rag_system.py.
# 6) vectorstore.get()['ids'] devuelve los ids guardados; su longitud es el nº de vectores.
# Comprobado (✅, embeddings simulados): después de construir, query_rag de rag_system.py abre
# la base y devuelve fragmentos con las claves que espera app.py.
# ==============================================================================
def construir(carpeta_pdf, reconstruir):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    documentos = PyPDFDirectoryLoader(str(carpeta_pdf)).load()
    if not documentos:
        raise SystemExit(f"No se encontraron PDF en {carpeta_pdf}")
    print(f"Se cargaron {len(documentos)} documentos desde {carpeta_pdf}.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Se crearon {len(chunks)} chunks de texto.")

    if reconstruir:
        Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH).delete_collection()
        print("Colección anterior eliminada.")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        ids=crear_ids(chunks),
        persist_directory=CHROMA_DB_PATH
    )
    print(f"La base contiene {len(vectorstore.get()['ids'])} vectores en {CHROMA_DB_PATH}.")


# ==============================================================================
# BLOQUE 5 — PUNTO DE ENTRADA
# ------------------------------------------------------------------------------
# if __name__ == "__main__": el código de dentro solo se ejecuta al lanzar el script
# directamente, no al importarlo.
# argparse define: un argumento opcional carpeta_pdf (nargs="?") con la carpeta por defecto, y
# el indicador --reconstruir (store_true: vale True si se escribe, False si no).
# Llama a construir(Path(carpeta), reconstruir).
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye (o actualiza) la base vectorial Chroma del asistente legal.")
    parser.add_argument("carpeta_pdf", nargs="?", default=str(CARPETA_PDF_POR_DEFECTO), help="Carpeta con los PDF")
    parser.add_argument("--reconstruir", action="store_true", help="Borra la colección existente antes de crearla")
    args = parser.parse_args()
    construir(Path(args.carpeta_pdf), args.reconstruir)


# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código se mantiene tal cual)
# ------------------------------------------------------------------------------
# 1. Pasar CHUNK_SIZE, CHUNK_OVERLAP y la carpeta a config.py para tener todos los parámetros
#    del proyecto en un solo sitio. No lo aplico.
# 2. Evitar recalcular los embeddings de chunks que ya existen (comprobar los ids con
#    vectorstore.get(ids=...) antes de embeber). No lo aplico (⚠️ no probado).
# 3. Borrar de la base los chunks cuyo id ya no corresponde a ningún PDF actual, en vez de
#    reconstruir todo. No lo aplico.
# 4. Mostrar un resumen por PDF (páginas y chunks). No lo aplico.
# 5. Aviso conocido: PyPDFDirectoryLoader sigue viniendo de langchain_community, por lo que
#    seguirá saliendo el aviso de retirada de ese paquete.
