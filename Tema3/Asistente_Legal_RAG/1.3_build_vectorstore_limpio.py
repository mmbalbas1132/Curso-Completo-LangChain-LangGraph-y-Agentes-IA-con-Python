import argparse
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, CHROMA_DB_PATH

# Carga OPENAI_API_KEY desde el archivo .env.
load_dotenv()

# Parámetros del troceado y carpeta de PDF por defecto (carpeta "contratos" junto al proyecto).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CARPETA_PDF_POR_DEFECTO = Path(__file__).resolve().parent.parent / "contratos"


def crear_ids(chunks):
    ids = []
    vistos = {}
    # Id estable por chunk: archivo + página + huella del texto (+ contador si se repite).
    for chunk in chunks:
        archivo = Path(chunk.metadata.get("source", "")).name
        pagina = chunk.metadata.get("page", "")
        huella = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        base = f"{archivo}|{pagina}|{huella}"
        n = vistos.get(base, 0)
        vistos[base] = n + 1
        ids.append(f"{base}|{n}")
    return ids


def construir(carpeta_pdf, reconstruir):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 1) Cargar los PDF de la carpeta (una Document por página).
    documentos = PyPDFDirectoryLoader(str(carpeta_pdf)).load()
    if not documentos:
        raise SystemExit(f"No se encontraron PDF en {carpeta_pdf}")
    print(f"Se cargaron {len(documentos)} documentos desde {carpeta_pdf}.")

    # 2) Trocear en chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Se crearon {len(chunks)} chunks de texto.")

    # Con --reconstruir se borra la colección anterior (útil si cambian los PDF o el troceado).
    if reconstruir:
        Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH).delete_collection()
        print("Colección anterior eliminada.")

    # 3) Calcular los embeddings y guardar en disco; con ids estables, re-ejecutar no duplica.
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        ids=crear_ids(chunks),
        persist_directory=CHROMA_DB_PATH
    )
    print(f"La base contiene {len(vectorstore.get()['ids'])} vectores en {CHROMA_DB_PATH}.")


if __name__ == "__main__":
    # Uso: python build_vectorstore.py [carpeta_pdf] [--reconstruir]
    parser = argparse.ArgumentParser(description="Construye (o actualiza) la base vectorial Chroma del asistente legal.")
    parser.add_argument("carpeta_pdf", nargs="?", default=str(CARPETA_PDF_POR_DEFECTO), help="Carpeta con los PDF")
    parser.add_argument("--reconstruir", action="store_true", help="Borra la colección existente antes de crearla")
    args = parser.parse_args()
    construir(Path(args.carpeta_pdf), args.reconstruir)
