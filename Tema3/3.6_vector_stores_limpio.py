from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# Carga todos los PDF de la carpeta: un Document por página de cada PDF.
loader = PyPDFDirectoryLoader("/home/mmbal/proyectos/curso-langchain/Tema3/contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio contratos.")

# Trocea en chunks de 1000 caracteres con un solape de 200.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")

# Crea la base vectorial Chroma: convierte cada chunk en un vector y lo guarda en disco.
# Ojo: cada ejecución vuelve a añadir los mismos chunks (se duplican).
vectorstore = Chroma.from_documents(
    docs_split,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)

# Consulta en lenguaje natural.
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos"

# Busca los 3 chunks más parecidos a la consulta y los muestra.
resultados = vectorstore.similarity_search(consulta, k=3)

print("Tres documentos más coincidentes con la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
