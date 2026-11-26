from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# Abre la base Chroma ya creada en 3.6 (mismo modelo de embeddings y misma carpeta).
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)

# Convierte la base en un retriever que devuelve los 2 documentos más parecidos.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

print("Top 2 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
