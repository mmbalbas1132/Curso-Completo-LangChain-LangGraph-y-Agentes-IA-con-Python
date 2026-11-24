from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

loader = PyPDFLoader("/home/mmbal/proyectos/curso-langchain/Tema3/CV_Manuel_Balbas_Naveira_2026-11-10.pdf")

pages = loader.load()

for i, page in enumerate(pages):
    print(f"Página {i + 1}:")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")


loader2 = WebBaseLoader("https://es.wikipedia.org/wiki/Inteligencia_artificial")
pages2 = loader2.load()

for i, page in enumerate(pages2):
    print(f"Página {i + 1}:")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")