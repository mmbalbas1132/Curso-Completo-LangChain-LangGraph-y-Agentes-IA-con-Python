from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# 1. Cargar el documento PDF
# PyPDFLoader devuelve un Document por página (el PDF tiene 471 páginas).
loader = PyPDFLoader("C:\\Users\\santiago\\curso_langchain\\Tema 3\\quijote.pdf")
pages = loader.load()

# 2. Combinar todas las páginas en un texto único
# Resultado: un solo string de unos 1,3 millones de caracteres.
full_text = ""
for page in pages:
    full_text += page.page_content + "\n"

# 3. Pasar el texto al LLM
# Se envía el libro entero en un único prompt: excede el contexto del modelo (error 400 de la API).
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente documento: {full_text}")

# Imprime el objeto AIMessage completo; el texto está en response.content.
print(response)