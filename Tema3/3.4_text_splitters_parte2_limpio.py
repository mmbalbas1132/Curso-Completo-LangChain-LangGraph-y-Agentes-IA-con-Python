from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# 1. Cargar el documento PDF
# PyPDFLoader devuelve un Document por página (471 páginas).
loader = PyPDFLoader("C:\\Users\\santiago\\curso_langchain\\Tema 3\\quijote.pdf")
pages = loader.load()

# Dividir el texto en chunks más pequeños
# chunk_size y chunk_overlap se miden en caracteres. Cada página mide menos de 10000,
# así que el splitter no parte nada: salen 471 chunks (uno por página).
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

# 3. Pasar el texto al LLM
# Solo se resumen los 11 primeros chunks (i = 0…10), es decir, las 11 primeras páginas.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
summaries = []

i = 0
for chunk in chunks:
    if i > 10:
        break
    response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(response.content)
    i += 1

print(summaries)

# Resumen final: une los resúmenes parciales en una sola petición. (El f-string con comillas
# dobles anidadas exige Python 3.12 o superior.)
final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {" ".join(summaries)}")
print(final_summary.content)