# ==============================================================================
# 3.4 — TEXT SPLITTERS, PARTE 2: TROCEAR EL TEXTO Y RESUMIR POR FRAGMENTOS
# ==============================================================================
# Continúa el script 3.3 (parte 1). Allí se enviaba el libro entero en un único prompt
# y la API lo rechazaba (error 400: 362.243 tokens frente a un máximo de 128.000).
# Aquí se intenta resolver con un "text splitter" y un resumen en dos fases:
#   1) resumir cada fragmento (chunk) por separado;
#   2) pedir al modelo que combine los resúmenes parciales en uno final.
#
# Ojo: tal como está escrito, el script NO resume el libro entero (ver BLOQUES 3 y 4).
#
# Requisitos: pip install -U langchain-community pypdf langchain-openai langchain-text-splitters python-dotenv
# Clave de API: OPENAI_API_KEY, leída del archivo .env con load_dotenv() (BLOQUE 1).
# Versión de Python: la última línea del script usa comillas dobles dentro de un f-string
# con comillas dobles, algo que solo admite Python 3.12 o superior (✅ en Python 3.10 da
# SyntaxError; en 3.13 compila).

# ==============================================================================
# BLOQUE 1 — IMPORTACIONES
# ------------------------------------------------------------------------------
# PyPDFLoader y ChatOpenAI: ver 3.3.
# load_dotenv: ver 3.3.
# RecursiveCharacterTextSplitter: clase del paquete langchain_text_splitters que divide
# un texto en fragmentos ("chunks"). La documentación oficial la recomienda como opción
# por defecto para texto genérico.
# ==============================================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# load_dotenv() busca un archivo .env empezando por la carpeta del script y subiendo por
# las carpetas superiores (✅ comprobado) y carga sus variables (aquí, OPENAI_API_KEY) en el
# entorno del proceso. Debe ejecutarse antes de crear ChatOpenAI.
# Ojo: por defecto NO sobrescribe una variable que ya exista en el entorno (✅): si el
# sistema o el IDE ya definen OPENAI_API_KEY (por ejemplo, una clave caducada), gana esa.
# Esta llamada la añadió el alumno al original (el script original no la tenía).
load_dotenv()

# ==============================================================================
# BLOQUE 2 — CARGAR EL PDF
# ------------------------------------------------------------------------------
# Igual que en 3.3: .load() devuelve 471 Document, uno por página (✅).
# Ojo: la ruta absoluta lleva el nombre de otro usuario y no existe en tu equipo
# (ver la sugerencia 1).
# ==============================================================================
# 1. Cargar el documento PDF
loader = PyPDFLoader("/home/mmbal/proyectos/curso-langchain/Tema3/quijote.pdf")
pages = loader.load()

# ==============================================================================
# BLOQUE 3 — DIVIDIR EN CHUNKS
# ------------------------------------------------------------------------------
# RecursiveCharacterTextSplitter(chunk_size, chunk_overlap):
#   - chunk_size: tamaño máximo de cada chunk, medido con la función de longitud del
#     splitter (por defecto len: caracteres; ✅ comprobado en el código instalado).
#   - chunk_overlap: solape objetivo entre chunks consecutivos, para no perder
#     información en los cortes.
#   - Por defecto prueba a cortar por ["\n\n", "\n", " ", ""], en ese orden (✅).
# split_documents(pages) recibe una lista de Document y devuelve otra lista de Document:
# trocea CADA Document por separado y copia sus metadatos (✅ 'page', 'title'…).
#
# Ojo — aquí el splitter no hace nada: cada página mide como máximo 4.939 caracteres,
# menos que chunk_size=10000, así que ninguna se parte. Resultado (✅): 471 páginas →
# 471 chunks, idénticos a las páginas. Para obtener chunks de ~10.000 caracteres sobre el
# libro entero habría que unir antes el texto (como en 3.3); entonces salen 136 chunks (✅).
# El comentario original dice "chunks mas pequeños": sin tilde en "más"; se respeta.
# ==============================================================================
# Dividir el texto en chunks mas pequeños
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

# ==============================================================================
# BLOQUE 4 — RESUMIR CADA CHUNK (FASE 1)
# ------------------------------------------------------------------------------
# llm: el mismo modelo que en 3.3. Necesita OPENAI_API_KEY (ver cabecera).
# summaries: lista donde se guardan los resúmenes parciales.
# i es un contador manual. El bucle recorre los chunks, pero con "if i > 10: break"
# se detiene al llegar a i = 11: solo se resumen los chunks con i = 0…10, es decir,
# los 11 PRIMEROS chunks = las 11 primeras páginas del PDF (✅: 22.707 caracteres de
# 1.323.105, un 1,7 %). El resto del libro no se procesa.
# Cada petición lleva una instrucción + el texto de un chunk (✅ entre ~180 y ~2.700
# caracteres con modelo simulado), muy por debajo del límite de contexto.
# response.content es el texto de la respuesta (AIMessage), no el objeto entero.
# Ojo: cada vuelta hace una llamada a la API (11 llamadas en total), con su coste.
# ==============================================================================
# 3. Pasar el texto al LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
summaries = []

i = 0
for chunk in chunks:
    if i > 10:
        break
    response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(response.content)
    i += 1

# ==============================================================================
# BLOQUE 5 — MOSTRAR LOS RESÚMENES PARCIALES
# ------------------------------------------------------------------------------
# Imprime la lista de strings con los 11 resúmenes.
# ==============================================================================
print(summaries)

# ==============================================================================
# BLOQUE 6 — RESUMEN FINAL (FASE 2)
# ------------------------------------------------------------------------------
# " ".join(summaries) une los resúmenes parciales separándolos con un espacio.
# Se envían al modelo con la instrucción de combinarlos y se imprime solo el texto
# (final_summary.content).
# Ojo: como solo hay 11 resúmenes, este "resumen completo" describe como mucho las 11
# primeras páginas, no el libro. Con la API real no se ejecutó aquí (✅ solo con un
# modelo simulado: 12 llamadas en total, 11 parciales + 1 final).
# Ojo: las comillas dobles dentro del f-string exigen Python 3.12 o superior.
# ==============================================================================
final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {" ".join(summaries)}")
print(final_summary.content)

# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ------------------------------------------------------------------------------
# 1. Evitar la ruta absoluta de otro usuario: ruta relativa o pathlib. No lo aplico.
# 2. Para trocear el libro de verdad: unir el texto de las páginas y usar split_text o
#    create_documents (136 chunks), o bajar chunk_size por debajo del tamaño de página.
#    No lo aplico.
# 3. Quitar el límite de 11 chunks (o parametrizarlo) si se quiere resumir todo el libro,
#    sabiendo que cada chunk es una llamada de pago. Alternativa más clara que el
#    contador manual: chunks[:11] o enumerate. No lo aplico. Si se resumieran los 471
#    chunks, el prompt final con 471 resúmenes podría volver a ser demasiado largo
#    (⚠️ no comprobado).
# 4. Numeración de pasos: los comentarios originales saltan de "# 1." a "# 3." (no hay
#    "# 2."). No lo aplico.
# 5. Evitar las comillas anidadas del f-string (calcular el " ".join antes) para que
#    funcione en Python < 3.12. No lo aplico.
# 6. En los prompts, "mas" lleva tilde ("más"). Es un literal; no lo aplico.