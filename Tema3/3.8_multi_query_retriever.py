# ==============================================================================
# 3.8 — MULTI-QUERY RETRIEVER: BUSCAR CON VARIAS REFORMULACIONES DE LA PREGUNTA
# ==============================================================================
# Problema: la búsqueda por similitud (3.6, 3.7) depende de cómo esté redactada la pregunta; dos
# formas de preguntar lo mismo pueden dar resultados distintos. Solución de este script: un
# modelo de chat (LLM) reescribe la pregunta de varias formas, se hace una búsqueda con cada
# una y se devuelve la unión de los resultados, sin documentos repetidos.
#
# Como 3.7, NO crea la base: abre la base Chroma que creó 3.6 (ejecuta 3.6 antes).
#
# Requisitos: pip install -U langchain-chroma langchain-classic langchain-openai chromadb
#             python-dotenv
# Clave de API: OPENAI_API_KEY, leída del archivo .env con load_dotenv() (BLOQUE 2). Sirve
# para el modelo de embeddings y para el modelo de chat.
# Coste: más que 3.7. Cada ejecución hace 1 llamada al modelo de chat (generar las preguntas)
# y 1 llamada de embeddings por cada pregunta generada (✅ 3 con la plantilla por defecto).
# Comprobado (✅): con modelos simulados y una base temporal. Con la API real: pendiente (BLOQUE 7).
#
# Cambios sobre el original (deliberados, pedidos para actualizar el script):
#   1. Chroma se importa de langchain_chroma (el original usaba langchain_community).
#   2. MultiQueryRetriever se importa de langchain_classic.retrievers.multi_query (el original
#      usaba langchain.retrievers.multi_query, que en langchain 1.x ya no existe).
#   El original completo se conserva en 3.8_multi_query_retriever_v1.py.

# ==============================================================================
# BLOQUE 1 — IMPORTACIONES
# ------------------------------------------------------------------------------
# Chroma: la base vectorial (ver 3.6). De langchain_chroma, el paquete que usa la documentación
#   oficial; la clase de langchain_community está obsoleta (ver 3.6).
# OpenAIEmbeddings: modelo de embeddings de OpenAI (ver 3.5).
# ChatOpenAI: modelo de chat de OpenAI; aquí hace de "reescritor" de preguntas.
# MultiQueryRetriever: el retriever de este script. Se importa de
#   langchain_classic.retrievers.multi_query. 🔀 El original usaba
#   "from langchain.retrievers.multi_query import …": con langchain 1.5.2 ese módulo ya no
#   existe (✅ ModuleNotFoundError: No module named 'langchain.retrievers'). El paquete
#   langchain-classic conserva estos retrievers; la página de referencia lo documenta en
#   langchain_classic.retrievers.multi_query.
# Ojo: ChatOpenAI y OpenAIEmbeddings se importan en dos líneas separadas del mismo paquete
#   (se respeta el código; ver la sugerencia 1).
# ==============================================================================
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# ==============================================================================
# BLOQUE 2 — CARGA DEL .env
# ------------------------------------------------------------------------------
# load_dotenv: lee el archivo .env y vuelca sus variables en el entorno (ver 3.3), para que
# OpenAIEmbeddings y ChatOpenAI encuentren OPENAI_API_KEY. Este bloque ya venía en el original.
# ==============================================================================
from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# ==============================================================================
# BLOQUE 3 — ABRIR LA BASE VECTORIAL EXISTENTE
# ------------------------------------------------------------------------------
# Igual que en 3.7: Chroma(embedding_function=..., persist_directory=...) abre la base de 3.6,
# con el mismo modelo de embeddings (text-embedding-3-large).
# Ojo (✅ visto en 3.7): si la ruta no existe, Chroma crea una carpeta vacía y las búsquedas
# devuelven listas vacías sin error. Ruta absoluta de Linux de tu equipo (sugerencia 2).
# ==============================================================================
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)

# ==============================================================================
# BLOQUE 4 — EL MODELO DE CHAT
# ------------------------------------------------------------------------------
# ChatOpenAI(model="gpt-4o-mini", temperature=0): modelo que generará las preguntas alternativas.
#   - model: nombre del modelo de OpenAI.
#   - temperature=0: controla la aleatoriedad de la respuesta; 0 la reduce al mínimo. ⚠️ Aun así
#     no garantiza que dos ejecuciones generen exactamente las mismas preguntas (explicación
#     general, no comprobada con la API).
# ==============================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# BLOQUE 5 — EL RETRIEVER BASE Y EL MULTI-QUERY
# ------------------------------------------------------------------------------
# base_retriever: el retriever de 3.7 (similarity, k=2): devuelve 2 documentos POR CADA consulta.
# MultiQueryRetriever.from_llm(retriever=..., llm=...) construye el retriever múltiple. Según el
# código instalado (✅):
#   - Monta una cadena: plantilla de prompt | modelo | analizador de líneas.
#   - La plantilla por defecto (DEFAULT_QUERY_PROMPT) pide al modelo 3 versiones distintas de la
#     pregunta, separadas por saltos de línea, "para superar algunas limitaciones de la búsqueda
#     por similitud basada en distancia". El analizador separa la respuesta por líneas y
#     descarta las vacías. El "3" está escrito en la plantilla, pero el número real de
#     preguntas lo decide el modelo (⚠️ no se comprueba).
#   - include_original=False por defecto: la pregunta ORIGINAL no se busca; solo se buscan
#     las preguntas generadas (✅ comprobado). Con include_original=True se añadiría.
#   - Se puede pasar otra plantilla con el parámetro prompt (página de referencia de from_llm).
# Resultado: con k=2 y 3 preguntas, hasta 6 documentos; los repetidos (documentos iguales:
# mismo contenido y metadatos) se eliminan ("unión única", ✅ según el código instalado).
# ==============================================================================
base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# ==============================================================================
# BLOQUE 6 — LA CONSULTA Y LA BÚSQUEDA
# ------------------------------------------------------------------------------
# retriever.invoke(consulta): (1) el modelo de chat genera las preguntas alternativas;
# (2) cada una se convierte en vector y se busca en la base (k=2 cada una); (3) se unen los
# resultados sin repetidos y se devuelve una lista de Document. Es lo mismo que 3.7 desde
# fuera: texto entra, lista de Document sale.
# Las preguntas generadas se registran con logger.info (el retriever tiene verbose=True por
# defecto ✅), pero solo se ven si configuras el logging a nivel INFO (✅); si no, no se
# muestran. Ver la sugerencia 3.
# ==============================================================================
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

# ==============================================================================
# BLOQUE 7 — MOSTRAR LOS RESULTADOS
# ------------------------------------------------------------------------------
# El título no dice "Top 2" (a diferencia de 3.7) porque el número de resultados varía: depende
# de cuántas preguntas genere el modelo y de cuántos documentos se repitan.
# enumerate(resultados, start=1) numera desde 1 (la variable i no se imprime).
# Salida real: pendiente de añadir (no se ha ejecutado con la API real todavía).
# ==============================================================================
print("Top documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")

# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ------------------------------------------------------------------------------
# 1. Unir las dos importaciones de langchain_openai en una:
#    "from langchain_openai import OpenAIEmbeddings, ChatOpenAI". No lo aplico.
# 2. Evitar rutas absolutas de un equipo concreto: ruta relativa o pathlib. No lo aplico.
# 3. Ver las preguntas generadas: activar el logging antes de invocar, por ejemplo con
#    logging.basicConfig() y logging.getLogger("langchain_classic.retrievers.multi_query")
#    .setLevel(logging.INFO) (✅ el nombre del logger es ese). No lo aplico.
# 4. Añadir include_original=True a from_llm para que la pregunta original también se busque
#    (parámetro ✅ de from_llm). No lo aplico.
# 5. Pasar una plantilla propia con prompt=... para generar las preguntas en español y adaptadas
#    a contratos (la plantilla por defecto está en inglés, ✅). No lo aplico.
# 6. Quitar la variable i (no se usa). No lo aplico.
# 7. Pasar el ejemplo a una función y envolver en "if __name__ == '__main__':". No lo aplico.
# 8. El orden de los imports del original deja "from dotenv import load_dotenv" a mitad del
#    archivo; es válido, pero se suele poner arriba. No lo aplico.
