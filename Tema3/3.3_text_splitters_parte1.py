# ==============================================================================
# 3.3 — TEXT SPLITTERS, PARTE 1: EL PROBLEMA DE LOS DOCUMENTOS LARGOS
# ==============================================================================
# Este script carga un PDF muy largo (el Quijote, 471 páginas), junta todo su
# texto en un único string y se lo pasa de golpe a un modelo de lenguaje (LLM)
# para que lo resuma.
#
# Ojo: este enfoque "todo el documento en un único prompt" funciona con textos
# cortos, pero con un libro entero lo normal es que el modelo lo rechace por
# exceso de tamaño (ver BLOQUE 4). Para eso existen los "text splitters": trocean
# el texto en fragmentos manejables. Este archivo (parte 1) NO usa splitters.
#
# Requisitos: pip install -U langchain-community pypdf langchain-openai python-dotenv
#             y la variable de entorno OPENAI_API_KEY con una clave válida.
# La clave se lee del archivo .env mediante load_dotenv() (BLOQUE 1).
# ==============================================================================


# ==============================================================================
# BLOQUE 1 — IMPORTACIONES
# ------------------------------------------------------------------------------
# PyPDFLoader: "document loader" (cargador de documentos) que lee un PDF con la
#   biblioteca pypdf y lo convierte en objetos Document (texto + metadatos).
#   Se importa de langchain_community (paquete de integraciones de la comunidad).
# ChatOpenAI: clase de langchain_openai que representa un modelo de chat de OpenAI.
# load_dotenv: función de python-dotenv que lee el archivo .env y vuelca sus variables en
#   el entorno (ver más abajo).
# ==============================================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
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
# PyPDFLoader(ruta) solo prepara el cargador; no lee nada todavía.
# .load() lee el PDF y devuelve una LISTA de Document: uno por página.
# Con quijote.pdf se comprobó (✅) que salen 471 Document, ninguno vacío, con
# entre 29 y 4.939 caracteres cada uno. Cada Document trae en .metadata datos
# como 'page', 'page_label', 'total_pages', 'title', 'author' y 'source'.
#
# Ojo: la ruta es una ruta absoluta de Windows con el nombre de otro usuario. Si esa
# ruta no existe en tu equipo, PyPDFLoader lanza ValueError ("is not a valid file or url").
# Se comprobó (✅) al ejecutar el script tal cual en este equipo. Se respeta el
# código; ver la sugerencia 1 al final.
# La doble barra invertida "\\" es la forma de escribir una "\" dentro de un
# string de Python.
# ==============================================================================
# 1. Cargar el documento PDF
loader = PyPDFLoader("/home/mmbal/proyectos/curso-langchain/Tema3/quijote.pdf")
pages = loader.load()


# ==============================================================================
# BLOQUE 3 — UNIR TODAS LAS PÁGINAS EN UN SOLO TEXTO
# ------------------------------------------------------------------------------
# page.page_content es el texto de una página (un string).
# El bucle parte de un string vacío y le añade, página a página, su texto más un
# salto de línea ("\n") para separar páginas.
# Con quijote.pdf el resultado mide 1.323.576 caracteres (✅; unas 240.000
# palabras, contando por espacios en blanco).
# ==============================================================================
# 2. Combinar todas las páginas en un texto único
full_text = ""
for page in pages:
    full_text += page.page_content + "\n"


# ==============================================================================
# BLOQUE 4 — PASAR EL TEXTO AL LLM
# ------------------------------------------------------------------------------
# ChatOpenAI(model=..., temperature=...):
#   - model: nombre del modelo de OpenAI que se usará ("gpt-4o-mini").
#   - temperature: ajuste de la variabilidad de la respuesta; el script usa 0.2.
#     (La página oficial consultada de ChatOpenAI lo lista como parámetro
#     opcional, pero no detalla su rango.)
#   - Necesita la variable de entorno OPENAI_API_KEY. Sin ella, la creación del
#     objeto falla con un error de credenciales ausentes (✅ comprobado).
#
# llm.invoke(texto) envía el texto al modelo y devuelve un objeto AIMessage (✅).
# Aquí el texto es una instrucción ("Haz un resumen…") seguida de TODO el libro:
# el prompt completo tiene unos 1,32 millones de caracteres (✅ medido).
#
# Ojo — tamaño: cada modelo admite un máximo de texto de entrada (su "ventana de
# contexto", medida en tokens). Con la API real, este script FALLA en esta línea
# (✅ observado al ejecutarlo con una clave válida):
#   openai.BadRequestError: Error code: 400 - This model's maximum context length
#   is 128000 tokens. However, your messages resulted in 362243 tokens.
# Es decir: el libro entero son ~362.000 tokens y el modelo admite 128.000. LangChain
# relanza el error como OpenAIInvalidRequestError. Por tanto no se obtiene ningún
# resumen y la línea de BLOQUE 5 no llega a ejecutarse.
# Cálculo (✅): 1.323.646 caracteres ÷ 362.243 tokens ≈ 3,65 caracteres por token.
# ==============================================================================
# 3. Pasar el texto al LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente documento: {full_text}")


# ==============================================================================
# BLOQUE 5 — MOSTRAR EL RESULTADO
# ------------------------------------------------------------------------------
# print(response) imprime el objeto AIMessage ENTERO (✅ con un modelo simulado):
# algo como  content='...' additional_kwargs={} response_metadata={...} id='...'
# El texto del resumen está solo en response.content (la documentación oficial
# también muestra .text). Ver la sugerencia 3.
# ==============================================================================
print(response)


# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ------------------------------------------------------------------------------
# 1. Evitar la ruta absoluta de otro equipo: usar una ruta relativa o construirla
#    con pathlib a partir de la ubicación del script. No lo aplico.
# 2. Evitar el problema de tamaño: trocear el texto con un text splitter (por
#    ejemplo RecursiveCharacterTextSplitter, el recomendado por la documentación
#    para texto genérico) y resumir por fragmentos. Es el tema de la parte 2;
#    no lo aplico.
# 3. Imprimir solo el texto: print(response.content). No lo aplico.
# 4. En el prompt, "mas" lleva tilde ("más"). Es un literal; no lo aplico.
# 5. Envolver el código en "if __name__ == '__main__':" para poder importarlo
#    sin que se ejecute. No lo aplico.
# 6. Aviso general (observado al importar langchain_community 0.4.2 en 3.1): el
#    paquete avisa de que está en proceso de retirada; los imports siguen
#    funcionando. No lo aplico.
