# ==============================================================================
# 3.5 — EMBEDDINGS CON LANGCHAIN: DEL TEXTO A UN VECTOR Y SIMILITUD COSENO
# ==============================================================================
# Este script convierte dos frases en vectores numéricos ("embeddings") con un modelo de
# OpenAI, muestra cuántos números tiene cada vector y calcula la similitud coseno entre los
# dos vectores para medir cuánto se parecen sus significados.
#
# Requisitos: pip install -U langchain-openai numpy python-dotenv
# Clave de API: OPENAI_API_KEY, leída del archivo .env con load_dotenv() (BLOQUE 1).
# Cada ejecución hace 2 llamadas a la API de OpenAI (una por frase).

# ==============================================================================
# BLOQUE 1 — IMPORTACIONES Y CARGA DEL .env
# ------------------------------------------------------------------------------
# OpenAIEmbeddings: clase de langchain_openai que representa un modelo de embeddings de OpenAI.
# numpy (np): biblioteca de cálculo numérico; aquí se usa para el producto escalar y la norma.
#   No es una dependencia de LangChain: hay que instalarla aparte (pip install numpy).
# load_dotenv: función de python-dotenv que lee el archivo .env y vuelca sus variables en el
#   entorno (aquí, OPENAI_API_KEY).
# ==============================================================================
from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

# load_dotenv() busca un archivo .env empezando por la carpeta del script y subiendo por las
# carpetas superiores (✅ comprobado). Debe ejecutarse antes de crear OpenAIEmbeddings.
# Ojo: por defecto NO sobrescribe una variable que ya exista en el entorno (✅).
# Esta llamada se añadió al original (el script original no la tenía); sin ella, OpenAIEmbeddings
# falla con "Missing credentials" si OPENAI_API_KEY no está ya definida (✅).
load_dotenv()

# ==============================================================================
# BLOQUE 2 — CREAR EL MODELO DE EMBEDDINGS
# ------------------------------------------------------------------------------
# Un embedding es una lista de números (un vector) que representa el significado de un texto:
# textos con ideas parecidas dan vectores cercanos (definición de la documentación de LangChain).
# OpenAIEmbeddings(model="text-embedding-3-large"): crea el modelo indicado. La documentación
# oficial usa este mismo nombre en su ejemplo y, para la familia text-embedding-3, permite elegir el
# tamaño del vector con el parámetro opcional dimensions (aquí no se usa).
# Ojo: si no se indica model, la clase usa por defecto otro modelo (text-embedding-ada-002 en la
# versión instalada, ✅), con otra dimensión; por eso aquí se indica explícitamente.
# ==============================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ==============================================================================
# BLOQUE 3 — LOS DOS TEXTOS
# ------------------------------------------------------------------------------
# Las dos frases comparten la palabra "París" pero con sentidos distintos (una ciudad y un
# nombre de mascota). Parece pensado para ver si el vector refleja el significado y no solo las
# palabras (es una lectura del ejemplo; el script no lo explica).
# ==============================================================================
texto1 = "La capital de Francia es París."
texto2 = "París es un nombre común para mascotas."

# ==============================================================================
# BLOQUE 4 — OBTENER LOS VECTORES
# ------------------------------------------------------------------------------
# embed_query(texto) envía el texto al modelo y devuelve su vector como lista de números
# (list[float], según la documentación de LangChain y ✅ comprobado con un modelo simulado).
# Cada llamada es una petición a la API (2 en total).
# La documentación distingue embed_query (un texto, pensado para consultas) y embed_documents
# (lista de textos, devuelve un vector por texto). Aquí se usa embed_query para ambos textos.
# ==============================================================================
vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)

# ==============================================================================
# BLOQUE 5 — DIMENSIÓN DE LOS VECTORES
# ------------------------------------------------------------------------------
# len(vec1) es la cantidad de números de la lista: la dimensión del embedding.
# Para text-embedding-3-large, OpenAI indica 3072 por defecto (⚠️ dato que no figura en las
# páginas de LangChain consultadas, y que no se ejecutó aquí contra la API real).
# Los dos vectores tienen la misma dimensión: es imprescindible para poder compararlos.
# ==============================================================================
print(f"Dimensión de los vectores: {len(vec1)}")

# ==============================================================================
# BLOQUE 6 — SIMILITUD COSENO
# ------------------------------------------------------------------------------
# Mide el ángulo entre dos vectores (definición de la documentación de LangChain):
#   cos_sim = (vec1 · vec2) / (‖vec1‖ × ‖vec2‖)
#   - np.dot(vec1, vec2): producto escalar (suma de multiplicar los números uno a uno).
#   - np.linalg.norm(v): norma (longitud) del vector.
# Resultado entre -1 y 1: 1 = misma dirección, 0 = perpendiculares, -1 = direcciones opuestas.
# Comprobado (✅) con vectores sencillos: idénticos → 1.0, perpendiculares → 0.0, opuestos → -1.0,
# y un vector multiplicado por 5 sigue dando 1.0 (importa la dirección, no la longitud).
# Ojo: si un vector fuera todo ceros habría una división por cero (no ocurre con embeddings reales).
# ==============================================================================
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ==============================================================================
# BLOQUE 7 — MOSTRAR EL RESULTADO
# ------------------------------------------------------------------------------
# {cos_sim:.3f} muestra el número con 3 decimales.
# Ojo: no se ejecutó con la API real, así que no se conoce el valor real ni la dimensión impresa.
# Con embeddings simulados (✅) el script imprime el formato esperado, pero los números no tienen
# ningún significado semántico.
# ==============================================================================
print(f"Similitud coseno entre vec1 y vec2: {cos_sim:.3f}")

# ==============================================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ------------------------------------------------------------------------------
# 1. Si se comparan más textos, usar embed_documents([...]) para obtener todos los vectores en
#    una sola llamada. No lo aplico.
# 2. Añadir un tercer texto (por ejemplo, sobre Francia) para tener un punto de comparación
#    del valor de similitud. No lo aplico.
# 3. Imprimir también los textos o usar una función auxiliar cos_sim(a, b) para reutilizarla.
#    No lo aplico.
# 4. Probar el parámetro dimensions (solo familia text-embedding-3) para reducir el tamaño del
#    vector. No lo aplico.
# 5. Envolver el código en "if __name__ == '__main__':" para poder importarlo sin ejecutarlo.
#    No lo aplico.