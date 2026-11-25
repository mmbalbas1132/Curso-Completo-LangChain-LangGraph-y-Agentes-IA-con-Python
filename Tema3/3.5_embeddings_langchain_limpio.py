from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()

# Modelo de embeddings de OpenAI: convierte un texto en un vector de números.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

texto1 = "La capital de Francia es París."
texto2 = "París es un nombre común para mascotas."

# embed_query devuelve el vector (lista de floats) de un texto.
vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)

# Dimensión del vector = longitud de la lista.
print(f"Dimensión de los vectores: {len(vec1)}")

# Similitud coseno = producto escalar / (norma1 × norma2). Va de -1 a 1; más cerca de 1 = más parecidos.
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Similitud coseno entre vec1 y vec2: {cos_sim:.3f}")