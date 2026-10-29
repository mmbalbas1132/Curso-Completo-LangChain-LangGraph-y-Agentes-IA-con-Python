# ─────────────────────────────────────────────────────────────
# 1. IMPORTACIONES (Siempre al principio del archivo)
# ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# NOTA: He quitado "import os" porque en este script no lo usas.
# ChatOpenAI buscará automáticamente la clave OPENAI_API_KEY que cargamos abajo.

# ─────────────────────────────────────────────────────────────
# 2. CARGA DEL ENTORNO (.env)
# ─────────────────────────────────────────────────────────────
# Lee el archivo .env de la carpeta actual y "mete" las claves
# en las variables de entorno del sistema.
# SIEMPRE debe ir antes de inicializar el modelo.
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 3. INICIALIZACIÓN DEL MODELO (LLM)
# ─────────────────────────────────────────────────────────────
# Creamos una instancia (un objeto) del modelo que vamos a usar.
# - model_name: "gpt-4o-mini" (rápido, barato y potente).
# - temperature: Controla la "creatividad" o aleatoriedad.
#   Va de 0.0 (muy lógico y determinista) a 1.0 (muy creativo).
#   Para datos exactos (como el año de la luna) usa 0.0. Aquí pusiste 0.7 (más natural).
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# ─────────────────────────────────────────────────────────────
# 4. EJECUCIÓN DE LA PREGUNTA
# ─────────────────────────────────────────────────────────────
pregunta = "¿En qué año llegó el ser humano a la luna por primera vez?"
print("→ Pregunta:", pregunta)

# .invoke() es el método estándar de LangChain para enviar algo al modelo
# y esperar a que termine de responder (es una llamada síncrona/bloqueante).
# Lo que devuelve NO es un texto plano, sino un objeto llamado "AIMessage".
respuesta = llm.invoke(pregunta)

# Por eso usamos ".content". Es la propiedad del objeto AIMessage
# que contiene únicamente el texto de la respuesta.
print("→ Respuesta del modelo:", respuesta.content)