# ─────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ─────────────────────────────────────────────────────────────

# dotenv es la librería que permite leer el archivo .env y cargar las variables de entorno donde se guardan las claves de API.
from dotenv import load_dotenv
# langchain_deepseek es la librería que permite usar el modelo de DeepSeek.
# ChatDeepSeek es la clase que representa el modelo de chat de DeepSeek.
from langchain_deepseek import ChatDeepSeek

# ─────────────────────────────────────────────────────────────
# 2. CARGA DEL ENTORNO (.env)
# ─────────────────────────────────────────────────────────────
# Carga las variables de tu archivo .env.
# Asegúrate de que tienes configurada la clave DEEPSEEK_API_KEY.
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 3. INICIALIZACIÓN DEL MODELO (LLM)
# ─────────────────────────────────────────────────────────────
# Usamos ChatDeepSeek, la clase específica para este proveedor.
# - model: "deepseek-chat" es el modelo general (rápido y versátil).
#   (También existe "deepseek-reasoner" para tareas de razonamiento complejo).
# - temperature: igual que antes, controla la creatividad (0.0 = determinista).
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# ─────────────────────────────────────────────────────────────
# 4. EJECUCIÓN DE LA PREGUNTA
# ─────────────────────────────────────────────────────────────
pregunta = "¿En qué año se enunció la teoría de la relatividad?"
print("→ Pregunta:", pregunta)

# .invoke() envía la pregunta y espera la respuesta completa.
# El objeto devuelto es un AIMessage, igual que con OpenAI.
respuesta = llm.invoke(pregunta)

# .content extrae únicamente el texto de la respuesta.
print("→ Respuesta del modelo (DeepSeek):", respuesta.content)