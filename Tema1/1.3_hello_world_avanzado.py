# ─────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ─────────────────────────────────────────────────────────────
# python-dotenv: Nos permite leer el archivo .env donde guardamos las claves.
from dotenv import load_dotenv

# ChatDeepSeek: La clase específica de LangChain para conectar con la API de DeepSeek.
from langchain_deepseek import ChatDeepSeek

# PromptTemplate: La herramienta para crear "moldes" de texto.
# Ojo: Se importa de "langchain_core", NO de "langchain.prompts" (la ruta antigua). Esta ruta es la correcta en LangChain 1.0 y posteriores.
# Si usas la ruta antigua, te dará un error de importación. from langchain.prompts import PromptTemplate Esta será la ruta antigua y obsoleta. No la uses.
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────
# 2. CARGA DEL ENTORNO
# ─────────────────────────────────────────────────────────────
# Lee el archivo .env, busca DEEPSEEK_API_KEY y la carga en la memoria del sistema.
# Es obligatorio ponerlo antes de inicializar el modelo.
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 3. INICIALIZACIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────
# Creamos el objeto que representa a la IA.
# - model: "deepseek-chat" es el modelo general (rápido y versátil).
# - temperature: 0.7 le da un toque de "creatividad". (0.0 sería muy robótico).
chat = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# ─────────────────────────────────────────────────────────────
# 4. CREACIÓN DE LA PLANTILLA (El "molde")
# ─────────────────────────────────────────────────────────────
# Una plantilla es un texto fijo con "huecos" (variables) que se rellenan después.
# - input_variables: Le dice a LangChain qué palabras entre llaves {} deben ser reemplazadas.
# - template: Es el texto real. El hueco aquí es "{nombre}".
plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre. \nNombre: {nombre}\nAsistente:"
)

# ─────────────────────────────────────────────────────────────
# 5. CREACIÓN DE LA CADENA (LCEL)
# ─────────────────────────────────────────────────────────────
# Aquí ocurre la magia. Usamos el operador tubería "|" (Pipe) de LCEL.
# Runnable sequences (LCEL) permite encadenar varios pasos de procesamiento.
# En este caso, la plantilla se rellena primero, y luego el resultado se pasa al chat de DeepSeek.
# Esto significa: "El resultado de la plantilla, pásaselo al chat".
# (Esto sustituye al antiguo y obsoleto LLMChain).
cadena = plantilla | chat

# ─────────────────────────────────────────────────────────────
# 6. EJECUCIÓN
# ─────────────────────────────────────────────────────────────
# Invocamos la cadena con .invoke().
# Le pasamos un diccionario con el valor de la variable ("nombre": "Juan").
# LangChain cogerá el diccionario, rellenará la plantilla, y enviará el texto resultante a DeepSeek.
resultado = cadena.invoke({"nombre": "Juan"}) # Proporcionamos el valor de la variable "nombre" como un diccionario.

# "resultado" es un objeto AIMessage, no un simple texto.
# Usamos .content para extraer únicamente la respuesta de texto generada por la IA.
print("→ Respuesta de DeepSeek:", resultado.content)