import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

print("→ Clave OpenAI detectada:", "sí" if os.getenv("OPENAI_API_KEY") else "NO")
print("→ Trazas LangSmith:", os.getenv("LANGSMITH_TRACING"))
print("→ Proyecto LangSmith:", os.getenv("LANGSMITH_PROJECT"))

model = init_chat_model("openai:gpt-4o-mini", temperature=0)
respuesta = model.invoke("Responde solo con: entorno configurado correctamente")
print("→ Respuesta del modelo:", respuesta.content)
