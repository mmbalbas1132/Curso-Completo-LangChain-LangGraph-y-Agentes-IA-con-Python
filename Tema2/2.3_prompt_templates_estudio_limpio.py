"""
Prompt Templates en LangChain: PromptTemplate vs ChatPromptTemplate.
"""

import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Aviso: OPENAI_API_KEY no está definida. "
          "Este archivo funciona sin ella, pero la sección 5 no podrá ejecutarse.")

# --- PromptTemplate (string) ---
template = "Eres un experto en marketing. Sugiere un eslogan creativo para un producto {producto}"

prompt = PromptTemplate(
    template=template,
    input_variables=["producto"],
)

prompt_lleno = prompt.format(producto="café orgánico")

print("=" * 70)
print("3. PromptTemplate (familia clásica)")
print("=" * 70)
print(prompt_lleno)
print(f"\n[DEBUG] Tipo del resultado: {type(prompt_lleno).__name__}")
print(f"[DEBUG] Variables esperadas: {prompt.input_variables}")

# --- ChatPromptTemplate (mensajes con rol) ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso."),
    ("human", "{texto}"),
])

mensajes = chat_prompt.format_messages(texto="Hola mundo, ¿cómo estás?")

print("\n" + "=" * 70)
print("4. ChatPromptTemplate (familia moderna)")
print("=" * 70)

for m in mensajes:
    print(f"  {type(m).__name__:15s} → {m.content}")

print(f"\n[DEBUG] Tipo del resultado: {type(mensajes).__name__}")
print(f"[DEBUG] Longitud de la lista: {len(mensajes)}")
print(f"[DEBUG] Variables esperadas: {chat_prompt.input_variables}")

# --- Comparativa ---
print("\n" + "=" * 70)
print("5. Comparativa: PromptTemplate vs ChatPromptTemplate")
print("=" * 70)
print("""
    ┌────────────────────────┬──────────────────────────────────────┐
    │ PromptTemplate         │ ChatPromptTemplate                   │
    ├────────────────────────┼──────────────────────────────────────┤
    │ .format() → str        │ .format_messages() → list[Message]   │
    │ Sin roles              │ Con roles (system/human/ai)          │
    │ Modelos "completion"   │ Modelos "chat" (TODOS los actuales)  │
    │ LEGACY, en desuso      │ ESTÁNDAR, recomendado                │
    │ Simple, directo        │ Expresivo, escalable a diálogos      │
    └────────────────────────┴──────────────────────────────────────┘

    Regla práctica:
        • ¿Vas a usar GPT-4o, Claude, Gemini?    → ChatPromptTemplate
        • ¿Mantienes código con text-davinci?    → PromptTemplate
        • ¿Prototipo rapidísimo en notebook?     → PromptTemplate (vale)
        • ¿Pipeline en producción?               → ChatPromptTemplate
""")

# --- Conexión con el LLM (opcional) ---
if True:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Opción A: ChatPromptTemplate + ChatOpenAI
    chain = chat_prompt | llm
    respuesta = chain.invoke({"texto": "Hola mundo, ¿cómo estás?"})
    print("\n[Sección 6] Respuesta del traductor:", respuesta.content)

    # Opción B: PromptTemplate + ChatOpenAI
    cadena_simple = prompt | llm
    respuesta2 = cadena_simple.invoke({"producto": "café orgánico"})
    print("[Sección 6] Eslogan sugerido:", respuesta2.content)