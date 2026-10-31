# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
# Importamos las clases necesarias. Además del contenedor principal
# (ChatPromptTemplate), usamos las dos sub-plantillas especializadas.
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ==============================================================================
# 2. DEFINICIÓN DE LAS SUB-PLANTILLAS (El "Lego" del prompt)
# ==============================================================================

# ➕ Añadido: `SystemMessagePromptTemplate` crea un molde que, al formatearse,
# se convertirá en un objeto `SystemMessage`. Es ideal para inyectar la
# personalidad del bot de forma dinámica.
plantilla_sistema = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
)

# ➕ Añadido: `HumanMessagePromptTemplate` crea un molde para el turno del
# usuario. En este caso, inyecta el tema y la pregunta específica.
plantilla_humano = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)


# ==============================================================================
# 3. ENSAMBLAJE DEL PROMPT FINAL
# ==============================================================================
# ➕ Añadido: `from_messages` une las piezas en el orden exacto en que queremos
# que el LLM las reciba: primero el contexto (sistema) y luego la acción (humano).
# Podría tener estas plantillas en archivos separados y cargarlas dinámicamente, pero aquí las definimos inline.
chat_prompt = ChatPromptTemplate.from_messages([
    plantilla_sistema,
    plantilla_humano
])


# ==============================================================================
# 4. INYECCIÓN DE DATOS (El "Relleno")
# ==============================================================================
# ➕ Añadido: Al llamar a `format_messages`, LangChain recorre ambas plantillas,
# detecta los 5 huecos {entre_llaves} y los sustituye por los valores del dict.
# El resultado es una lista de objetos BaseMessage listos para el LLM.
mensajes = chat_prompt.format_messages(
    rol="nutricionista",
    especialidad="dietas veganas",
    tono="profesional pero accesible",
    tema="proteínas vegetales",
    pregunta="¿Cuáles son las mejores fuentes de proteína vegana para un atleta profesional?"
)


# ==============================================================================
# 5. VISUALIZACIÓN
# ==============================================================================
# ➕ Añadido: Iteramos sobre los mensajes generados. Comprobaremos que hay 2:
# un SystemMessage (el primero) y un HumanMessage (el segundo).
for m in mensajes:
    print(m.content)