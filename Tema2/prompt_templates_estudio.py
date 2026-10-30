"""
================================================================================
MATERIAL DE ESTUDIO: PROMPT TEMPLATES EN LANGCHAIN
================================================================================
Este archivo combina y compara las DOS familias de templates de prompts:

    ┌────────────────────────────┬──────────────────────────────────────────┐
    │ PromptTemplate             │ ChatPromptTemplate                       │
    ├────────────────────────────┼──────────────────────────────────────────┤
    │ Produce UN STRING          │ Produce una LISTA DE MENSAJES            │
    │ Modelos "completion"       │ Modelos "chat" (GPT-4, Claude, etc.)     │
    │ (legacy)                   │ (estándar actual)                        │
    │ .format() → str            │ .format_messages() → list[BaseMessage]   │
    │ Sin roles                  │ Con roles: system / human / ai           │
    └────────────────────────────┴──────────────────────────────────────────┘

CONCEPTO CENTRAL
----------------
Un "prompt template" es una PLANTILLA con HUECOS (variables). Separar la
plantilla (fija) de los valores (variables) es lo que permite reutilizar el
mismo prompt con MIL entradas distintas sin reescribirlo.

    SIN template:                        CON template:
    "Eres un experto en marketing.       template = "Eres un experto... {producto}"
    Sugiere un eslogan para café"        template.format(producto="café")
    "Eres un experto en marketing.       template.format(producto="té")
     Sugiere un eslogan para té"         template.format(producto="agua")

    → código duplicado, frágil           → una plantilla, N usos

¿POR QUÉ IMPORTA EN LANGCHAIN?
    Los templates son el PRIMER eslabón de casi cualquier cadena LCEL:
        prompt_template | llm | parser
    Permiten:
        • Reutilización (una plantilla, N inputs).
        • Testeo (puedes verificar la plantilla sin llamar al LLM).
        • Composición con LCEL (son Runnables: aceptan `|`).
        • Trazabilidad (LangSmith ve qué prompt exacto se envió).

FILOSOFÍA DEL ESTUDIO
---------------------
Igual que en las Partes 1 y 2:
    • Encabezados conceptuales ("¿por qué?").
    • Código comentado ("¿cómo?").
    • Notas "🔎 Para profundizar" con matices y errores comunes.

🆕 CONVENCIÓN DEL PROYECTO
--------------------------
Aunque este archivo NO llame al LLM en su flujo principal, incluimos
`load_dotenv()` como PATRÓN ESTÁNDAR al inicio. Motivos:
    1. Consistencia: todos los archivos del curso siguen la misma cabecera.
    2. Preparación: si mañana añades `llm.invoke(...)`, ya funciona sin tocar
       la cabecera.
    3. Seguridad: evita hardcodear claves si algún ejercicio las necesita.
================================================================================
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
# ¿POR QUÉ `langchain_core.prompts`?
#   `langchain_core` es el paquete base con las PRIMITIVAS (independientes de
#   proveedor). Los templates viven aquí porque son "agnósticos": el mismo
#   PromptTemplate sirve para OpenAI, Anthropic o un modelo local.
#
# ¿POR QUÉ DOS IMPORTS DE PROMPTS?
#   Son DOS clases distintas, con APIs distintas:
#     • PromptTemplate     → produce un str
#     • ChatPromptTemplate → produce una lista de mensajes con roles
#   No hay "una que englobe a la otra". Conviven y se eligen según el modelo.
# ------------------------------------------------------------------------------
import os                                       # 🆕 Para leer la API key del entorno

from dotenv import load_dotenv                   # 🆕 Cargar .env → os.environ

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 🔎 Para profundizar
#   • También existe `FewShotPromptTemplate` (prompts con ejemplos),
#     `PipelinePromptTemplate` (prompts compuestos) y `MessagesPlaceholder`
#     (para insertar historial de chat). Los verás en proyectos reales.
#   • En LangChain antiguo (<0.2) estos imports vivían en `langchain.prompts`.
#     Ahora están en `langchain_core.prompts` por la reorganización modular.
#   • 🆕 No importamos `ChatOpenAI` aquí porque este archivo no llama al LLM
#     en su flujo principal. Aparecerá solo en la sección 5 (opcional).
#     Mantener los imports "honestos" (solo lo que se usa) es buena práctica.


# ==============================================================================
# 2. 🆕 CARGA DE VARIABLES DE ENTORNO (patrón estándar del proyecto)
# ==============================================================================
# ¿POR QUÉ CARGAR .env SI ESTE ARCHIVO NO LLAMA AL LLM?
#   Porque es el PATRÓN ESTÁNDAR del curso. Cargar variables al inicio:
#     • Es barato (milisegundos, un archivo pequeño).
#     • No tiene efectos secundarios (no imprime, no falla si no existe).
#     • Deja el archivo LISTO para añadir un LLM sin tocar la cabecera.
#     • Evita "sorpresas" si en un futuro ejercicio se importa este archivo
#       desde otro que sí use el LLM.
#
# ¿QUÉ HACE load_dotenv()?
#   1. Busca ".env" en el cwd y en directorios padres.
#   2. Lee líneas `CLAVE=valor` (ignora comentarios con `#`).
#   3. Escribe las claves en os.environ SOLO si no existen ya.
# ------------------------------------------------------------------------------
load_dotenv()

# ¿POR QUÉ UN CHEQUEO "SUAVE" (WARNING) Y NO FAIL-FAST AQUÍ?
#   En las Partes 1 y 2 hacíamos `raise RuntimeError` porque el archivo
#   SÍ llamaba al LLM: sin clave, no había nada que ejecutar.
#   Aquí el flujo principal NO necesita la clave. Un fail-fast impediría
#   estudiar los templates (que es el objetivo del archivo).
#   Por eso usamos un WARNING: informamos, pero no bloqueamos.
#   → Regla general: fail-fast si la clave es IMPRESCINDIBLE; warning si es
#     OPCIONAL (como aquí, donde solo hace falta en la sección 5).
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Aviso: OPENAI_API_KEY no está definida. "
          "Este archivo funciona sin ella, pero la sección 5 no podrá ejecutarse.")

# 🔎 Para profundizar
#   • Si copias este archivo a un entorno donde NO habrá LLM (p. ej. un
#     script de tests unitarios de plantillas), puedes omitir load_dotenv()
#     sin problema. La convención es para el curso, no una obligación.
#   • `load_dotenv()` NO lanza error si no existe `.env`: simplemente no
#     hace nada. Es seguro llamarlo siempre.
#   • Nunca imprimas la clave. En este bloque solo comprobamos EXISTENCIA
#     con `os.getenv(...)` → bool, sin exponer el valor.


# ==============================================================================
# 3. PROMPTTEMPLATE — la familia "clásica" (string)
# ==============================================================================
# ¿QUÉ PRODUCE?
#   Un STRING plano. Ideal para modelos "completion" (los pre-ChatGPT, tipo
#   text-davinci-003). Hoy en día es LEGACY, pero sigue siendo útil porque:
#     • Es la forma más simple de entender el concepto de "plantilla".
#     • Algunos wrappers y utilidades internas siguen usándolo.
#     • Verás código antiguo en tutoriales y repos.
#
# ¿CÓMO FUNCIONA?
#   1. Defines una plantilla con HUECOS entre llaves: {variable}.
#   2. Declaras qué variables espera (input_variables).
#   3. Llamas a .format(**kwargs) para rellenarlos → str final.
# ------------------------------------------------------------------------------

# --- 3.1 Plantilla como string ---
template = "Eres un experto en marketing. Sugiere un eslogan creativo para un producto {producto}"
#                                                                           └──────┘
#                                                           HUECO a rellenar en .format()

# ¿POR QUÉ NO CONCATENAR DIRECTAMENTE CON f-string?
#   Podrías hacer f"... producto {producto}". Funciona, pero:
#     ❌ La plantilla queda "atada" al valor (no puedes reutilizarla).
#     ❌ No puedes validar variables ni detectar errores de formato.
#     ❌ No se integra con LCEL (no tiene `|`).
#   El template desacopla la FORMA (fija) de los DATOS (variables).

prompt = PromptTemplate(
    template=template,
    input_variables=["producto"],   # Declaración explícita de variables
)
# ¿POR QUÉ DECLARAR `input_variables` SI YA SE VEN EN EL STRING?
#   • Validación temprana: si en el futuro cambias el template y añades u
#     olvidas una variable, LangChain puede avisar ANTES de llamar al LLM.
#   • Documentación implícita: quien lee el código sabe qué necesita.
#   • En versiones modernas puedes omitirlo (LangChain infiere del template),
#     pero declararlo es buena práctica y a prueba de futuro.

# --- 3.2 Rellenar la plantilla ---
prompt_lleno = prompt.format(producto="café orgánico")
#                            └──────────────────────┘
#                            Sustituye el hueco {producto}

print("=" * 70)
print("3. PromptTemplate (familia clásica)")
print("=" * 70)
print(prompt_lleno)
# Salida:
#   Eres un experto en marketing. Sugiere un eslogan creativo para un
#   producto café orgánico
print(f"\n[DEBUG] Tipo del resultado: {type(prompt_lleno).__name__}")  # → str
print(f"[DEBUG] Variables esperadas: {prompt.input_variables}")

# 🔎 Para profundizar
#   • ¿Qué pasa si olvidas rellenar una variable?
#       prompt.format()  →  KeyError: 'producto'
#     Es un "fail-fast" útil: detectas el error ANTES de gastar tokens.
#   • ¿Qué pasa si pasas una variable EXTRA?
#       prompt.format(producto="X", color="Y")  →  se ignora "color".
#     LangChain no valida estrictamente por defecto en PromptTemplate.
#   • `.format()` devuelve un `str`. Es lo que se pasa a `llm.invoke(str)`
#     en modelos chat (el string se envuelve como HumanMessage).
#   • Hay variantes: `.format_prompt()` devuelve un `StringPromptValue`,
#     que es un objeto (no un str) útil para ciertas cadenas. Para el 99%
#     de los casos, `.format()` es lo que quieres.


# ==============================================================================
# 4. CHATPROMPTTEMPLATE — la familia "moderna" (mensajes con rol)
# ==============================================================================
# ¿QUÉ PRODUCE?
#   Una LISTA DE MENSAJES, cada uno con un ROL:
#     • "system"    → instrucciones permanentes (tono, formato, restricciones)
#     • "human"     → lo que dice el usuario
#     • "ai"        → lo que responde el asistente (para historial)
#     • "tool"      → salidas de herramientas (function calling)
#
# ¿POR QUÉ ESTA ES LA "MODERNA"?
#   Todos los modelos actuales (GPT-4o, Claude, Gemini…) son CHAT models:
#   esperan una lista de mensajes con roles, no un string plano. La API
#   distingue system de human internamente. ChatPromptTemplate refleja
#   esa estructura directamente.
#
# ¿CÓMO FUNCIONA?
#   1. Defines una lista de tuplas (rol, contenido).
#   2. El contenido puede tener huecos {variable}.
#   3. Llamas a .format_messages(**kwargs) → list[BaseMessage].
# ------------------------------------------------------------------------------

# --- 4.1 Plantilla como lista de mensajes ---
chat_prompt = ChatPromptTemplate.from_messages([
    # ¿POR QUÉ "system" PRIMERO?
    #   El mensaje de sistema establece el CONTEXTO PERSISTENTE: personalidad,
    #   idioma, formato de salida. Va primero y NUNCA se varía entre llamadas
    #   del mismo "diálogo". Es como el "contrato" del asistente.
    ("system", "Eres un traductor del español al inglés muy preciso."),

    # ¿POR QUÉ "human" DESPUÉS?
    #   Es la entrada del usuario. Aquí va la parte variable del prompt.
    #   El hueco {texto} se rellena en .format_messages().
    ("human", "{texto}"),
])

# ¿POR QUÉ `from_messages` Y NO `ChatPromptTemplate(...)` DIRECTO?
#   `from_messages` es el CONSTRUCTOR RECOMENDADO. Acepta tuplas
#   (rol, contenido) o instancias de BaseMessage (SystemMessage, HumanMessage…).
#   Usar tuplas es más legible para prompts cortos; las instancias se
#   reservan para casos avanzados (plantillas dentro de mensajes, etc.).
#
# ¿POR QUÉ TUPLAS Y NO STRINGS SUELTOS?
#   Porque necesitamos el ROL. Un string "Eres un traductor..." no dice
#   si es system o human. Las tuplas hacen explícito el rol y permiten al
#   LLM tratarlo con la prioridad adecuada (system > human en jerarquía).

# --- 4.2 Rellenar la plantilla ---
mensajes = chat_prompt.format_messages(texto="Hola mundo, ¿cómo estás?")
#                                      └────────────────────────────┘
#                                      Sustituye el hueco {texto}

print("\n" + "=" * 70)
print("4. ChatPromptTemplate (familia moderna)")
print("=" * 70)

for m in mensajes:
    # `type(m).__name__` → nos dice QUÉ TIPO de mensaje es.
    # `m.content`        → el texto ya relleno.
    print(f"  {type(m).__name__:15s} → {m.content}")

# Salida esperada:
#   SystemMessage   → Eres un traductor del español al inglés muy preciso.
#   HumanMessage    → Hola mundo, ¿cómo estás?
print(f"\n[DEBUG] Tipo del resultado: {type(mensajes).__name__}")   # → list
print(f"[DEBUG] Longitud de la lista: {len(mensajes)}")              # → 2
print(f"[DEBUG] Variables esperadas: {chat_prompt.input_variables}")

# 🔎 Para profundizar
#   • ¿Qué es un `BaseMessage`? La clase base de todos los mensajes:
#       SystemMessage, HumanMessage, AIMessage, ToolMessage, FunctionMessage.
#     Cada uno tiene `.content` (str o lista) y `.additional_kwargs` (dict).
#   • `.format_messages()` devuelve `list[BaseMessage]`. Esto es EXACTAMENTE
#     lo que un chat model espera: `llm.invoke(mensajes)`.
#   • Algunos modelos NO soportan "system" (o lo tratan distinto). En esos
#     casos puedes usar `("human", "Eres un traductor...")` como workaround,
#     pero es menos idiomático.
#   • Puedes añadir un turno de "ai" para hacer few-shot:
#       ("human", "Hola"), ("ai", "Hello"), ("human", "{texto}")
#     Esto le muestra al modelo un ejemplo de cómo responder.


# ==============================================================================
# 5. COMPARATIVA DIRECTA: ¿CUÁNDO USAR CUÁL?
# ==============================================================================
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

# 🔎 Para profundizar
#   • Ambos son RUNNABLES. Puedes hacer:
#         chain = chat_prompt | llm | parser
#     `chat_prompt` recibe un dict {"texto": "..."} y devuelve mensajes;
#     `llm` recibe mensajes y devuelve AIMessage; `parser` lo convierte.
#   • Puedes convertir uno en otro:
#         chat_prompt = ChatPromptTemplate.from_template(template)
#     pero NO al revés directamente (los roles se pierden).
#   • `ChatPromptTemplate` tiene `.from_template()` y `.from_messages()`:
#       - `.from_template(t)`      → atajo para 1 solo HumanMessage.
#       - `.from_messages([...])`  → control total (lo que usamos).


# ==============================================================================
# 6. 🆕 CONEXIÓN CON EL LLM (usando el .env cargado en la sección 2)
# ==============================================================================
# Este bloque SÍ puede ejecutarse (la clave ya está cargada en os.environ).
# Está protegido por `if False` para no gastar tokens al estudiar. Cámbialo
# a `if True` cuando quieras ver la cadena completa en acción.
#
# Flujo LCEL:
#     {"texto": "..."} → chat_prompt → [SystemMessage, HumanMessage]
#                                     → llm → AIMessage
#                                     → .content (str)
# ------------------------------------------------------------------------------
if False:  # Cambia a True para probar (necesita .env con OPENAI_API_KEY)
    from langchain_openai import ChatOpenAI
    # 🆕 Import LOCAL: solo se ejecuta si activas el bloque. Es una buena
    # práctica cuando el import es caro o solo se usa en una rama opcional.
    # No contamina la cabecera del archivo con dependencias no esenciales
    # para el flujo principal (los templates).

    # 🆕 ChatOpenAI lee OPENAI_API_KEY de os.environ automáticamente
    # (cargada por load_dotenv() en la sección 2). No hace falta pasar api_key.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- OPCIÓN A: ChatPromptTemplate + ChatOpenAI (patrón moderno) ---
    #
    # 1. `chat_prompt` recibe {"texto": "..."} → devuelve list[Message].
    # 2. `llm` recibe list[Message]           → devuelve AIMessage.
    # 3. `|` los encadena.
    chain = chat_prompt | llm
    respuesta = chain.invoke({"texto": "Hola mundo, ¿cómo estás?"})
    print("\n[Sección 6] Respuesta del traductor:", respuesta.content)
    # → "Hello world, how are you?"

    # --- OPCIÓN B: PromptTemplate + ChatOpenAI (compatibilidad) ---
    #
    # Un ChatOpenAI también acepta strings. Internamente los envuelve
    # como HumanMessage. Es lo que pasaba en las Partes 1 y 2 cuando
    # hacíamos `llm.invoke(prompt_str)`.
    cadena_simple = prompt | llm
    respuesta2 = cadena_simple.invoke({"producto": "café orgánico"})
    print("[Sección 6] Eslogan sugerido:", respuesta2.content)

# 🔎 Para profundizar
#   • Fíjate en el cambio de CONTRATO de entrada:
#       PromptTemplate       → espera un dict con las variables
#       ChatPromptTemplate   → espera un dict con las variables
#     Ambos son Runnables: el input es un dict, el output es "algo" que el
#     siguiente eslabón consume.
#   • ¿Por qué `chain.invoke({"texto": "..."})` y no `chain.invoke("...")`?
#     Porque la cadena espera un DICT con los nombres de las variables.
#     Pasar un string directamente daría error (o se interpretaría como
#     entrada suelta, según el template).
#   • Con `temperature=0.7` el eslogan será creativo y variado. Con 0,
#     será siempre el mismo. Elige según el caso de uso.
#   • 🆕 Aquí se ve POR QUÉ cargamos .env en la sección 2 aunque el flujo
#     principal no lo necesite: en el momento en que activas este bloque,
#     la clave ya está disponible sin tocar nada más.


# ==============================================================================
# 7. RESUMEN DE PATRONES REUTILIZABLES
# ==============================================================================
#
# 1. 🆕 PATRÓN "CABECERA ESTÁNDAR CON dotenv"
#    load_dotenv() + os.getenv(...) SIEMPRE al inicio. Warning si la clave
#    es opcional; fail-fast si es imprescindible.
#
# 2. PATRÓN "PLANTILLA + VARIABLES"
#    Define UNA plantilla con huecos {var} y rellénala con .format() o
#    .format_messages(). Reutilización total.
#
# 3. PATRÓN "SYSTEM + HUMAN"
#    En ChatPromptTemplate, empieza con un system (contexto) y sigue con
#    human (entrada). Es el esqueleto de todo prompt conversacional.
#
# 4. PATRÓN "DECLARAR input_variables"
#    Aunque LangChain lo infiere, declararlo documenta y valida temprano.
#
# 5. PATRÓN "FAIL-FAST CON format()"
#    Si falta una variable, .format() lanza KeyError → detectas antes de
#    gastar tokens en una llamada a la API.
#
# 6. PATRÓN "TEMPLATES SON RUNNABLES"
#    Ambos se pueden encadenar con `|` directamente a un LLM o a un parser.
#    No necesitas "convertirlos" antes.
#
# 7. PATRÓN "ELEGIR SEGÚN MODELO"
#    Chat models → ChatPromptTemplate. Completion models → PromptTemplate.
#    En 2024+, casi siempre es ChatPromptTemplate.
#
# ==============================================================================
# 🎯 EJERCICIOS PROPUESTOS
# ==============================================================================
# A) Modifica el template de marketing para que pida TAMBIÉN "tono"
#    (formal/informal). Pista: añade un hueco {tono} y pásalo en .format().
#
# B) Crea un ChatPromptTemplate con TRES mensajes:
#       system  → "Eres un chef experto."
#       human   → "Dame una receta con {ingrediente}."
#       ai      → "Aquí tienes una receta con {ingrediente}: ..."
#    Formatea y observa los tipos de mensaje.
#
# C) ¿Qué pasa si en `PromptTemplate` llamas a `.format()` SIN el argumento
#    `producto`? Ejecuta y lee el error. ¿Por qué es útil ese error?
#
# D) Convierte el `PromptTemplate` de marketing en un `ChatPromptTemplate`
#    usando `.from_template(template)`. ¿Cuántos mensajes y de qué tipo
#    produce? Compara con el original.
#
# E) Añade un mensaje "ai" al `chat_prompt` del traductor con un ejemplo
#    few-shot: ("human", "Buenos días"), ("ai", "Good morning").
#    ¿Cambia el comportamiento del modelo? (Esto es few-shot prompting.)
#
# F) 🆕 Comenta la sección 2 (`load_dotenv()`) y activa la sección 6
#    (`if False` → `if True`). ¿Qué error obtienes al invocar la cadena?
#    ¿Por qué es útil el patrón de la cabecera estándar?
#
# G) (Avanzado) Descomenta la sección 6 y prueba AMBOS pipelines (A y B).
#    Mide latencia con `time.perf_counter()` y compara los resultados.
#    ¿En qué se diferencian las salidas de A y B?
#
# ==============================================================================
# 🔗 CONEXIÓN CON LAS PARTES 1 Y 2
# ==============================================================================
# En las Partes 1 y 2, los prompts eran F-STRINGS dentro de cada función:
#
#     prompt = f"Analiza el sentimiento del siguiente texto: {text}"
#     response = llm.invoke(prompt)
#
# Funciona, pero:
#     ❌ El prompt está "atado" al código (no reutilizable).
#     ❌ No puedes testear el prompt sin ejecutar la función.
#     ❌ No puedes cambiarlo sin tocar el código (imagina prompts en A/B).
#     ❌ No hay validación de variables ni trazabilidad limpia.
#
# Con PromptTemplate / ChatPromptTemplate:
#     ✅ Prompt separado del código → reutilizable.
#     ✅ Testeable: `chat_prompt.format_messages(...)` sin llamar al LLM.
#     ✅ Composable: `chat_prompt | llm | parser` (LCEL).
#     ✅ Trazable: LangSmith ve qué prompt exacto se envió.
#     ✅ Parametrizable: cambias la plantilla en BD o en YAML.
#
# El siguiente paso natural es REESCRIBIR las Partes 1 y 2 usando
# ChatPromptTemplate en lugar de f-strings. Pruébalo como ejercicio:
# verás cómo el código queda más limpio y modular. 🚀
#
# Y recuerda la CONVENCIÓN DEL PROYECTO: todo archivo del curso empieza con
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()
# aunque el flujo principal no use el LLM. Es preparación + consistencia.
# ==============================================================================