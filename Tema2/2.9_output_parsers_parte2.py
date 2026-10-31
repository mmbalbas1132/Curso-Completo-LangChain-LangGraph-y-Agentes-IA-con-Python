# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
# ➕ Añadido: Importamos `os` y `load_dotenv` para cargar la API Key.
#
# 🧠 ¿Por qué `os`? Porque LangChain, internamente, lee las credenciales
#    desde variables de entorno. No las pasas como argumento al modelo:
#    el SDK de OpenAI busca `OPENAI_API_KEY` en el entorno del proceso.
#    `os` nos permite interactuar con ese entorno (aunque aquí no lo usemos
#    directamente, sí lo usan las librerías por debajo).
import os
from dotenv import load_dotenv

# ➕ Añadido: `Field` nos permite añadir descripciones y reglas a cada campo,
# lo cual es VITAL para que el LLM entienda qué queremos extraer.
#
# 🧠 `BaseModel` es la clase base de Pydantic. Todo esquema que quieras usar
#    como "contrato" de salida debe heredar de ella.
# 🧠 `Field(...)` es un helper que envuelve la definición de un atributo para
#    añadir metadatos: descripción, valor por defecto, validaciones, etc.
#    Cuando LangChain genera el JSON Schema que enviará al LLM, esos
#    metadatos viajan dentro del propio schema y el modelo los "lee" como
#    instrucciones. Por eso la `description` NO es decorativa: es prompt.
from pydantic import BaseModel, Field

# 🧠 `ChatOpenAI` es la clase "wrapper" de LangChain sobre el cliente oficial
#    de OpenAI. Nos da una interfaz uniforme (`invoke`, `stream`, `batch`…)
#    que es la misma para todos los proveedores soportados.
from langchain_openai import ChatOpenAI


# ==============================================================================
# 2. CARGA DE VARIABLES DE ENTORNO (Patrón estándar)
# ==============================================================================
# ➕ Añadido: Cargamos el archivo .env. Sin esto, ChatOpenAI no sabe
# cuál es tu clave de OpenAI y lanzará un error de autenticación.
#
# 🧠 `load_dotenv()` busca un archivo llamado `.env` en el directorio actual
#    (o en los padres) y vuelca su contenido a `os.environ`. A partir de esa
#    línea, cualquier librería que haga `os.getenv("OPENAI_API_KEY")`
#    encontrará el valor.
# ⚠️ Debe llamarse ANTES de instanciar cualquier cliente que necesite keys.
load_dotenv()


# ==============================================================================
# 3. DEFINICIÓN DEL ESQUEMA (El "Contrato" de salida)
# ==============================================================================
class AnalisisTexto(BaseModel):
    # 🧠 Una clase que hereda de `BaseModel` es un "esquema": declara qué
    #    campos forman el objeto, de qué tipo son y qué significan.
    # 🧠 `resumen: str` significa "este campo es obligatorio (no tiene
    #    default) y debe ser un string". Si el LLM intentara meter un número,
    #    Pydantic lanzaría un `ValidationError` automáticamente.

    # ➕ Añadido: El LLM leerá la `description` para saber exactamente qué poner aquí.
    # 🧠 `Field(description=...)` NO es solo documentación para humanos:
    #    LangChain incluye esa frase en el JSON Schema que le pasa al modelo.
    #    Cuanto más clara sea la descripción, mejor será la extracción.
    resumen: str = Field(description="Resumen breve del texto.")

    # ➕ Añadido: En este caso, le damos una pista de los valores posibles en la descripción.
    # 🧠 Aquí el tipo sigue siendo `str` (no un Enum), pero le "sugerimos"
    #    los valores esperados en la propia descripción. Es una técnica común
    #    cuando quieres flexibilidad (p. ej. "Positivo" vs "positivo") sin
    #    forzar un `Literal[...]` que fallaría con variaciones de mayúsculas.
    # 💡 Sugerencia (no aplicada): podrías usar `Literal["Positivo", "neutro", "negativo"]`
    #    para forzar valores exactos. Se mantiene el original por respeto al
    #    código del alumno; cámbialo tú si quieres experimentar.
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo)")


# ==============================================================================
# 4. CONFIGURACIÓN DEL MODELO
# ==============================================================================
# ➕ Añadido: Usamos una temperatura de 0.6. Aunque para extracción de datos
# suele preferirse 0 (determinista), aquí LangChain obliga al modelo a seguir
# el esquema, por lo que una temperatura más alta permite resúmenes más naturales.
#
# 🧠 `model="gpt-4o-mini"` → el identificador exacto del modelo en la API de
#    OpenAI. Si lo cambias por uno inexistente, obtendrás un 404.
# 🧠 `temperature=0.6` → parámetro de "creatividad" (0.0 = determinista,
#    1.0+ = muy variado). Para structured output, valores bajos son más
#    seguros, pero 0.6 es aceptable porque el schema "ancla" la salida.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)


# ==============================================================================
# 5. MAGIA: FORZANDO LA ESTRUCTURA (Structured Output)
# ==============================================================================
# ➕ Añadido: Aquí ocurre la magia. `with_structured_output` envuelve al LLM.
# Le dice: "No me devuelvas texto libre. Usa la herramienta que te doy
# (basada en el esquema AnalisisTexto) para generar tu respuesta".
# El resultado `structured_llm` ya no se comporta como un LLM normal,
# sino como un extractor de datos tipado.
#
# 🧠 ¿Qué ocurre por dentro?
#    1. LangChain convierte `AnalisisTexto` en un JSON Schema.
#    2. Ese schema se envía al modelo como la definición de una "herramienta"
#       (function / tool).
#    3. Se le pide al modelo que llame a esa herramienta con los argumentos
#       correctos.
#    4. LangChain intercepta la respuesta, parsea los argumentos y los
#       valida con Pydantic, devolviéndote una **instancia de `AnalisisTexto`**.
#
# 🧠 La variable `structured_llm` conserva la misma interfaz que `llm`
#    (`.invoke`, `.stream`, `.batch`) pero cambia el tipo de retorno:
#    de un `AIMessage` a un objeto Pydantic.
structured_llm = llm.with_structured_output(AnalisisTexto)


# ==============================================================================
# 6. EJECUCIÓN
# ==============================================================================
# 🧠 Este es el texto que vamos a analizar. En un caso real vendría de un
#    usuario, de un fichero, de una BD, etc. Aquí está hardcodeado con fines
#    didácticos.
texto_prueba = "Me encantó la nueva película de acción, tiene muchos efectos especiales y emoción."

# ➕ Añadido: Invocamos con un string normal. El LLM procesará el texto
# y nos devolverá un objeto AnalisisTexto en la variable `resultado`.
#
# 🧠 `.invoke(...)` es el método síncrono estándar de LangChain para enviar
#    una entrada y recibir una salida. También existen `.stream()` (streaming)
#    y `.batch()` (varias entradas a la vez).
# 🧠 Ojo: podríamos pasar una lista de mensajes (`[HumanMessage(...)]`) o un
#    simple string; LangChain lo normaliza internamente.
# 🧠 `resultado` NO es un string. Es una instancia de `AnalisisTexto`
#    (gracias al schema que definimos antes).
resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")


# ==============================================================================
# 7. RESULTADO
# ==============================================================================
# ➕ Añadido: `resultado` es un objeto Pydantic. Podríamos acceder a
# `resultado.resumen` o `resultado.sentimiento` directamente en Python.
# Para imprimirlo como un JSON string, usamos `model_dump_json()`.
#
# 🧠 `model_dump_json()` es el método de Pydantic v2 que serializa el modelo
#    a una string JSON (útil para logs, HTTP, guardar en fichero…).
#    En Pydantic v1 el método equivalente era `.json()`, pero está deprecado.
# 🧠 Alternativas muy útiles:
#      - `resultado.model_dump()` → devuelve un `dict` de Python (no string).
#      - `resultado.resumen`      → accede directamente al campo, ya tipado.
#      - `resultado.sentimiento`  → idem.
print(resultado.model_dump_json())