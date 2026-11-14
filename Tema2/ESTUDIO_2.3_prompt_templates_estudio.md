# 2.3 Plantillas de prompts: `PromptTemplate` y `ChatPromptTemplate`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.3_prompt_templates_estudio.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es una plantilla de prompt y por qué conviene separar la **plantilla** (fija) de los **valores** (variables).
2. Usar `PromptTemplate` (produce un `str`) y `ChatPromptTemplate` (produce una lista de mensajes con roles).
3. Saber cuál elegir y por qué.
4. Conectar una plantilla con un modelo mediante `|`.
5. Aplicar la "cabecera estándar" del curso con `load_dotenv()`.

### Qué necesitas

- Python básico y *f-strings*.
- Conocer `RunnableLambda` y el operador `|` (documento 2.1) y los mensajes `SystemMessage` / `HumanMessage` / `AIMessage`.
- Una clave `OPENAI_API_KEY` **solo** si activas la sección opcional de la Lección 5.

### Mapa del tema

```text
Plantilla con huecos {variable}
        │
        ├── PromptTemplate      ──.format(...)──────────►  str
        │
        └── ChatPromptTemplate  ──.format_messages(...)──►  [SystemMessage, HumanMessage, ...]
                    │
                    └── prompt | llm   (los dos se pueden encadenar con |)
```

### Glosario

| Término | Significado |
|---|---|
| **Plantilla de prompt** | Texto con huecos `{variable}` que se rellenan en cada uso. |
| **`PromptTemplate`** | Plantilla que produce **un `str`**. Familia "clásica". 📘 |
| **`ChatPromptTemplate`** | Plantilla que produce **una lista de mensajes con rol**. Familia "moderna". 📘 |
| **Rol** | Quién habla: `system` (instrucciones permanentes), `human` (usuario), `ai` (asistente). |
| **`.format(**kwargs)`** | Rellena un `PromptTemplate` y devuelve un `str`. |
| **`.format_messages(**kwargs)`** | Rellena un `ChatPromptTemplate` y devuelve una lista de mensajes. |
| **`input_variables`** | Lista de variables que espera la plantilla. |
| **`from_messages`** | Constructor de `ChatPromptTemplate` a partir de una lista de tuplas `(rol, contenido)`. |
| **Few-shot** | Dar al modelo ejemplos de entrada y respuesta dentro del prompt. |
| **Fail-fast / warning** | Detener el programa si falta algo imprescindible / avisar si es opcional. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-core langchain-openai python-dotenv
```

Ejecución:

```bash
python 2.3_prompt_templates_estudio.py
```

Las secciones 3 y 4 funcionan **sin clave de API**: no llaman al modelo.

---

### Lección 1: cabecera estándar y entorno (secciones 1 y 2 del archivo)

**Objetivo.** Conocer la cabecera que el curso usa en todos sus archivos.

```python
import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Aviso: OPENAI_API_KEY no está definida. "
          "Este archivo funciona sin ella, pero la sección 5 no podrá ejecutarse.")
```

| Bloque | Qué hace y por qué |
|---|---|
| `from langchain_core.prompts import PromptTemplate, ChatPromptTemplate` | Importa las dos clases. Son **dos clases distintas, con APIs distintas**; ninguna engloba a la otra. 📘 Viven en `langchain_core`, el paquete base de primitivas independientes del proveedor. 📘 |
| `load_dotenv()` | Carga el `.env`. No falla si no existe. 📘 ⚠️ *(comportamiento de `python-dotenv`, no de LangChain)* |
| `if not os.getenv(...)`: `print(...)` | **Aviso (warning)**, no *fail-fast*: el flujo principal no necesita la clave y un fallo impediría estudiar las plantillas. |

**Regla práctica del archivo.** *Fail-fast* si la clave es imprescindible; *warning* si es opcional. 📘

**Por qué se carga `.env` aunque no se llame al modelo.** Es la convención del curso: consistencia entre archivos y preparación por si se añade `llm.invoke(...)`. 📘 No es una obligación.

---

### Lección 2: `PromptTemplate`, la familia de cadenas de texto (sección 3 del archivo)

**Objetivo.** Crear una plantilla con un hueco y rellenarla.

```python
template = "Eres un experto en marketing. Sugiere un eslogan creativo para un producto {producto}"

prompt = PromptTemplate(
    template=template,
    input_variables=["producto"],
)

prompt_lleno = prompt.format(producto="café orgánico")

print(prompt_lleno)
```

| Paso | Qué hace |
|---|---|
| `template = "... {producto}"` | Texto con un hueco entre llaves. |
| `PromptTemplate(template=..., input_variables=["producto"])` | Crea la plantilla y declara explícitamente sus variables. |
| `prompt.format(producto="café orgánico")` | Sustituye el hueco y devuelve un `str`. |

**Resultado.** El texto impreso es una sola línea:

```text
Eres un experto en marketing. Sugiere un eslogan creativo para un producto café orgánico
```

(El archivo lo muestra partido en dos líneas en un comentario solo por espacio.)

**Por qué no usar directamente un f-string.** El archivo señala tres inconvenientes: la plantilla queda atada al valor, no se pueden validar variables y no se integra con `|`. 📘

**Por qué declarar `input_variables`.** Validación temprana y documentación implícita. En versiones modernas se puede omitir porque LangChain las infiere, pero declararlas es buena práctica. 📘 La referencia oficial de `PromptTemplate` incluye comprobaciones de que la plantilla y las variables sean consistentes.

**Comprobaciones de depuración del archivo:**

```python
print(f"\n[DEBUG] Tipo del resultado: {type(prompt_lleno).__name__}")
print(f"[DEBUG] Variables esperadas: {prompt.input_variables}")
```

Según el archivo, el tipo es `str` y las variables esperadas son `['producto']`.

**Comportamientos que señala el archivo** 📘 ⚠️ *(no contrastados)*:

- Si olvidas una variable, `prompt.format()` lanza `KeyError: 'producto'`: detectas el error **antes** de gastar tokens.
- Si pasas una variable extra, se ignora.
- Existe `.format_prompt()`, que devuelve un `StringPromptValue`; para casi todos los casos basta `.format()`.

**Puntos clave.** `PromptTemplate` produce un `str`; se usa con `.format(...)`. Según el archivo es la familia clásica, pensada para modelos de tipo *completion*, hoy considerada *legacy*, aunque sigue siendo la forma más simple de entender qué es una plantilla. 📘 ⚠️ *(no contrastado)*

---

### Lección 3: `ChatPromptTemplate`, la familia de mensajes con rol (sección 4 del archivo)

**Objetivo.** Crear una plantilla con un mensaje de sistema y otro de usuario.

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso."),
    ("human", "{texto}"),
])

mensajes = chat_prompt.format_messages(texto="Hola mundo, ¿cómo estás?")

for m in mensajes:
    print(f"  {type(m).__name__:15s} → {m.content}")
```

| Elemento | Qué hace |
|---|---|
| `ChatPromptTemplate.from_messages([...])` | Constructor recomendado. Recibe una lista de tuplas `(rol, contenido)`. |
| `("system", ...)` | Instrucciones permanentes: personalidad, idioma, formato. Va primero. 📘 |
| `("human", "{texto}")` | Entrada del usuario; aquí está la parte variable. |
| `format_messages(texto=...)` | Devuelve una **lista** de mensajes con el hueco ya relleno. |
| `type(m).__name__` y `m.content` | Muestran el tipo de cada mensaje y su texto. |

**Salida esperada** (según el archivo):

```text
  SystemMessage   → Eres un traductor del español al inglés muy preciso.
  HumanMessage    → Hola mundo, ¿cómo estás?
```

Y los datos de depuración: el resultado es una `list` de longitud `2`. 📘

**Por qué tuplas y no strings sueltos.** Hace falta el rol: un texto suelto no indica si es `system` o `human`. 📘 La documentación oficial usa este mismo formato `(rol, contenido)` en sus ejemplos, con los roles `system`, `human`, `ai` y `placeholder`.

**Por qué `from_messages` y no el constructor directo.** El archivo lo presenta como el constructor recomendado: acepta tuplas o instancias de mensajes (`SystemMessage`, `HumanMessage`…). Las tuplas son más legibles en prompts cortos. 📘

**Más detalles del archivo** 📘 ⚠️ *(no contrastados)*:

- Roles posibles que cita: `system`, `human`, `ai` y `tool`.
- `BaseMessage` es la clase base de todos los mensajes; cada uno tiene `.content` y `.additional_kwargs`.
- Algunos modelos no soportan `system`; como alternativa se puede usar `("human", "Eres un traductor...")`, que es menos idiomático.
- Un turno `ai` permite hacer *few-shot*: `("human", "Hola"), ("ai", "Hello"), ("human", "{texto}")`.
- `ChatPromptTemplate.from_template(t)` es un atajo para un solo `HumanMessage`.

**Contraste de tipos** (la idea central de las lecciones 2 y 3):

| Método | Devuelve |
|---|---|
| `PromptTemplate.format(...)` | `str` |
| `ChatPromptTemplate.format_messages(...)` | `list[BaseMessage]` |

---

### Lección 4: comparativa y elección (sección 5 del archivo)

| `PromptTemplate` | `ChatPromptTemplate` |
|---|---|
| `.format()` → `str` | `.format_messages()` → lista de mensajes |
| Sin roles | Con roles (`system` / `human` / `ai`) |
| Modelos de tipo *completion* | Modelos de chat |
| Legacy, en desuso | Estándar, recomendado |
| Simple y directo | Expresivo; escala a diálogos |

*(Contenido de la tabla del archivo 📘.)*

**Regla práctica del archivo** 📘:

| Situación | Elección |
|---|---|
| Usas un modelo de chat (GPT-4o, Claude, Gemini…) | `ChatPromptTemplate` |
| Mantienes código con modelos *completion* antiguos | `PromptTemplate` |
| Prototipo rápido en un cuaderno | `PromptTemplate` (vale) |
| Pipeline en producción | `ChatPromptTemplate` |

**Más del archivo** 📘 ⚠️ *(no contrastados)*:

- Ambos son Runnables y se pueden encadenar: `chain = chat_prompt | llm | parser`.
- Se puede pasar de `PromptTemplate` a `ChatPromptTemplate` con `.from_template(template)`, pero no al revés directamente (se perderían los roles).

---

### Lección 5: conectar con el modelo (sección 6 del archivo, **no se ejecuta por defecto**)

El bloque está dentro de `if False:` para no gastar tokens al estudiar. Cambia a `True` para ejecutarlo; necesita `OPENAI_API_KEY`.

```python
if False:  # Cambia a True para probar (necesita .env con OPENAI_API_KEY)
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = chat_prompt | llm
    respuesta = chain.invoke({"texto": "Hola mundo, ¿cómo estás?"})
    print("\n[Sección 6] Respuesta del traductor:", respuesta.content)

    cadena_simple = prompt | llm
    respuesta2 = cadena_simple.invoke({"producto": "café orgánico"})
    print("[Sección 6] Eslogan sugerido:", respuesta2.content)
```

**Flujo de la opción A:**

```text
{"texto": "..."}  →  chat_prompt  →  [SystemMessage, HumanMessage]  →  llm  →  AIMessage  →  .content
```

| Opción | Cadena | Entrada de `.invoke` |
|---|---|---|
| A (moderna) | `chat_prompt \| llm` | `{"texto": "..."}` |
| B (compatibilidad) | `prompt \| llm` | `{"producto": "..."}` |

**Detalles:**

- El `import` dentro del `if` es local: solo se ejecuta si activas el bloque. 📘
- `ChatOpenAI` lee `OPENAI_API_KEY` del entorno automáticamente (la documentación oficial confirma esa variable).
- El archivo anota como respuesta esperada de A `"Hello world, how are you?"`; es texto generado por el modelo y puede variar, por lo que no se trata como salida garantizada.
- Un `ChatOpenAI` acepta también `str`, que envuelve como `HumanMessage`. 📘 La documentación oficial indica que pasar una cadena es un atajo de un único `HumanMessage`.

**Sobre la entrada de la cadena.** El archivo dice que `chain.invoke("...")` con un `str` daría error o se interpretaría como entrada suelta "según el template". 🔀 *(matiz)* La referencia oficial de `ChatPromptTemplate` indica que, si la plantilla tiene **una sola variable**, se puede pasar un argumento que no sea diccionario y se inyecta en esa variable (`template.invoke("Hello, there!")` equivale a `{"user_input": "Hello, there!"}`). Con varias variables sí hace falta un diccionario.

---

### Ejercicios propuestos en el archivo

| | Ejercicio |
|---|---|
| A | Modifica el template de marketing para que pida también "tono" (formal/informal). Pista: añade un hueco `{tono}` y pásalo en `.format()`. |
| B | Crea un `ChatPromptTemplate` con tres mensajes: `system` → "Eres un chef experto.", `human` → "Dame una receta con {ingrediente}.", `ai` → "Aquí tienes una receta con {ingrediente}: ...". Formatea y observa los tipos de mensaje. |
| C | ¿Qué pasa si llamas a `.format()` sin el argumento `producto`? Lee el error. ¿Por qué es útil? |
| D | Convierte el `PromptTemplate` de marketing en un `ChatPromptTemplate` con `.from_template(template)`. ¿Cuántos mensajes y de qué tipo produce? Compara con el original. |
| E | Añade un mensaje `ai` al `chat_prompt` del traductor con un ejemplo *few-shot*: `("human", "Buenos días"), ("ai", "Good morning")`. ¿Cambia el comportamiento del modelo? |
| F | Comenta `load_dotenv()` y activa la sección 6. ¿Qué error obtienes al invocar la cadena? ¿Por qué es útil la cabecera estándar? |
| G | (Avanzado) Activa la sección 6 y prueba ambos pipelines (A y B). Mide la latencia con `time.perf_counter()` y compara las salidas. |

---

## Guía práctica

### ¿Cómo creo y relleno una plantilla de texto?

```python
prompt = PromptTemplate(template=template, input_variables=["producto"])
texto = prompt.format(producto="café orgánico")
```

### ¿Cómo creo una plantilla con roles?

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso."),
    ("human", "{texto}"),
])
mensajes = chat_prompt.format_messages(texto="Hola mundo, ¿cómo estás?")
```

### ¿Cómo compruebo la plantilla sin llamar al modelo?

Usa `.format()` o `.format_messages()` e imprime el resultado. El archivo lo presenta como una ventaja: las plantillas se pueden testear sin gastar tokens. 📘

### ¿Cómo uso la plantilla con un modelo?

```python
chain = chat_prompt | llm
respuesta = chain.invoke({"texto": "Hola mundo, ¿cómo estás?"})
```

### ¿Cómo añado un ejemplo (few-shot) a una plantilla de chat?

Añade turnos `human` y `ai` antes de la entrada final: 📘

```text
("human", "Hola"), ("ai", "Hello"), ("human", "{texto}")
```

### ¿Cómo evito que falle por falta de clave cuando esta es opcional?

Usa un aviso en lugar de detener el programa (Lección 1).

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `KeyError: 'producto'` al llamar a `.format()` | Falta una variable | Pasar todas las variables de la plantilla 📘 |
| Aviso "OPENAI_API_KEY no está definida" | La clave no está en el entorno | Solo es un problema si activas la sección 6; definirla en `.env` |
| Error al invocar la cadena con la sección 6 activa y sin `.env` | No se cargó la clave | Mantener `load_dotenv()` en la cabecera (ejercicio F) |
| El modelo no respeta el mensaje `system` | Algunos modelos no lo soportan | Usar `("human", "...")` como alternativa 📘 |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `import os` | Biblioteca estándar | Leer variables de entorno |
| `from dotenv import load_dotenv` | `python-dotenv` | Cargar `.env` |
| `from langchain_core.prompts import PromptTemplate, ChatPromptTemplate` | `langchain-core` | Plantillas |
| `from langchain_openai import ChatOpenAI` | `langchain-openai` | Solo en la sección 6 (opcional) |

### `PromptTemplate`

| Elemento | Descripción | Confirmado en documentación oficial |
|---|---|---|
| `template` | Texto con huecos `{variable}` | Sí ("The prompt template.") |
| `input_variables` | Variables esperadas | Aparece como atributo heredado; descripción cortada |
| `template_format` | Formato de la plantilla | Sí: `'f-string'`, `'mustache'`, `'jinja2'`; se recomienda `f-string` frente a `jinja2` por seguridad |
| `.format(**kwargs)` | Devuelve `str` | Sí ("Format the prompt with the inputs") |
| `.from_template(...)` | Crea una plantilla desde un texto | Sí ("Load a prompt template from a template") |
| `.format_prompt(...)` | Devuelve `StringPromptValue` | ⚠️ No contrastado |

### `ChatPromptTemplate`

| Elemento | Descripción | Confirmado en documentación oficial |
|---|---|---|
| `.from_messages([...])` | Lista de tuplas `(rol, contenido)` o mensajes | Los ejemplos oficiales usan tuplas; la página no describe el método |
| Roles en tuplas | `system`, `human`, `ai`, `tool` (archivo) | Los ejemplos oficiales usan `system`, `human`, `ai` y `placeholder` |
| `.format_messages(**kwargs)` | Devuelve lista de mensajes | Listado, sin descripción |
| `.from_template(t)` | Atajo para un solo `HumanMessage` | Listado, sin descripción ⚠️ |
| `.invoke({...})` | Devuelve un `ChatPromptValue` con los mensajes formateados | Sí |
| `.input_variables` | Variables esperadas | Atributo heredado |

### Tipos de mensaje

| Clase | Rol | Notas |
|---|---|---|
| `SystemMessage` | `system` | Instrucciones permanentes |
| `HumanMessage` | `human` / `user` | Entrada del usuario |
| `AIMessage` | `ai` / `assistant` | Respuesta del modelo |
| `ToolMessage` | `tool` | Salida de herramientas; el archivo lo cita sin usarlo |

La documentación oficial describe `SystemMessage`, `HumanMessage`, `AIMessage` y `ToolMessage`; el atributo `content` admite cadenas y listas. El archivo cita además `FunctionMessage` ⚠️ *(no contrastado)*.

### Otras plantillas citadas en el archivo (no se usan)

| Clase | Para qué (según el archivo) |
|---|---|
| `FewShotPromptTemplate` | Prompts con ejemplos |
| `PipelinePromptTemplate` | Prompts compuestos |
| `MessagesPlaceholder` | Insertar historial de chat |

### Comparativa de métodos

| | `PromptTemplate` | `ChatPromptTemplate` |
|---|---|---|
| Método de relleno | `.format()` | `.format_messages()` |
| Salida | `str` | `list[BaseMessage]` |
| Entrada en una cadena | `dict` con variables | `dict` con variables |

### Patrones reutilizables (resumen del archivo)

| # | Patrón | Idea |
|---|---|---|
| 1 | Cabecera estándar con `dotenv` | `load_dotenv()` + `os.getenv`; warning si es opcional, fail-fast si es imprescindible |
| 2 | Plantilla + variables | Una plantilla, muchos usos |
| 3 | System + human | Esqueleto de un prompt conversacional |
| 4 | Declarar `input_variables` | Documenta y valida |
| 5 | Fail-fast con `format()` | Detectar variables faltantes antes de llamar a la API |
| 6 | Las plantillas son Runnables | Se encadenan con `\|` |
| 7 | Elegir según el modelo | Chat → `ChatPromptTemplate` |

---

## Explicación

### ¿Por qué usar plantillas en lugar de f-strings?

Porque separan la forma (fija) de los datos (variables). El archivo enumera ventajas: reutilización, posibilidad de testear sin llamar al modelo, composición con `|` y trazabilidad (LangSmith ve el prompt exacto). 📘 ⚠️ *(trazabilidad con LangSmith no contrastada)*

### ¿Por qué hay dos familias?

Según el archivo, los modelos antiguos de tipo *completion* esperaban un texto plano, mientras que los modelos de chat esperan una lista de mensajes con roles; cada familia refleja una de esas estructuras. 📘 ⚠️ *(no contrastado)*

### ¿Por qué el mensaje `system` va primero?

Establece el contexto persistente (personalidad, idioma, formato) y no varía entre llamadas del mismo diálogo. 📘

### ¿Por qué `format()` lanza error si falta una variable?

Es un *fail-fast* útil: el error aparece antes de enviar nada al modelo. 📘

### ¿Por qué se puede encadenar una plantilla con `|`?

El archivo afirma que ambas heredan de `BasePromptTemplate`, que es un `Runnable`. 📘 ⚠️ *(no contrastado: la referencia oficial de `PromptTemplate` muestra como clase base `StringPromptTemplate` y no describe la relación con `Runnable`)* Lo que sí confirma la documentación oficial es que una plantilla de chat se ejecuta con `.invoke({...})` y devuelve un `ChatPromptValue`.

### ¿Cuándo usar cada enfoque?

| Situación | Enfoque |
|---|---|
| Modelo de chat | `ChatPromptTemplate` |
| Código antiguo con modelos de texto | `PromptTemplate` |
| Prueba rápida de una plantilla | `PromptTemplate` |
| Producción | `ChatPromptTemplate` |

### Lo que este documento no cubre

- `FewShotPromptTemplate`, `PipelinePromptTemplate` y `MessagesPlaceholder` (solo se citan).
- La reescritura de los análisis de sentimientos con `ChatPromptTemplate` que el archivo sugiere como siguiente paso.
- LangSmith.
- La salida real de la sección 6: no se ejecuta por defecto y el archivo no incluye una salida real.

### Observaciones sobre el archivo

- El archivo afirma que todos los modelos actuales (GPT-4o, Claude, Gemini) son modelos de chat y que `PromptTemplate` está en desuso. No se ha podido contrastar y se mantiene como material del curso.
- 🔀 El archivo indica que pasar un `str` a una cadena de plantilla "daría error o se interpretaría como entrada suelta según el template"; la documentación oficial precisa que con una sola variable se inyecta directamente.
- El archivo afirma que una variable extra en `.format()` se ignora sin error; no se pudo contrastar.

---

## Autoevaluación

1. ¿Qué diferencia hay entre `PromptTemplate` y `ChatPromptTemplate`?
2. ¿Qué devuelve `PromptTemplate.format()`? ¿Y `ChatPromptTemplate.format_messages()`?
3. ¿Por qué se usan tuplas `(rol, contenido)` en `from_messages`?
4. ¿Qué ocurre, según el archivo, si olvidas una variable en `.format()`?
5. ¿Por qué el archivo usa un *warning* y no un *fail-fast* en su cabecera?
6. ¿Qué entrada espera `chat_prompt | llm` cuando la plantilla tiene la variable `{texto}`?
7. ¿Por qué el bloque de la sección 6 no se ejecuta por defecto?
8. ¿Qué mensajes produce el `chat_prompt` del traductor al formatearlo?
9. ¿Qué es *few-shot* y cómo se añade a una plantilla de chat?
10. ¿Qué indica la documentación oficial sobre pasar un `str` a un modelo de chat?

### Respuestas

1. `PromptTemplate` produce un `str`; `ChatPromptTemplate` produce una lista de mensajes con rol.
2. Un `str`; una lista de mensajes (aquí `SystemMessage` y `HumanMessage`).
3. Porque hace falta indicar el rol de cada mensaje; un texto suelto no dice si es `system` o `human`.
4. Se lanza `KeyError`; así detectas el fallo antes de gastar tokens.
5. Porque el flujo principal no necesita la clave; un *fail-fast* impediría estudiar las plantillas.
6. Un diccionario con la variable: `{"texto": "..."}` (con una sola variable, la documentación oficial admite también pasar el valor directamente).
7. Está dentro de `if False:` para no gastar tokens mientras se estudia.
8. Un `SystemMessage` con la instrucción del traductor y un `HumanMessage` con el texto relleno.
9. Dar ejemplos de entrada y respuesta dentro del prompt; se añaden turnos `human` y `ai` antes de la entrada final.
10. Que pasar una cadena es un atajo de un único `HumanMessage`.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Plantilla | Texto fijo con huecos `{variable}` |
| `PromptTemplate` | Produce un `str` con `.format()` |
| `ChatPromptTemplate` | Produce mensajes con rol con `.format_messages()` |
| `from_messages` | Constructor a partir de tuplas `(rol, contenido)` |
| `system` / `human` / `ai` | Roles de los mensajes |
| `prompt \| llm` | Encadena plantilla y modelo |
| Cabecera estándar | `load_dotenv()` + comprobación de la clave |
| Warning vs fail-fast | Aviso si es opcional; error si es imprescindible |

---

## Fuentes consultadas

- Referencia de `ChatPromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate>
- Referencia de `PromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate>
- Documentación oficial de mensajes: <https://docs.langchain.com/oss/python/langchain/messages>
- Documentación oficial de `ChatOpenAI` (variable `OPENAI_API_KEY`): <https://python.langchain.com/docs/integrations/chat/openai/> (redirige a <https://docs.langchain.com/oss/python/integrations/chat/openai>)
- Material del tema: `2.3_prompt_templates_estudio.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
