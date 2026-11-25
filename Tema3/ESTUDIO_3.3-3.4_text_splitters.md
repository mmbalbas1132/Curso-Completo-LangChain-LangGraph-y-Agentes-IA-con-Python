# 3.3-3.4 Documentos largos: ventana de contexto y Text Splitters (partes 1 y 2)

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.3_text_splitters_parte1.py` y `3.4_text_splitters_parte2.py` (versiones con comentarios didácticos; su código es idéntico al de sus `_limpio.py`, ✅ comprobado) y de la documentación oficial de LangChain. **Cambio sobre los originales:** los cuatro scripts incluyen ahora `from dotenv import load_dotenv` y `load_dotenv()`, añadidos por el alumno para leer la clave del `.env`; el resto del código es idéntico al original. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = lo observado o la documentación difiere del archivo; ✅ *(ejecutado)* = comportamiento observado con `langchain-community` 0.4.2, `langchain-core` 1.7.7, `langchain-openai` 1.7.4, `langchain-text-splitters` 1.1.3 y `pypdf` 7.1.0. El script de la parte 2 se probó con un modelo simulado en Python 3.13 (necesita Python ≥ 3.12, ver lección 9) y con `langchain-core` 1.8.4 y `langchain-text-splitters` 1.2.0; el troceado se midió con las versiones anteriores y dio los mismos resultados.
>
> **Alcance.** Este documento integra las dos partes: primero lo común (cargar el PDF y llamar al modelo), después lo que añade cada una. Yo no ejecuté nada contra la API real de OpenAI (no hay clave): lo que depende del modelo lo comprobé con un modelo simulado. Los errores de la API (401 y 400) de la parte 1 **los observó el alumno** al ejecutar el script con su clave y se citan tal cual. La parte 2 **no** la he ejecutado contra la API real (el alumno indica que ahora funciona con su clave y su `.env`, sin aportar la salida).

---

## Antes de empezar

### Qué vas a aprender

1. Cargar un PDF largo con `PyPDFLoader` y medir lo que has cargado.
2. Unir las páginas en un único texto y enviarlo a un LLM con `ChatOpenAI.invoke`.
3. Por qué enviar un libro entero en un único prompt es un problema (ventana de contexto).
4. Qué devuelve `invoke` (`AIMessage`) y cómo leer el texto de la respuesta.
5. Por qué existen los *text splitters* y cómo se usa `RecursiveCharacterTextSplitter` con `chunk_size` y `chunk_overlap` (parte 2).
6. Resumir un documento largo por fragmentos y combinar los resúmenes.
7. Por qué el script de la parte 2, tal como está, no resume el libro entero.

### Qué necesitas

- Haber visto el documento 3.1 (`PyPDFLoader`, `Document`, metadatos).
- `pip install -U langchain-community pypdf langchain-openai langchain-text-splitters python-dotenv`.
- Una clave de OpenAI en `OPENAI_API_KEY`, guardada en un archivo `.env` (los scripts la leen con `load_dotenv()`), solo para ejecutar los scripts completos. Para entender el documento no hace falta.
- **Python 3.12 o superior** para la parte 2 (lección 9).
- El archivo `quijote.pdf` (471 páginas).

### Mapa del tema

```text
quijote.pdf (471 páginas)
     │  PyPDFLoader(ruta).load()
     ▼
[Document × 471]  ──►  for page: full_text += page.page_content + "\n"
                              │
                              ▼
                   full_text  (1.323.576 caracteres)
                              │  f"Haz un resumen…: {full_text}"
                              ▼
              ChatOpenAI("gpt-4o-mini").invoke(prompt)   ← prompt ≈ 1,32 M caracteres
                              │
                              ▼
                  AIMessage  ──► print(response)

PARTE 2:  [Document × 471] ──► RecursiveCharacterTextSplitter(10000, 200).split_documents
                              ──► [Document × 471]  (¡un chunk por página!)
                              ──► for chunk in chunks (solo los 11 primeros): llm.invoke(resumen)
                              ──► summaries (11 textos) ──► llm.invoke(combinar) ──► final_summary
```

### Glosario

| Término | Significado |
|---|---|
| **Document loader** | Objeto que lee una fuente (PDF, web…) y devuelve objetos `Document` (ver 3.1). |
| **`Document`** | Objeto con `page_content` (texto) y `metadata` (diccionario). |
| **LLM / modelo de chat** | Modelo de lenguaje al que se envía texto y que responde con texto. Aquí, `ChatOpenAI`. |
| **Prompt** | Texto que se envía al modelo (instrucción + datos). |
| **`invoke`** | Método que ejecuta el modelo con una entrada y devuelve la respuesta. |
| **`AIMessage`** | Objeto de mensaje que devuelve el modelo; el texto está en `.content`. |
| **Token** | Unidad en la que el modelo cuenta el texto (no es igual a un carácter ni a una palabra). |
| **Ventana de contexto** | Máximo de tokens que un modelo admite en una petición. ✅ El error de la API indica 128.000 para `gpt-4o-mini`; la página oficial de `ChatOpenAI` consultada no da cifras. |
| **Text splitter** | Utilidad que parte un texto largo en fragmentos (*chunks*). Lo usa la parte 2. |
| **Chunk** | Cada fragmento que produce un splitter (aquí, un `Document`). |
| **`RecursiveCharacterTextSplitter`** | Splitter de `langchain_text_splitters`; la documentación oficial lo recomienda por defecto para texto genérico. |
| **`chunk_size`** | Tamaño máximo de un chunk, medido por la `length_function` del splitter (por defecto `len`: caracteres ✅). |
| **`chunk_overlap`** | Solape objetivo entre chunks consecutivos, para no perder información en los cortes. |
| **`split_documents`** | Método que trocea una lista de `Document` y devuelve otra lista de `Document`. |

---

## Tutorial

### Preparación (común a las dos partes)

Coloca `quijote.pdf` junto al script e instala:

```bash
pip install -U langchain-community pypdf langchain-openai langchain-text-splitters python-dotenv
```

**La clave de API.** Necesitas `OPENAI_API_KEY`. Guárdala en un archivo `.env` (formato `OPENAI_API_KEY=tu_clave`, sin comillas ni espacios alrededor del `=`) y no la escribas en el código. Los cuatro scripts empiezan con:

```python
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv()` busca el `.env` empezando por la carpeta del script y subiendo por las carpetas superiores, y carga sus variables en el entorno (✅ comprobado con un `.env` de prueba). Dos precauciones:

- Por defecto **no sobrescribe** una variable que ya exista en el entorno ✅: si el sistema o el IDE ya definen `OPENAI_API_KEY` (por ejemplo, una clave caducada), gana esa y la del `.env` se ignora.
- ✅ Sin clave, `ChatOpenAI(...)` falla al crearse con un `OpenAIError` ("Missing credentials…").

---

### PARTE 1 (3.3): el problema

### Lección 1: importaciones

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
```

`PyPDFLoader` viene de `langchain_community` y `ChatOpenAI` de `langchain_openai`, como indican las páginas oficiales. `load_dotenv` (paquete `python-dotenv`) lee el `.env` y carga la clave en el entorno **antes** de crear el modelo (ver *Preparación*). Es una línea añadida por el alumno; el script original no la tenía. Sobre el aviso de retirada de `langchain-community`, ver el documento 3.1.

### Lección 2: cargar el PDF

```python
# 1. Cargar el documento PDF
loader = PyPDFLoader("C:\\Users\\santiago\\curso_langchain\\Tema 3\\quijote.pdf")
pages = loader.load()
```

`PyPDFLoader(ruta)` crea el cargador y `.load()` lee el PDF y devuelve una lista con un `Document` por página.

✅ Con `quijote.pdf`: **471** `Document`, ninguno vacío, de 29 a 4.939 caracteres (mediana 3.144). Las claves de `metadata` son: `producer`, `creator`, `creationdate`, `title`, `author`, `subject`, `moddate`, `total_pages`, `page`, `page_label`, `source`.

🔀 **La ruta del script no funciona en este equipo.** Es una ruta absoluta de Windows con el nombre de otro usuario. ✅ Ejecutado tal cual, `PyPDFLoader` lanzó `ValueError: File path … is not a valid file or url`. Se respeta el código; en la guía práctica verás la forma portable.

### Lección 3: unir las páginas

```python
# 2. Combinar todas las páginas en un texto único
full_text = ""
for page in pages:
    full_text += page.page_content + "\n"
```

Parte de un string vacío y concatena el texto de cada página más un salto de línea. ✅ Resultado: **1.323.576 caracteres** (unas 240.000 palabras separando por espacios).

### Lección 4: enviar el texto al modelo

```python
# 3. Pasar el texto al LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente documento: {full_text}")
```

| Código | Qué hace |
|---|---|
| `ChatOpenAI(model="gpt-4o-mini", temperature=0.2)` | Crea el modelo. `model` elige el modelo; `temperature` regula la variabilidad (la página oficial consultada lo lista como parámetro opcional sin detallar su rango). |
| `f"Haz un resumen…: {full_text}"` | Construye un único prompt: instrucción + libro entero. ✅ Mide 1.323.646 caracteres (70 de instrucción + 1.323.576 de texto). |
| `llm.invoke(prompt)` | Envía el prompt y devuelve un `AIMessage`. |

#### El problema de tamaño

Un modelo solo admite un máximo de **tokens** por petición. ✅ Al ejecutar el script con una clave válida, la API respondió:

```text
openai.BadRequestError: Error code: 400 - This model's maximum context length is 128000 tokens.
However, your messages resulted in 362243 tokens. Please reduce the length of the messages.
```

LangChain relanza ese error como `OpenAIInvalidRequestError`. Es decir: el libro son **362.243 tokens** y `gpt-4o-mini` admite **128.000**. El script **termina con error en `llm.invoke(...)`** y nunca llega a imprimir un resumen. Con el prompt de 1.323.646 caracteres, salen ≈ 3,65 caracteres por token (cálculo propio sobre esos dos datos reales).

Esta es la motivación de los *text splitters*: trocear el texto antes de enviarlo (ver *Explicación*).

### Lección 5: mostrar la respuesta

```python
print(response)
```

✅ Esta línea **no llega a ejecutarse** con la API real, porque falla antes (lección 4). Con un modelo simulado, `print(response)` imprime el objeto `AIMessage` entero:

```text
content='RESUMEN FALSO' additional_kwargs={} response_metadata={} id='lc_run--…' tool_calls=[] …
```

El texto del resumen está en `response.content`. La página oficial de `ChatOpenAI` muestra `ai_msg.text` y, en otros ejemplos, `response.content`.

---

### PARTE 2 (3.4): trocear y resumir por fragmentos

Lo común (importar `PyPDFLoader` y `ChatOpenAI`, cargar el PDF con `.load()`, crear `ChatOpenAI` e `invoke`) funciona como en la parte 1. Esta parte añade un splitter, un bucle que resume cada chunk y una llamada final que combina los resúmenes. El script completo tiene 32 líneas.

#### Lección 6: la importación nueva

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Viene del paquete independiente `langchain-text-splitters`, como indica la página oficial.

#### Lección 7: dividir en chunks

```python
# Dividir el texto en chunks mas pequeños
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)
```

| Código | Qué hace |
|---|---|
| `chunk_size=10000` | Máximo de 10.000 caracteres por chunk (✅ el splitter mide con `len` por defecto). |
| `chunk_overlap=200` | Solape objetivo de 200 caracteres entre chunks consecutivos. |
| `split_documents(pages)` | Trocea **cada** `Document` por separado y copia sus metadatos (✅ se conservan `page`, `title`…). |

Por defecto el splitter intenta cortar por `"\n\n"`, luego `"\n"`, luego `" "` y por último `""` (✅ código instalado; la página oficial indica los mismos separadores).

🔀 **Aquí el splitter no parte nada.** ✅ Cada página mide como máximo 4.939 caracteres, menos que `chunk_size=10000`. Resultado: **471 páginas → 471 chunks**, uno por página, de 29 a 4.939 caracteres, ninguno de más de 10.000. El comentario del archivo dice "dividir el texto en chunks más pequeños", pero con estos valores el resultado equivale a las páginas originales.

✅ Para obtener chunks de ~10.000 caracteres habría que trocear el texto ya unido (como `full_text` de la parte 1) con `split_text`: salen **136 chunks** (de 3.296 a 9.999 caracteres). Ver *Guía práctica*.

#### Lección 8: resumir cada chunk

```python
# 3. Pasar el texto al LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
summaries = []

i = 0
for chunk in chunks:
    if i > 10:
        break
    response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(response.content)
    i += 1

print(summaries)
```

| Código | Qué hace |
|---|---|
| `summaries = []` | Lista donde se guardan los resúmenes parciales. |
| `i = 0` … `i += 1` | Contador manual de chunks procesados. |
| `if i > 10: break` | Sale del bucle cuando `i` vale 11: se procesan los chunks con `i` de 0 a 10, o sea **11 chunks**. |
| `chunk.page_content` | Texto del chunk. |
| `response.content` | Texto de la respuesta (el `AIMessage` tiene más campos). |
| `print(summaries)` | Imprime la lista de 11 strings. |

🔀 **Solo se resumen los 11 primeros chunks.** ✅ Como cada chunk es una página, son las páginas 0–10 del PDF: 22.707 caracteres de 1.323.105, un **1,7 %** del libro. El resto no se procesa. Cada petición lleva entre ~180 y ~2.700 caracteres (medido con modelo simulado), muy por debajo del límite de contexto; por eso aquí no se repite el error de la parte 1.

#### Lección 9: el resumen final

```python
final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {" ".join(summaries)}")
print(final_summary.content)
```

`" ".join(summaries)` une los resúmenes parciales con un espacio. Se envían al modelo con la orden de combinarlos y se imprime solo el texto (`.content`).

⚠️ **Requiere Python 3.12 o superior.** La expresión usa comillas dobles dentro de un f-string delimitado también por comillas dobles. ✅ En Python 3.10 da `SyntaxError: f-string: expecting '}'`; en 3.13 compila. (Que el cambio de sintaxis llegó con Python 3.12 es lo observado aquí; no se contrastó con la documentación de Python.)

🔀 Como solo hay 11 resúmenes, el "resumen coherente y completo" describe como mucho las 11 primeras páginas, no el libro.

#### Lección 10: qué verás al ejecutarlo

✅ Con un modelo simulado: **12 llamadas** al LLM (11 parciales + 1 final). `print(summaries)` muestra una lista de 11 textos y `print(final_summary.content)` el texto final. Con la API real no se ha ejecutado; coste: 12 llamadas de pago.

---

## Guía práctica

### ¿Cómo ejecuto el script en mi equipo?

1. Copia `quijote.pdf` junto al script.
2. Sustituye la ruta por `"quijote.pdf"` (o por una ruta tuya). El archivo original no lo hace.
3. Pon `OPENAI_API_KEY` en tu `.env` (ver *Preparación*).
4. Con clave válida, el script falla con el error 400 de la lección 4: es lo esperado.

### ¿Cómo hago la ruta portable?

```python
from pathlib import Path
ruta = Path(__file__).parent / "quijote.pdf"
loader = PyPDFLoader(str(ruta))
```

⚠️ Forma habitual de Python, no tomada del archivo ni de LangChain; no se ejecutó con LangChain, pero `PyPDFLoader("quijote.pdf")` ✅ funcionó con ruta relativa.

### ¿Cómo imprimo solo el resumen?

```python
print(response.content)
```

### ¿Cómo compruebo cuánto texto voy a enviar?

```python
print(len(pages), len(full_text))
```

✅ Da `471` y `1323576` con este PDF.

### ¿Cómo troceo el libro entero en chunks de ~10.000 caracteres?

```python
chunks = text_splitter.split_text(full_text)   # lista de strings (no de Document)
```

✅ Con `full_text` construido como en la parte 1: **136 chunks**. Como `split_text` devuelve strings, en el bucle se usaría `chunk` en vez de `chunk.page_content` ⚠️ (adaptación no ejecutada con el script completo). `create_documents([full_text])` devuelve 136 `Document`, pero sin metadatos de página ✅ (`metadata` vacío).

### ¿Cómo resumo todos los chunks y no solo 11?

Quita el contador y el `break` (o pon un límite mayor). ⚠️ Cada chunk es una llamada de pago, y con muchos resúmenes el prompt final podría volver a ser demasiado largo; no está comprobado. Una alternativa más clara que el contador manual es `chunks[:11]` o `enumerate`.

### ¿Cómo evito el `SyntaxError` del f-string?

Actualiza a Python 3.12 o superior, o calcula la unión antes:

```python
unido = " ".join(summaries)
final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {unido}")
```

⚠️ Reescritura habitual; no es el código del archivo.

### Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `ValueError: File path … is not a valid file or url` | La ruta absoluta no existe en tu equipo ✅ | Usa una ruta válida (ver arriba). |
| `OpenAIError: Missing credentials…` | Falta `OPENAI_API_KEY` ✅ | Define la variable de entorno. |
| `OpenAIAuthenticationError: Error code: 401 - Your API key has expired. Create a new API key to continue.` (observado por el alumno al ejecutar con clave) | La clave que llega al proceso está caducada. Si la clave de tu `.env` es nueva y aun así falla, puede haber otra `OPENAI_API_KEY` ya definida en el entorno de la terminal o del IDE (por defecto `load_dotenv()` no sobrescribe variables ya existentes) | Crea una clave nueva en la plataforma de OpenAI, actualiza el `.env` y revisa que no quede otra definida en el entorno |
| `OpenAIError: Missing credentials…` aunque la clave esté en `.env` | Falta `load_dotenv()` antes de crear `ChatOpenAI`, el `.env` no está en la carpeta del script ni en una superior, o el nombre de la variable no es `OPENAI_API_KEY` | Comprueba esas tres cosas (ver *Preparación*). |
| `SyntaxError: f-string: expecting '}'` en la parte 2 ✅ (Python 3.10) | Comillas dobles anidadas en el f-string; exige Python ≥ 3.12 | Usa Python 3.12+ o calcula el `join` antes. |
| `ModuleNotFoundError: langchain_text_splitters` | Falta el paquete | `pip install langchain-text-splitters`. |
| `ModuleNotFoundError: langchain_openai` | No está instalado ✅ | `pip install langchain-openai`. |
| `OpenAIInvalidRequestError: Error code: 400 - This model's maximum context length is 128000 tokens. However, your messages resulted in 362243 tokens…` (observado por el alumno) | El prompt (1,32 M caracteres ≈ 362.243 tokens) supera la ventana de contexto | Trocear el texto (text splitters) y procesar por fragmentos. |
| La salida muestra `content='…' additional_kwargs=…` | `print(response)` imprime el objeto ✅ | Usa `response.content`. |

---

## Referencia

### Importaciones

| Importación | Paquete | Para qué |
|---|---|---|
| `PyPDFLoader` | `langchain_community.document_loaders` | Cargar un PDF como `Document` por página |
| `ChatOpenAI` | `langchain_openai` | Modelo de chat de OpenAI |

### Parámetros y métodos usados

| Elemento | Detalle |
|---|---|
| `PyPDFLoader(ruta)` | Ruta o URL del PDF; si no es válida → `ValueError` ✅ |
| `.load()` | Devuelve `list[Document]` |
| `ChatOpenAI(model, temperature)` | Necesita `OPENAI_API_KEY` ✅. Otros parámetros que lista la página oficial: `max_tokens`, `timeout`, `max_retries` (2 por defecto), `api_key`, `base_url` |
| `llm.invoke(entrada)` | Devuelve `AIMessage` ✅ |
| `AIMessage.content` | Texto de la respuesta. La documentación también menciona `.text`, `response_metadata` y `usage_metadata` |
| `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` | Splitter recomendado por la documentación para texto genérico. `chunk_size`: tamaño máximo medido por `length_function` (por defecto `len` ✅); `chunk_overlap`: solape objetivo |
| `.split_text(texto)` | Devuelve `list[str]` ✅ |
| `.create_documents([texto])` | Devuelve `list[Document]` ✅ (la documentación oficial lo describe) |
| `.split_documents(docs)` | Devuelve `list[Document]` conservando metadatos ✅. ⚠️ La página oficial consultada no lo menciona; existe en el paquete instalado |
| `load_dotenv()` | De `python-dotenv`. Busca `.env` desde la carpeta del script hacia arriba y carga sus variables; no sobrescribe las ya definidas ✅ |
| Separadores por defecto | `["\n\n", "\n", " ", ""]` ✅ (coincide con la página oficial) |

### Medidas de `quijote.pdf` ✅

| Dato | Valor |
|---|---|
| Páginas / `Document` | 471 |
| Páginas vacías | 0 |
| Caracteres por página | mín. 29 · mediana 3.144 · máx. 4.939 |
| `full_text` | 1.323.576 caracteres, ~240.000 palabras |
| Prompt completo (parte 1) | 1.323.646 caracteres ≈ 362.243 tokens (✅ dato de la API) |

### Medidas de la parte 2 ✅

| Dato | Valor |
|---|---|
| `split_documents(pages)` con `chunk_size=10000` | 471 chunks (uno por página), de 29 a 4.939 caracteres |
| `split_text(full_text)` con los mismos parámetros | 136 chunks, de 3.296 a 9.999 caracteres |
| Chunks resumidos por el script | 11 (páginas 0–10): 22.707 de 1.323.105 caracteres (1,7 %) |
| Llamadas al LLM (modelo simulado) | 12 (11 parciales + 1 final) |
| Tamaño de los prompts parciales | entre ~180 y ~2.700 caracteres |

---

## Explicación

### ¿Por qué falla (o fallaría) este enfoque?

Porque el prompt (362.243 tokens) supera con mucho lo que el modelo admite en una petición (128.000; lección 4, ✅ observado). Aunque cupiera, enviar todo costaría muchos tokens por petición ⚠️ (la página oficial de *chunking* habla de límites de tokens y coste al trocear para *embeddings*; no analiza este caso concreto).

### ¿Qué hacen los text splitters?

La página oficial de *chunking* dice que el troceado parte los documentos en piezas más pequeñas antes de indexarlos, y recomienda `RecursiveCharacterTextSplitter` como opción por defecto para texto genérico (ejemplo: `chunk_size=1000`, `chunk_overlap=200`, ambos en caracteres, como punto de partida). La página del splitter explica que el solape ayuda a no perder información cuando el contexto se reparte entre chunks. La parte 2 los usa para evitar el error de la parte 1.

### ¿Por qué falla la idea de la parte 2?

Porque combina dos decisiones que, juntas, no cubren el libro: `chunk_size=10000` es mayor que cualquier página (no se parte nada) y el bucle se detiene tras 11 chunks. El enfoque general (resumir cada fragmento y combinar los resúmenes) es válido 📘; lo que falla son los valores y el límite.

### ¿Por qué el solape de 200 si no se aplica?

Porque solo actúa entre chunks que salen del **mismo** texto cortado. Con `split_documents` sobre páginas que no se parten, no hay cortes y por tanto no hay solape.

### ¿Qué enfoque usar en cada caso?

| Situación | Enfoque |
|---|---|
| Texto corto que cabe holgadamente en el modelo | Enviarlo entero (como el script) |
| Texto largo (libro, informe extenso) | Trocear con un text splitter y procesar por fragmentos |
| Construir un RAG | Trocear → *embeddings* → vector store (visto en el proyecto del asistente legal) |
| Resumir un libro con pocas llamadas | Unir el texto, trocear con `split_text` (136 chunks) y resumir por fragmentos, sabiendo que cada chunk es una llamada |

### Lo que este documento no cubre

- Otros splitters (por tokens, por Markdown, por código…) y cómo elegir `chunk_size`.
- Cadenas LCEL o utilidades de LangChain para resumir (el script hace las llamadas a mano).
- Evaluar la calidad de los resúmenes.
- La ejecución con éxito contra OpenAI (el script falla antes de obtener respuesta).

### Observaciones sobre el archivo

1. 🔀 La ruta absoluta de otro usuario impide ejecutarlo tal cual (✅).
2. ✅ El prompt excede la ventana de contexto: la API devuelve el error 400 (362.243 tokens frente a 128.000). El script no produce ningún resumen.
3. `print(response)` muestra el objeto completo; para el texto, `response.content`.
4. "mas" va sin tilde en el prompt (es un literal; no se modifica).
5. Los comentarios originales solo son títulos de paso (`# 1.`, `# 2.`, `# 3.`) y no contienen afirmaciones técnicas.
6. El código se respetó: las tres versiones de cada script tienen el mismo código ✅, salvo las dos líneas de `load_dotenv` añadidas por el alumno a los cuatro archivos.

**Parte 2 (3.4):**

7. 🔀 `chunk_size=10000` no parte nada: 471 páginas → 471 chunks ✅.
8. 🔀 El bucle (`if i > 10: break`) resume solo 11 chunks (1,7 % del libro) ✅; el "resumen completo" final cubre solo esas páginas.
9. ⚠️ Los comentarios originales saltan de `# 1.` a `# 3.` (no hay `# 2.`) y dicen "chunks mas pequeños" sin tilde.
10. ⚠️ Python ≥ 3.12 por el f-string de la última línea ✅.
11. La carga del `.env` se resolvió con `load_dotenv()` (✅ comprobado con un `.env` de prueba). La parte 2 no la he ejecutado con la API real.

---

## Autoevaluación

1. ¿Cuántos `Document` devuelve `PyPDFLoader.load()` con `quijote.pdf` y por qué?
2. ¿Qué contiene `full_text` y cuántos caracteres tiene?
3. ¿Qué devuelve `llm.invoke(...)`?
4. ¿Qué error da `ChatOpenAI(...)` sin `OPENAI_API_KEY`?
5. ¿Qué error da la API al ejecutar el script con una clave válida y por qué?
6. ¿Cómo imprimirías solo el texto del resumen?
7. ¿Qué error da el script en un equipo donde la ruta no existe?
8. ¿Qué hace un text splitter y cuál recomienda la documentación para texto genérico?
9. En el script 3.4, ¿cuántos chunks salen de `split_documents(pages)` y por qué?
10. ¿Qué fracción del libro resume realmente el script 3.4 y por qué?

### Respuestas

1. 471: uno por página del PDF.
2. El texto de todas las páginas, separadas por `"\n"`: 1.323.576 caracteres.
3. Un `AIMessage`.
4. `OpenAIError` ("Missing credentials…") al crear el objeto.
5. Error 400: el prompt son 362.243 tokens y el modelo admite 128.000 (ventana de contexto). Habría que trocear.
6. `print(response.content)`.
7. `ValueError: File path … is not a valid file or url`.
8. Parte un texto largo en fragmentos; `RecursiveCharacterTextSplitter`.
9. 471, uno por página: cada página mide como máximo 4.939 caracteres, menos que `chunk_size=10000`, así que no se parte ninguna.
10. Unos 1,7 %: el bucle se detiene con `if i > 10: break` y procesa solo los 11 primeros chunks (páginas 0–10).

---

## Resumen

| Concepto | Idea clave |
|---|---|
| `PyPDFLoader.load()` | Un `Document` por página (471 en el Quijote) |
| `full_text` | Todo el libro en un string de 1,32 M caracteres |
| `ChatOpenAI.invoke` | Envía el prompt y devuelve un `AIMessage` |
| `response.content` | Texto de la respuesta |
| Ventana de contexto | Límite de tokens por petición (128.000 en `gpt-4o-mini`, ✅ según el error de la API); el libro entero (362.243) no cabe |
| Text splitter | Solución: trocear antes de enviar |
| `RecursiveCharacterTextSplitter` | `chunk_size` y `chunk_overlap` en caracteres; no parte textos más cortos que `chunk_size` |
| `split_documents` | Trocea cada `Document` por separado y conserva metadatos |
| Parte 2 tal cual | 471 chunks, solo 11 resumidos (1,7 %), 12 llamadas y Python ≥ 3.12 |

---

## Fuentes consultadas

- Integración `ChatOpenAI`: <https://docs.langchain.com/oss/python/integrations/chat/openai>
- Chunking: <https://docs.langchain.com/oss/python/langchain/chunking>
- `RecursiveCharacterTextSplitter`: <https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter>
- `PyPDFLoader` (ver documento 3.1)
- Archivos de origen: `3.3_text_splitters_parte1.py` (comentado), `3.3_text_splitters_parte1_limpio.py` (limpio), `3.4_text_splitters_parte2.py` (comentado), `3.4_text_splitters_parte2_limpio.py` (limpio) y `quijote.pdf` (solo estructura y medidas).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
