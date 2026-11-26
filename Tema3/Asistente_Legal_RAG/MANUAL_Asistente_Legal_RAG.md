# Manual: cómo crear un asistente RAG desde cero (caso práctico: Asistente Legal RAG)

> **Cómo usar este manual.** Está escrito para quien empieza en este tema. Léelo en orden: primero los **conceptos** (Parte 1), después la **arquitectura** (Parte 2) y, por último, el **paso a paso** (Parte 3), donde se construye el proyecto archivo a archivo. Las partes 4 a 8 son de consulta: recetas para adaptarlo, referencia, errores típicos, mejoras y una plantilla reutilizable. Termina con una autoevaluación.
>
> **Procedencia.** Elaborado a partir de los archivos del proyecto `Asistente_Legal_RAG` (`config.py`, `prompts.py`, `build_vectorstore.py`, `rag_system.py`, `app.py` y sus versiones numeradas 1.1 a 1.5), de los documentos `ESTUDIO_1.x` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo de los archivos; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = lo observado difiere de lo esperado o del original; ✅ *(ejecutado)* = comportamiento observado al ejecutar el código con `langchain` 1.5.2, `langchain-core` 1.8.4, `langchain-classic` 1.1.0, `langchain-chroma` 1.2.0, `chromadb` 1.5.9 y `streamlit` 1.67.0.
>
> **Qué se ha probado y qué no.** El proyecto completo (construir la base → consultar → interfaz) se ejecutó con **modelos simulados** (embeddings y modelos de chat falsos), documentos de prueba inventados y una base temporal (✅). Con la **API real** de OpenAI se ejecutaron por separado los scripts del tema 3 (embeddings, Chroma y el retriever simple de `3.7`); **el MultiQueryRetriever (`3.8`) y el proyecto completo con la API real no se han ejecutado todavía** (⚠️). Por eso este manual **no incluye una salida real del asistente**.
>
> **Privacidad.** Los contratos de ejemplo contienen datos personales. Este manual solo describe su estructura (número de PDF, páginas, chunks); **no reproduce su contenido**.

---

# Parte 1 — Conceptos básicos (léelo antes de tocar código)

## 1.1 El problema: un modelo de lenguaje no conoce tus documentos

Un **LLM** (*Large Language Model*, modelo de lenguaje, como GPT-4o) responde a partir de lo que aprendió en su entrenamiento. No conoce tus contratos, tus apuntes ni los documentos internos de tu empresa. Si le preguntas por ellos, puede decir que no sabe o, peor, **inventar** una respuesta convincente.

Dos soluciones obvias y por qué no bastan:

| Idea | Problema |
|---|---|
| Pegar todos los documentos en la pregunta | Los modelos tienen un límite de texto por petición (la *ventana de contexto*). ✅ Un caso real de este curso: una petición de unos 362.000 *tokens* fue rechazada por superar el límite de 128.000 del modelo (error 400). Además se paga por cada token. |
| Reentrenar el modelo con tus documentos | Es caro y complejo, y no se actualiza cuando cambian los documentos. |

## 1.2 La solución: RAG

**RAG** (*Retrieval-Augmented Generation*, generación aumentada por recuperación) consiste en **buscar primero** los fragmentos de tus documentos que tienen que ver con la pregunta y **dárselos al modelo junto con la pregunta**, para que conteste basándose en ellos. La documentación oficial de LangChain lo describe como dar al modelo acceso, en el momento de la consulta, a datos con los que no fue entrenado.

Analogía: es como un examen **a libro abierto**. El alumno (el LLM) no se sabe el libro de memoria, pero un ayudante (el buscador) le abre el libro por las páginas adecuadas antes de que conteste.

## 1.3 Las dos fases de un sistema RAG

La documentación oficial distingue una fase que se hace **antes** de las consultas (indexación) y otra que se hace **en cada consulta**.

```text
FASE 1 — INDEXACIÓN (se hace una vez, o cuando cambian los documentos)
  PDF ─► [cargar] ─► páginas ─► [trocear] ─► chunks ─► [embeddings] ─► vectores ─► [guardar] ─► base vectorial
        (Document)               (fragmentos)           (números)                    (Chroma en disco)

FASE 2 — CONSULTA (se hace con cada pregunta)
  pregunta ─► [recuperar] ─► fragmentos relevantes ─► [prompt] ─► [LLM] ─► respuesta
              (retriever)                              (pregunta + fragmentos)
```

## 1.4 Glosario

| Término | Qué significa (explicado desde cero) |
|---|---|
| **`Document`** | Objeto de LangChain con dos campos: `page_content` (el texto) y `metadata` (datos sobre él: de qué archivo viene, qué página…). |
| **Loader (cargador)** | Lee archivos (PDF, etc.) y los convierte en `Document`. Aquí, `PyPDFDirectoryLoader` genera **un `Document` por página**. |
| **Chunk** | Fragmento de texto de tamaño limitado (por ejemplo, 1000 caracteres) obtenido al trocear un documento. Se trocea porque un fragmento pequeño y concreto se busca mejor y cabe en el prompt. |
| **Solape (*overlap*)** | Cantidad de texto que se repite entre un chunk y el siguiente, para no cortar una idea justo por la mitad. |
| **Embedding** | Lista de números (un *vector*) que representa el **significado** de un texto. Textos con significado parecido tienen vectores parecidos. |
| **Modelo de embeddings** | El modelo que calcula esos vectores. Aquí, `text-embedding-3-large` de OpenAI. |
| **Vector store (base vectorial)** | Base de datos que guarda los textos junto a sus vectores y permite buscar «los más parecidos a esta pregunta». Aquí, **Chroma**. |
| **Búsqueda por similitud** | Convertir la pregunta en vector y devolver los chunks cuyos vectores están más cerca. |
| **Retriever** | Objeto estándar de LangChain que recibe una pregunta (texto) y devuelve una lista de `Document`. Un vector store se convierte en retriever con `as_retriever`. |
| **MMR** (*Maximal Marginal Relevance*) | Tipo de búsqueda que busca resultados parecidos a la pregunta **y a la vez distintos entre sí** (evita devolver tres veces lo mismo). |
| **MultiQueryRetriever** | Retriever que usa un LLM para reescribir la pregunta de varias formas, busca con cada una y une los resultados. |
| **EnsembleRetriever** | Retriever que combina los resultados de varios retrievers, dando un peso a cada uno. |
| **Prompt** | Texto que se envía al LLM con las instrucciones y los datos. |
| **Plantilla de prompt** | Prompt con huecos `{así}` que se rellenan en cada consulta. |
| **LCEL** | La forma de LangChain de encadenar piezas con el operador `\|` (tubería): la salida de una pieza es la entrada de la siguiente. |
| **Runnable** | Cualquier pieza de LangChain que se pueda ejecutar con `.invoke(entrada)`: modelos, retrievers, plantillas, cadenas. |
| **Streamlit** | Biblioteca de Python para crear una página web interactiva sin escribir HTML ni JavaScript. |
| **`.env`** | Archivo de texto, **fuera del control de versiones**, donde se guardan las claves (por ejemplo `OPENAI_API_KEY`). |

---

# Parte 2 — Arquitectura del proyecto

## 2.1 Principio de diseño: separar responsabilidades

Cada archivo hace **una sola cosa**. Así puedes cambiar un modelo, un prompt o la interfaz sin tocar el resto.

| Archivo | Responsabilidad | Se ejecuta… |
|---|---|---|
| `config.py` | Constantes: modelos, ruta de la base, parámetros de búsqueda | Nunca solo; lo importan otros |
| `prompts.py` | Textos de los prompts | Nunca solo; lo importan otros |
| `build_vectorstore.py` | **Fase 1**: construir la base vectorial a partir de los PDF | A mano, una vez (o al cambiar los PDF) |
| `rag_system.py` | **Fase 2**: recuperar + generar la respuesta | Lo importa `app.py` |
| `app.py` | Interfaz web de chat | `streamlit run app.py` |

## 2.2 Flujo completo

```text
 PDF en ../contratos
      │   build_vectorstore.py  (embeddings: OpenAI)
      ▼
 chroma_db/  ◄──────────────────────────────┐
      │                                      │ (misma carpeta y mismo modelo de embeddings)
      │   rag_system.py                      │
      ▼                                      │
 retrievers (MMR + MultiQuery + similitud) ──┘
      │  fragmentos
      ▼
 prompt (prompts.py) ─► LLM (GPT-4o) ─► respuesta ─► app.py (Streamlit) ─► usuario
```

## 2.3 Estructura de carpetas esperada

```text
Tema3/
├── contratos/                 ← los PDF (no se suben a ningún repositorio)
├── chroma_db/                 ← la crea build_vectorstore.py
└── Asistente_Legal_RAG/
    ├── .env                   ← OPENAI_API_KEY (¡nunca se comparte!)
    ├── config.py
    ├── prompts.py
    ├── build_vectorstore.py
    ├── rag_system.py
    └── app.py
```

> 📘 **Versiones numeradas del curso.** En la carpeta del curso existen también `1.1_config.py` … `1.5_app_limpio.py` (versión **comentada** y versión **limpia** de cada archivo) y `modulos_limpios.py`. Equivalencias: `1.1` = `config.py`; `1.2` = `prompts.py`; `1.3` = `build_vectorstore.py`; `1.4` = `rag_system.py`; `1.5` = `app.py`. Son material de estudio. `modulos_limpios.py` existe **solo** porque un archivo cuyo nombre empieza por un número (`1.1_config_limpio.py`) no se puede importar con `import` normal. **En un proyecto real no hace falta**: usa nombres sin número, como en este manual.

---

# Parte 3 — Paso a paso

## Paso 0 — Preparar el entorno

**Objetivo:** tener Python, las bibliotecas y la clave de API listas.

1. **Crea un entorno virtual** (una carpeta aislada con las bibliotecas de este proyecto, para que no choquen con las de otros):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Instala las bibliotecas.** Cada una cumple una función:

```bash
pip install -U langchain-core langchain-classic langchain-chroma langchain-community \
    langchain-openai langchain-text-splitters chromadb pypdf streamlit python-dotenv
```

| Paquete | Para qué |
|---|---|
| `langchain-core` | Piezas base: `Document`, plantillas, `Runnable`, retrievers |
| `langchain-chroma` | Conexión con Chroma (la forma vigente) |
| `langchain-openai` | Modelos de OpenAI: `OpenAIEmbeddings`, `ChatOpenAI` |
| `langchain-classic` | Retrievers «clásicos»: `MultiQueryRetriever`, `EnsembleRetriever` |
| `langchain-community` | Aquí, solo el cargador de PDF por carpeta |
| `langchain-text-splitters` | Troceado de texto |
| `chromadb`, `pypdf` | Motor de la base vectorial y lector de PDF |
| `streamlit` | La interfaz web |
| `python-dotenv` | Lee el archivo `.env` |

3. **Guarda la clave en `.env`** (archivo de texto llamado exactamente `.env`, en la carpeta del proyecto o en una superior):

```text
OPENAI_API_KEY=pon-aquí-tu-clave
```

> ⚠️ **Seguridad.** La clave es como una contraseña con tarjeta de crédito asociada. Nunca la pegues en el código, en un chat ni en un repositorio. Si usas Git, añade `.env` al archivo `.gitignore`. ✅ Un error real de este curso: `401 … API key has expired` por usar una clave caducada.

4. **Cómo se lee la clave en el código.** Todos los scripts que llaman a OpenAI incluyen:

```python
from dotenv import load_dotenv

load_dotenv()
```

`load_dotenv()` busca el `.env` empezando por la carpeta del script y subiendo por las superiores, y vuelca su contenido en las variables de entorno (✅). No sobrescribe una variable que ya exista en el entorno: si tienes una clave vieja exportada en tu terminal, **gana la vieja**.

## Paso 1 — `config.py`: los ajustes en un solo sitio

**Objetivo:** que los valores que quizá quieras cambiar (modelos, rutas, parámetros) estén juntos y con nombre, en lugar de repartidos por el código.

```python
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

CHROMA_DB_PATH = "/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"

SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70
```

| Constante | Qué controla |
|---|---|
| `EMBEDDING_MODEL` | Modelo que convierte textos en vectores. **Debe ser el mismo al construir la base y al consultarla.** |
| `QUERY_MODEL` | Modelo barato y rápido que **reescribe la pregunta** (`gpt-4o-mini`). |
| `GENERATION_MODEL` | Modelo que **redacta la respuesta final** (`gpt-4o`). |
| `CHROMA_DB_PATH` | Carpeta donde vive la base. Ruta absoluta de tu WSL; si mueves el proyecto, cámbiala. |
| `SEARCH_TYPE` | Tipo de búsqueda del retriever base: `mmr`. |
| `MMR_DIVERSITY_LAMBDA` | De 0 a 1: **0 = máxima diversidad**, 1 = mínima (según la documentación del método de Chroma ✅). |
| `MMR_FETCH_K` | Cuántos candidatos trae MMR antes de elegir los finales. |
| `SEARCH_K` | Cuántos fragmentos finales devuelve cada retriever. |
| `ENABLE_HYBRID_SEARCH` | Si es `True`, se combinan MMR y similitud (Paso 5). |
| `SIMILARITY_THRESHOLD` | ⚠️🔀 **Trampa:** se pasa a `EnsembleRetriever`, que no tiene ese parámetro; ✅ se ignora sin error y **no tiene ningún efecto**. |

**Buena práctica:** ¿por qué usar un modelo barato para reescribir y uno potente para responder? Reescribir una pregunta es una tarea sencilla; redactar la respuesta final es la que más calidad necesita. 📘

## Paso 2 — `prompts.py`: las instrucciones al modelo

**Objetivo:** guardar los textos de los prompts fuera del código lógico, para poder afinarlos sin tocar nada más.

El prompt principal (`RAG_TEMPLATE`) tiene dos huecos, `{context}` (los fragmentos recuperados) y `{question}` (la pregunta del usuario). Tiene este esqueleto:

```python
RAG_TEMPLATE = """Eres un asistente legal especializado en contratos de arrendamiento.
Basándote ÚNICAMENTE en los siguientes fragmentos de contratos, responde a la pregunta del usuario.

FRAGMENTOS DE CONTRATOS:
{context}

PREGUNTA: {question}
```

seguido de una lista de instrucciones (citar textualmente cuando sea relevante, incluir nombres, direcciones, importes y fechas, **indicar claramente si la información no está disponible**, y especificar a qué contrato o persona se refiere cada dato) y de la palabra `RESPUESTA:` al final.

**Por qué funciona así:**

- **«ÚNICAMENTE en los fragmentos»**: es la instrucción que reduce que el modelo invente datos. Sin ella, el modelo mezclaría los fragmentos con lo que «sabe» del mundo. ⚠️ Reduce el riesgo, **no lo elimina**: hay que comprobar las respuestas.
- **«Si no está disponible, indícalo»**: le da permiso para decir «no lo sé».
- Las variables entre llaves **deben coincidir con lo que entrega la cadena** (Paso 5): si la plantilla pide `{context}` y la cadena entrega otro nombre, falla.

El segundo prompt, `MULTI_QUERY_PROMPT`, pide al modelo que genere **3 versiones alternativas de la pregunta, una por línea, sin numeración**. Su variable es `{question}`. Sugiere variar la forma de nombrar personas (nombre completo, apellidos…), usar sinónimos legales y términos de ubicación y propiedad.

📘 `prompts.py` define además `RELEVANCE_PROMPT` y `ENTITY_EXTRACTION_PROMPT`, que **no se usan** en este proyecto.

## Paso 3 — `build_vectorstore.py`: construir la base vectorial (Fase 1)

**Objetivo:** leer los PDF, trocearlos, calcular los embeddings y guardarlos en Chroma. Se ejecuta **una vez** (o cuando cambian los PDF).

### 3.1 Importaciones y parámetros

```python
import argparse
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, CHROMA_DB_PATH

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CARPETA_PDF_POR_DEFECTO = Path(__file__).resolve().parent.parent / "contratos"
```

- `from config import ...`: la base se construye con **el mismo modelo y la misma carpeta** que luego usará `rag_system.py`. Es la razón de tener un `config.py`.
- `CHUNK_SIZE = 1000` y `CHUNK_OVERLAP = 200` (caracteres): son los valores del curso (📘). Con la carpeta de prueba del curso, 15 páginas dieron 39 chunks (✅).
- `Path(__file__).resolve().parent.parent / "contratos"`: la carpeta `contratos` situada un nivel por encima del proyecto. ⚠️ Es una suposición de estructura; se puede pasar otra carpeta como argumento.

### 3.2 Ids estables: evitar duplicados

```python
def crear_ids(chunks):
    ids = []
    vistos = {}
    for chunk in chunks:
        archivo = Path(chunk.metadata.get("source", "")).name
        pagina = chunk.metadata.get("page", "")
        huella = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        base = f"{archivo}|{pagina}|{huella}"
        n = vistos.get(base, 0)
        vistos[base] = n + 1
        ids.append(f"{base}|{n}")
    return ids
```

**El problema (✅ ocurrió de verdad en el curso):** si ejecutas la construcción dos veces **sin ids**, Chroma añade los mismos chunks otra vez (39 → 78 vectores) y una búsqueda puede devolver **el mismo fragmento tres veces**.

**La solución:** darle a cada chunk un **id que siempre sea el mismo para el mismo contenido**: nombre del archivo + página + una huella (*hash*) del texto + un contador por si dos chunks de la misma página fueran idénticos. Con ids repetidos, Chroma **actualiza** en vez de duplicar (✅ 39 → 39).

Límites que debes conocer:

- ⚠️ Los embeddings **se recalculan todos** en cada ejecución (hay coste de API aunque no crezca la base).
- Si cambia el texto de un chunk, cambia su huella y su id: **el chunk antiguo se queda en la base**. Si modificas PDF o el troceado, usa `--reconstruir`.

### 3.3 Construir

```python
def construir(carpeta_pdf, reconstruir):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    documentos = PyPDFDirectoryLoader(str(carpeta_pdf)).load()
    if not documentos:
        raise SystemExit(f"No se encontraron PDF en {carpeta_pdf}")
    print(f"Se cargaron {len(documentos)} documentos desde {carpeta_pdf}.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Se crearon {len(chunks)} chunks de texto.")

    if reconstruir:
        Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH).delete_collection()
        print("Colección anterior eliminada.")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        ids=crear_ids(chunks),
        persist_directory=CHROMA_DB_PATH
    )
    print(f"La base contiene {len(vectorstore.get()['ids'])} vectores en {CHROMA_DB_PATH}.")
```

| Paso del código | Qué hace |
|---|---|
| `PyPDFDirectoryLoader(...).load()` | Lee todos los PDF de la carpeta: **un `Document` por página** (✅ 5 PDF de 3 páginas = 15 `Document`). |
| `if not documentos: raise SystemExit(...)` | Si la carpeta está vacía, se detiene con un mensaje claro en vez de fallar más adelante. |
| `RecursiveCharacterTextSplitter(...).split_documents(...)` | Trocea **cada página por separado** y conserva los metadatos. Una página más corta que `chunk_size` no se parte. |
| `delete_collection()` (solo con `--reconstruir`) | Borra la colección anterior. Se hace **después** de cargar y trocear, para no perder la base antigua si la lectura de PDF falla. |
| `Chroma.from_documents(...)` | Calcula el vector de cada chunk (**llamadas de pago a OpenAI**) y guarda textos, metadatos y vectores en disco (`persist_directory`). La colección se llama `langchain` (la de por defecto ✅). |
| `vectorstore.get()['ids']` | Los ids guardados; su longitud es el número de vectores. |

### 3.4 Punto de entrada y ejecución

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye (o actualiza) la base vectorial Chroma del asistente legal.")
    parser.add_argument("carpeta_pdf", nargs="?", default=str(CARPETA_PDF_POR_DEFECTO), help="Carpeta con los PDF")
    parser.add_argument("--reconstruir", action="store_true", help="Borra la colección existente antes de crearla")
    args = parser.parse_args()
    construir(Path(args.carpeta_pdf), args.reconstruir)
```

`if __name__ == "__main__":` hace que el código de dentro se ejecute solo al lanzar el archivo directamente, no al importarlo. Uso:

```bash
python build_vectorstore.py                      # usa ../contratos
python build_vectorstore.py ruta/a/otra/carpeta  # otra carpeta de PDF
python build_vectorstore.py --reconstruir        # borra y crea de nuevo
```

Salida esperada (con la carpeta de prueba del curso): `Se cargaron 15 documentos…`, `Se crearon 39 chunks…`, `La base contiene 39 vectores en …`. ✅ Probado con embeddings simulados; ⚠️ pendiente con la API real.

## Paso 4 — Comprobar la recuperación **antes** de añadir el modelo

**Objetivo:** verificar que la base devuelve fragmentos sensatos antes de construir lo demás. Si la recuperación falla, ningún prompt lo arreglará.

El script `3.7_retrievers_langchain.py` del curso hace exactamente eso: **abre** la base (no la recrea), crea un retriever y le hace una pregunta:

```python
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
```

Y se consulta con `retriever.invoke(consulta)`, que devuelve una lista de `Document`. ✅ **Salida real del curso:** con `k=2` devolvió 2 fragmentos: el 1.º contenía la descripción y ubicación del local (respondía a la pregunta) y el 2.º era la página de firmas del mismo contrato (solo mencionaba el nombre).

**Lecciones de esa prueba:**

1. La búsqueda por similitud devuelve **los `k` más parecidos, no necesariamente los relevantes**.
2. ⚠️ **Fallo silencioso (✅ comprobado):** si `persist_directory` apunta a una carpeta que no existe, Chroma **crea una vacía** y el retriever devuelve `[]` **sin ningún error**. Comprueba siempre que la base tiene datos: `len(vectorstore.get()["ids"])`.
3. Si usas otro modelo de embeddings que el de la base, Chroma falla por tamaño de vector distinto (✅).

## Paso 5 — `rag_system.py`: la lógica RAG (Fase 2)

**Objetivo:** recuperar fragmentos, construir el prompt, llamar al modelo y devolver la respuesta. Es el cerebro del asistente; `app.py` solo lo usa.

### 5.1 Importaciones y configuración

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st
from dotenv import load_dotenv

from config import *
from prompts import *

load_dotenv()
```

- `from config import *` y `from prompts import *` traen **todas** las constantes de esos archivos.
- 🔀 **Importaciones vigentes.** El código original del curso estaba escrito para LangChain 0.x; con LangChain 1.x fallaban `langchain.prompts` y `langchain.retrievers` (✅ `ModuleNotFoundError`). Ahora se usan `langchain_core.prompts` y `langchain_classic.retrievers`, y `Chroma` viene de `langchain_chroma` (la de `langchain_community` está obsoleta).
- `streamlit` se importa solo para `st.cache_resource` (más abajo).

### 5.2 Inicializar el sistema una sola vez

```python
@st.cache_resource
def initialize_rag_system():
```

Montar los retrievers y modelos es lento; no debe repetirse con cada pregunta. `@st.cache_resource` hace que la función se ejecute **una vez** y su resultado se reutilice (✅ dos llamadas seguidas devolvieron el mismo objeto y los modelos se crearon una sola vez).

### 5.3 Base vectorial y modelos

```python
    vectorestore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )

    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)
```

Se **abre** la base (como en el Paso 4) y se crean **dos** modelos de chat: uno barato para reescribir preguntas y otro potente para responder. `temperature=0` reduce la aleatoriedad de la respuesta (⚠️ no garantiza respuestas idénticas entre ejecuciones). *(El original lleva una `\` sobrante al final de la línea del `embedding_function`; es inofensiva y aquí se omite. `vectorestore` es una errata del nombre de la variable, conservada del original.)*

### 5.4 Los tres retrievers

```python
    base_retriever = vectorestore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    similarity_retriever = vectorestore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K}
    )

    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )
```

| Objeto | Qué hace |
|---|---|
| `base_retriever` | Búsqueda **MMR**: trae `fetch_k=20` candidatos y elige `k=2` equilibrando parecido y diversidad (`lambda_mult=0.7`). |
| `similarity_retriever` | Búsqueda por similitud simple, `k=2`. |
| `multi_query_prompt` | `MULTI_QUERY_PROMPT` convertido en plantilla (`PromptTemplate.from_template`). |
| `mmr_multi_retriever` | El `base_retriever` envuelto en **MultiQuery**: `llm_queries` reescribe la pregunta con **tu** plantilla, se busca con cada versión y se unen los resultados sin repetidos (✅; por defecto la pregunta original **no** se busca, `include_original=False`). |

> 🔀 El comentario del original dice «Maximal *Margin* Relevance»; el nombre correcto es *Maximal **Marginal** Relevance* (✅ documentación del método de Chroma).

### 5.5 El retriever híbrido

```python
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # mayor peso a MMR
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever
```

`EnsembleRetriever` combina los resultados de varios retrievers con una **fusión por rangos** (✅ docstring de la clase instalada) y los pesos `[0.7, 0.3]` (más peso a MMR+MultiQuery). Idea del curso: que la diversidad de MMR y la precisión de la similitud simple se complementen (📘, ⚠️ **no se midió** si mejora los resultados).

Recuerda la trampa del Paso 1: `similarity_threshold` **se ignora**. Y los pesos están escritos a mano aquí, no en `config.py`.

### 5.6 Formatear los fragmentos para el prompt

```python
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"

            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")

        return "\n\n".join(formatted)
```

`format_docs` convierte la lista de `Document` en **un único texto** con un encabezado por fragmento (`[Fragmento 1] - Fuente: … - Pagina: …`) para ponerlo en el hueco `{context}` del prompt. Poner la fuente y la página ayuda al modelo a citar de dónde sale cada dato. ✅ El `.split("\\")` quita la ruta de Windows y deja solo el nombre del archivo.

### 5.7 La cadena RAG (LCEL)

```python
    rag_chain = (
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever
```

Se lee de arriba abajo, como una tubería:

```text
pregunta (texto)
   ├─► "context":  final_retriever ─► format_docs ─► texto con fragmentos
   └─► "question": RunnablePassthrough ─► la misma pregunta, sin cambios
          │
          ▼  {"context": ..., "question": ...}
      prompt  (RAG_TEMPLATE rellena {context} y {question})
          ▼
      llm_generation (GPT-4o)  ─► mensaje del modelo
          ▼
      StrOutputParser ─► texto plano (la respuesta)
```

- Un diccionario `{...}` dentro de una cadena LCEL ejecuta **en paralelo** cada entrada con la misma pregunta y entrega un diccionario al siguiente paso.
- `RunnablePassthrough()` deja pasar la entrada sin tocarla.
- `StrOutputParser()` convierte el mensaje del modelo en un `str`.
- La función **devuelve dos cosas**: la cadena y el retriever `mmr_multi_retriever` (**no** el `ensemble`). Eso tiene una consecuencia, comentada en la Parte 7.

### 5.8 `query_rag`: la función que usa la interfaz

```python
def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        response = rag_chain.invoke(question)

        docs = retriever.invoke(question)

        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)

        return response, docs_info

    except Exception as e:
        error_msg = f"Error al procesar la cosulta: {str(e)}"
        return error_msg, []
```

- Devuelve **dos cosas**: la respuesta (texto) y una lista de diccionarios con los fragmentos (`fragmento`, `contenido`, `fuente`, `pagina`) para mostrarlos en pantalla.
- Se usa **`.invoke(...)`**. 🔀 El original usaba `get_relevant_documents`, que ✅ ya no existe en LangChain 1.x.
- El `try/except` **captura cualquier error y lo devuelve como texto de respuesta** (✅): la interfaz nunca recibe una excepción. A cambio, **un error se muestra como si fuera la respuesta del asistente** (por ejemplo, una clave caducada). Si ves un texto que empieza por «Error al procesar…», mira ahí. *(«cosulta» es una errata del original.)*

### 5.9 `get_retriever_info`

```python
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }
```

Devuelve un diccionario con la configuración para mostrarla en la barra lateral (la interfaz solo usa `tipo`).

## Paso 6 — `app.py`: la interfaz de chat con Streamlit

**Objetivo:** una página web con un chat. **Idea clave de Streamlit:** cada vez que el usuario interactúa (escribe, pulsa un botón), Streamlit **vuelve a ejecutar el script completo de arriba abajo**. Por eso lo que debe sobrevivir entre ejecuciones (el historial) se guarda en `st.session_state`.

```python
import streamlit as st
from rag_system import query_rag, get_retriever_info

st.set_page_config(
    page_title="Sistema RAG - Asistente Legal",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
```

- `st.set_page_config(...)`: título de la pestaña, icono y diseño ancho. Colócala al principio del script (convención; ✅ en Streamlit 1.67 no dio error al llamarla después de otro comando, ⚠️ no se comprobó en versiones anteriores).
- `if "messages" not in st.session_state`: crea el historial **solo la primera vez**; en las siguientes ejecuciones ya existe.

**Barra lateral y botón de limpiar:**

```python
with st.sidebar:
    st.header("📋 Información del Sistema")

    retriever_info = get_retriever_info()

    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {retriever_info['tipo']}")

    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
```

**Dos columnas: chat a la izquierda, fragmentos a la derecha:**

```python
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
```

Cada mensaje del historial es un diccionario con `role` (`"user"` o `"assistant"`) y `content`; los del asistente llevan además `docs` (los fragmentos). La columna derecha muestra, dentro de un `st.expander`, los fragmentos de la **última** respuesta (fuente, página y contenido).

**Recibir la pregunta y responder:**

```python
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🔍 Analizando..."):
        response, docs = query_rag(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})

    st.rerun()
```

| Código | Qué hace |
|---|---|
| `st.chat_input(...)` | Caja de texto de chat. Devuelve el texto cuando el usuario envía, y `None` si no. |
| `if prompt := ...` | Operador «morsa» (`:=`): asigna y comprueba a la vez. |
| `messages.append(...)` | Guarda la pregunta **antes** de llamar al sistema. |
| `st.spinner(...)` | Muestra «Analizando…» mientras `query_rag` trabaja. |
| `st.rerun()` | Relanza el script para pintar los mensajes nuevos (el historial ya se dibujó más arriba). |

✅ Con `streamlit.testing` y modelos simulados: la app arranca sin excepciones, acepta una pregunta y muestra la respuesta.

## Paso 7 — Ejecutar el proyecto

```bash
source .venv/bin/activate
python build_vectorstore.py        # Fase 1: una vez
streamlit run app.py               # Fase 2: abre la interfaz en el navegador
```

Orden obligatorio: **primero la base, después la interfaz**. Streamlit imprime una dirección local (normalmente `http://localhost:8501`).

## Paso 8 — Comprobar y depurar

Lista de comprobación cuando algo no responde bien (de lo más barato a lo más caro):

1. ¿`.env` existe y contiene `OPENAI_API_KEY`? ¿No hay una clave vieja exportada en la terminal?
2. ¿`CHROMA_DB_PATH` apunta a la carpeta correcta? ¿`len(vectorstore.get()["ids"])` es mayor que 0?
3. ¿El `EMBEDDING_MODEL` es el mismo con el que se construyó la base?
4. ¿Los PDF son texto real (no imágenes escaneadas)? ⚠️ `PyPDFDirectoryLoader` extrae el texto que contiene el PDF; un PDF escaneado sin texto daría documentos vacíos (no probado).
5. ¿La respuesta empieza por «Error al procesar…»? Es una excepción capturada: lee el mensaje.
6. ¿Los fragmentos mostrados son los correctos? Ojo: ver «Limitaciones» en la Parte 7.
7. Recupera sin modelo (Paso 4) para saber si el fallo es de **recuperación** o de **generación**.

---

# Parte 4 — Guía práctica: adaptar el proyecto a otro dominio

| Quiero… | Cambio |
|---|---|
| Usar mis propios documentos | Pon los PDF en otra carpeta y ejecuta `python build_vectorstore.py ruta/carpeta --reconstruir` |
| Cambiar el tema del asistente (RRHH, medicina, apuntes…) | Reescribe `RAG_TEMPLATE` y `MULTI_QUERY_PROMPT` en `prompts.py` (rol, instrucciones, tipo de datos a destacar) |
| Que devuelva más fragmentos | Sube `SEARCH_K` en `config.py` (más contexto = más coste y más ruido) |
| Quitar el modo híbrido | `ENABLE_HYBRID_SEARCH = False` (solo MMR + MultiQuery) |
| Más variedad en los resultados | Baja `MMR_DIVERSITY_LAMBDA` (más cerca de 0 = más diversidad) |
| Cambiar de modelo | Cambia `GENERATION_MODEL` / `QUERY_MODEL`; no necesitas reconstruir la base |
| Cambiar el modelo de **embeddings** | Cambia `EMBEDDING_MODEL` **y reconstruye la base** (`--reconstruir`) |
| Cambiar el tamaño de chunk | Cambia `CHUNK_SIZE` / `CHUNK_OVERLAP` y reconstruye |
| Añadir PDF nuevos | Ponlos en la carpeta y ejecuta `build_vectorstore.py` (los ya existentes no se duplican) |
| Quitar un PDF de la base | `--reconstruir` (los ids de PDF borrados no se eliminan solos) |
| Ver las preguntas que genera el MultiQuery | Activa el logging INFO del logger `langchain_classic.retrievers.multi_query` (✅ visible solo así) |
| Mover el proyecto | Cambia `CHROMA_DB_PATH` en `config.py` |

---

# Parte 5 — Referencia

## 5.1 Qué llamadas a la API hace cada operación

| Operación | Llamadas | Coste relativo |
|---|---|---|
| `build_vectorstore.py` | 1 embedding **por chunk** (39 en el ejemplo), en **cada** ejecución ✅ | Alto la 1.ª vez; se repite en cada ejecución |
| Una pregunta en el asistente | ✅ con modelos simulados que devuelven 3 variantes: **3 llamadas a modelos de chat** (2 de `gpt-4o-mini` para reescribir + 1 de `gpt-4o` para responder) y **7 de embeddings** | Medio por pregunta |

Por qué 7 embeddings y 2 reescrituras: la cadena ejecuta el MultiQuery (3 variantes → 3 embeddings) **y** el retriever de similitud (1 embedding); luego `query_rag` vuelve a ejecutar el MultiQuery para mostrar fragmentos (otra reescritura y 3 embeddings más). ⚠️ El número real de variantes lo decide el modelo; las cifras son aproximadas.

## 5.2 Versiones y paquetes (los de las pruebas)

`langchain` 1.5.2 · `langchain-core` 1.8.4 · `langchain-classic` 1.1.0 · `langchain-chroma` 1.2.0 · `langchain-community` 0.4.2 · `langchain-openai` 1.8.5 · `chromadb` 1.5.9 · `streamlit` 1.67.0 · `python-dotenv` 1.2.3 · Python 3.13 en las pruebas (el entorno del alumno usa 3.14).

## 5.3 Tabla de piezas de LangChain usadas

| Pieza | Paquete | Función |
|---|---|---|
| `PyPDFDirectoryLoader` | `langchain_community` | PDF → `Document` (uno por página) |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | `Document` → chunks |
| `OpenAIEmbeddings` | `langchain_openai` | Texto → vector |
| `Chroma` | `langchain_chroma` | Base vectorial |
| `as_retriever` | método del vector store | Base → retriever |
| `MultiQueryRetriever` | `langchain_classic.retrievers.multi_query` | Reescribe la pregunta con un LLM |
| `EnsembleRetriever` | `langchain_classic.retrievers` | Combina retrievers con pesos |
| `PromptTemplate` | `langchain_core.prompts` | Plantilla con huecos |
| `RunnablePassthrough`, `StrOutputParser` | `langchain_core` | Piezas de LCEL |
| `ChatOpenAI` | `langchain_openai` | Modelo de chat |

---

# Parte 6 — Errores típicos y cómo resolverlos

Todos ocurrieron (✅) durante la construcción de este proyecto o del tema que lo precede.

| Síntoma | Causa | Solución |
|---|---|---|
| `OpenAIError: Missing credentials…` | No se encuentra `OPENAI_API_KEY` | Ponla en `.env` y asegúrate de llamar a `load_dotenv()` |
| `AuthenticationError 401 … API key has expired` | Clave caducada, o una clave vieja ya definida en el entorno (que `load_dotenv()` no sobrescribe) | Clave nueva en `.env` y revisa las variables de entorno |
| `400 … context length` (362.243 tokens frente a 128.000) | Se envió demasiado texto en una sola petición (se resumía todo el documento de una vez) | Trocear antes y enviar solo fragmentos (justo lo que hace RAG) |
| Los 3 resultados son el mismo fragmento | Base con vectores duplicados por ejecutar la construcción varias veces sin ids | Borrar la carpeta de la base o `--reconstruir`; usar ids estables |
| El asistente no encuentra nada, sin error | `CHROMA_DB_PATH` apunta a una carpeta que no existe (Chroma la crea vacía) o a una ruta de Windows en Linux | Corregir la ruta; comprobar `len(vectorstore.get()["ids"])` |
| `ModuleNotFoundError: No module named 'langchain.prompts'` / `'langchain.retrievers'` | Código escrito para LangChain 0.x | `langchain_core.prompts` y `langchain_classic.retrievers`; `pip install -U langchain-classic` |
| `Error al procesar la cosulta: … get_relevant_documents` | Método eliminado en LangChain 1.x | Usar `.invoke(...)` |
| `LangChainDeprecationWarning: The class Chroma was deprecated` (solo con todos los avisos activos) | `Chroma` importado de `langchain_community` | `from langchain_chroma import Chroma` |
| `DeprecationWarning: langchain-community is being sunset` | Se importa un cargador de `langchain_community` | Inofensivo por ahora: no hay cargador equivalente en otro paquete en este proyecto |
| `InvalidArgumentError … dimension` | Modelo de embeddings distinto al de la base | Mismo `EMBEDDING_MODEL`, o reconstruir la base |
| `SyntaxError` con comillas dentro de un f-string (en 3.4) | Comillas dobles anidadas en f-string; solo válidas desde Python 3.12 | Usar Python ≥ 3.12 o comillas distintas |
| `ValueError … is not a valid file or url` | Ruta de PDF inexistente (por ejemplo, ruta de Windows `\\wsl.localhost\…` o de otro equipo) | Ruta válida en el sistema donde ejecutas |

---

# Parte 7 — Limitaciones conocidas y mejoras posibles

## 7.1 Limitaciones de este proyecto (comprobadas en el código)

1. **Los fragmentos mostrados no son exactamente los usados.** `query_rag` muestra los de `mmr_multi_retriever`, mientras que la respuesta se genera con el retriever híbrido (`final_retriever`). Además se hace una segunda reescritura (coste extra) y, al reformular con un LLM, las variantes pueden diferir entre ambas llamadas. ✅
2. **`SIMILARITY_THRESHOLD` no hace nada.** ✅
3. **Los errores se disfrazan de respuesta.** ✅
4. **Pesos `[0.7, 0.3]` fijos en el código**, no en `config.py`.
5. **No hay evaluación de calidad.** No se ha medido si el modo híbrido, MMR o el MultiQuery mejoran las respuestas (⚠️).
6. **Una sola consulta de prueba.** Lo observado con la API real en el curso es **una consulta con `k=2` o `k=3`**; no basta para concluir nada sobre la calidad general.

## 7.2 Mejoras posibles (sugerencias, no aplicadas)

- Devolver `final_retriever` para mostrar los fragmentos realmente usados y **evitar la segunda reescritura** (una sola recuperación reutilizada).
- Mover pesos, tamaño de chunk y carpeta de PDF a `config.py`.
- Evitar recalcular embeddings de chunks que ya existen (comprobar los ids antes de embeber). ⚠️ no probado.
- Borrar de la base los chunks de PDF que ya no existen, en lugar de reconstruir.
- Pasar la ruta de la base como ruta relativa al proyecto (`pathlib`) o leerla del `.env`.
- Mostrar al usuario si la respuesta se basa o no en los fragmentos, y citar la fuente y la página en la respuesta.
- Crear un pequeño conjunto de preguntas de prueba con respuesta conocida para **medir** cambios (evaluación).

## 7.3 Privacidad y seguridad en proyectos con documentos reales

- **Los fragmentos recuperados se envían al proveedor del modelo** (aquí, OpenAI) en cada pregunta, y los textos de todos los chunks, para calcular embeddings, al construir la base. Si los documentos contienen **datos personales** (nombres, DNI, direcciones…), revisa las condiciones del proveedor y la normativa aplicable antes de usar documentos reales. ⚠️ Esto es una advertencia general, **no asesoramiento legal**.
- La carpeta `chroma_db` **contiene el texto de los documentos** (no solo vectores): trátala con la misma protección que los originales y no la subas a repositorios públicos.
- No compartas ni subas `.env`, `credentials.json` ni `token.json`.

---

# Parte 8 — Plantilla reutilizable para un nuevo proyecto RAG

**Orden de trabajo recomendado** (verifica cada paso antes de pasar al siguiente):

1. **Define el caso:** qué documentos, qué preguntas debe poder responder y qué hacer cuando no sepa.
2. **Entorno:** `venv`, paquetes, `.env` con la clave (y `.gitignore`).
3. **`config.py`:** modelos, rutas y parámetros de búsqueda con nombres claros.
4. **`prompts.py`:** prompt principal (con `{context}` y `{question}`) y, si usas MultiQuery, su prompt. Incluye siempre «responde únicamente con los fragmentos» y «si no está, dilo».
5. **`build_vectorstore.py`:** cargar → trocear → ids estables → embeddings → guardar. Mensajes claros y opción `--reconstruir`.
6. **Comprueba la base:** ¿tiene datos? ¿devuelve fragmentos sensatos con un retriever simple (sin LLM)?
7. **`rag_system.py`:** retriever(s) → formateo de fragmentos → cadena LCEL → función `query_rag` que devuelva respuesta **y** fragmentos.
8. **Prueba la cadena sin interfaz** con 5-10 preguntas conocidas y anota los resultados.
9. **`app.py`:** interfaz mínima (historial en `session_state`, `chat_input`, fragmentos visibles).
10. **Documenta:** qué se ejecutó con la API real y qué no; límites conocidos.

**Estructura mínima de archivos:** `.env` · `config.py` · `prompts.py` · `build_vectorstore.py` · `rag_system.py` · `app.py`.

**Reglas de oro:**

- La base y la consulta usan **el mismo modelo de embeddings**.
- **Construir** y **consultar** son procesos separados.
- Comprueba la **recuperación** antes de culpar al prompt.
- Ids estables para no duplicar; `--reconstruir` cuando cambie el contenido.
- Un error nunca debe parecer una respuesta: muéstralo como error (mejora sobre este proyecto).
- Nunca pongas claves en el código.

---

# Autoevaluación

1. ¿Qué problema resuelve RAG y por qué no basta con pegar todos los documentos en la pregunta?
2. Describe las dos fases de un sistema RAG y di en qué archivo del proyecto está cada una.
3. ¿Qué es un chunk y para qué sirve el solape?
4. ¿Qué es un embedding y qué relación tiene con la búsqueda «por significado»?
5. ¿Por qué la base y la consulta deben usar el mismo modelo de embeddings?
6. ¿Qué diferencia hay entre una búsqueda por similitud y una MMR?
7. ¿Qué hace el `MultiQueryRetriever` y cuántas llamadas extra a modelos añade por pregunta?
8. ¿Por qué `from_documents` sin ids duplicaba los vectores y cómo lo evitan los ids estables? ¿Qué no resuelven?
9. En la cadena `{"context": final_retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`, ¿qué entra y qué sale en cada paso?
10. ¿Por qué `query_rag` captura todas las excepciones y qué inconveniente tiene?
11. En Streamlit, ¿por qué el historial se guarda en `st.session_state`?
12. Si el asistente contesta «no encuentro información» a una pregunta cuya respuesta está en los PDF, ¿qué comprobarías primero y en qué orden?

## Respuestas

1. Los modelos no conocen tus documentos y pueden inventar; pegarlos todos choca con el límite de contexto y con el coste (✅ error 400 por 362.243 tokens frente a 128.000). RAG busca primero los fragmentos relevantes y se los da al modelo con la pregunta.
2. Indexación (cargar, trocear, embeddings, guardar): `build_vectorstore.py`. Consulta (recuperar, construir el prompt, generar): `rag_system.py`, usado por `app.py`.
3. Un fragmento de tamaño limitado de un documento. El solape repite parte del texto entre chunks consecutivos para no cortar una idea por la mitad.
4. Una lista de números que representa el significado de un texto; textos de significado parecido tienen vectores cercanos, y la búsqueda devuelve los chunks más cercanos al vector de la pregunta.
5. Para que los vectores sean comparables; con tamaños distintos Chroma falla (✅) y con modelos distintos del mismo tamaño los resultados no tendrían sentido (⚠️).
6. La similitud devuelve los más parecidos a la pregunta; MMR busca parecido a la pregunta **y** diversidad entre los resultados.
7. Reescribe la pregunta en varias versiones con un LLM, busca con cada una y une los resultados sin repetidos. Añade 1 llamada al modelo de reescritura (y ~3 de embeddings) por ejecución del retriever (✅).
8. Sin ids, cada ejecución añade los mismos chunks otra vez. Con ids estables (archivo+página+huella+contador), Chroma actualiza en lugar de duplicar. No evitan recalcular embeddings ni eliminan los chunks de textos que han cambiado.
9. Entra la pregunta; en paralelo se obtiene el contexto (retriever → texto formateado) y la pregunta intacta; el diccionario rellena el prompt; el modelo genera un mensaje; el analizador lo convierte en texto.
10. Para que la interfaz nunca reciba una excepción; el inconveniente es que un error se muestra como si fuera una respuesta.
11. Porque Streamlit vuelve a ejecutar el script entero en cada interacción; las variables normales se reinician y `session_state` conserva el valor.
12. 1) Clave y entorno; 2) que la base tenga datos y la ruta sea correcta; 3) mismo modelo de embeddings; 4) que el retriever simple devuelva el fragmento correcto; 5) el prompt y el `k`.

---

# Resumen

| Concepto | Idea clave |
|---|---|
| RAG | Buscar fragmentos relevantes y dárselos al LLM junto con la pregunta |
| Dos fases | Indexar (una vez) · Consultar (cada pregunta) |
| Chunk + embedding | Texto troceado → vectores que representan significado |
| Chroma | Base vectorial en disco; misma carpeta y mismo modelo al construir y consultar |
| Retriever | Texto → lista de `Document`; `similarity`, `mmr`, MultiQuery, Ensemble |
| Cadena LCEL | `{contexto, pregunta} \| prompt \| LLM \| parser` |
| Ids estables | Evitan duplicados al re-ejecutar; no evitan el coste ni limpian lo obsoleto |
| `.env` | La clave nunca va en el código; `load_dotenv()` la lee |
| Streamlit | Reejecuta el script en cada interacción; el historial va en `session_state` |
| Límites del proyecto | Fragmentos mostrados ≠ usados; umbral ignorado; errores como respuesta; sin evaluación |

---

# Fuentes consultadas

- LangChain, *Build a RAG agent* (conceptos de indexación y recuperación): <https://docs.langchain.com/oss/python/langchain/rag>
- LangChain, integración con Chroma: <https://docs.langchain.com/oss/python/integrations/vectorstores/chroma>
- LangChain, retrievers: <https://docs.langchain.com/oss/python/integrations/retrievers>
- LangChain Reference, `MultiQueryRetriever`: <https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever>
- Documentación del código instalado (`Chroma.max_marginal_relevance_search`, `EnsembleRetriever`), comprobada con ejecución (✅).
- Archivos del proyecto: `config.py`, `prompts.py`, `build_vectorstore.py`, `rag_system.py`, `app.py`; versiones `1.1` a `1.5` (comentadas y `_limpio`); `3.7_retrievers_langchain.py`; documentos `ESTUDIO_1.1` a `ESTUDIO_1.5`.

*Documento generado a partir de los materiales del proyecto y la documentación oficial de LangChain. Las afirmaciones sobre Streamlit proceden de los archivos del proyecto y de ejecuciones con `streamlit.testing` (✅), no de su documentación oficial.*
