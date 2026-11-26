# 1.4 El motor RAG: Chroma, MultiQuery, búsqueda híbrida y cadena LCEL (`rag_system.py`)

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `rag_system.py` (original; **cambios posteriores, ✅ las únicas líneas de código distintas:** `Chroma` se importa de `langchain_chroma` en lugar de `langchain_community`; `PromptTemplate`, `MultiQueryRetriever` y `EnsembleRetriever` se importan de `langchain_core` / `langchain_classic` en lugar de `langchain.*`; y `get_relevant_documents` se sustituye por `.invoke`), sus versiones `1.4_rag_system.py` (comentada) y `1.4_rag_system_limpio.py` (limpia; código idéntico, ✅ AST) y la documentación oficial de LangChain. Se leyeron `config.py` y `prompts.py` (documentos 1.1 y 1.2) solo para saber qué valen las constantes importadas con `from ... import *`. Marcas: 📘 *(material del curso)*; ⚠️ *(no contrastado)*; 🔀 *(discrepancia)*; ✅ *(ejecutado)*: el módulo **se ejecutó con modelos y embeddings simulados** (`FakeListChatModel`, `DeterministicFakeEmbedding`) y una base Chroma de prueba con 3 fragmentos, en un entorno con `langchain-core` 1.8.4, `langchain-classic` 1.1.0, `langchain-community` 0.4.2, `langchain-openai` 1.8.5, `chromadb` 1.5.9 y `streamlit` 1.67.0. **No se llamó a OpenAI**: las respuestas del modelo y la calidad de la recuperación no se describen. Lo que se afirma de Streamlit es 📘 o ✅.

>
> **Cambios de proyecto (rutas e importaciones).** Los archivos numerados se importan entre sí desde las versiones `_limpio`: antes de `from config import *` se añadió `import modulos_limpios` (ver `modulos_limpios.py`), que traduce `config`, `prompts` y `rag_system` a `1.1_config_limpio.py`, `1.2_prompts_limpio.py` y `1.4_rag_system_limpio.py`. Hace falta porque un módulo cuyo nombre empieza por un número y contiene un punto no se puede importar con `import` normal. Los archivos sin número (`app.py`, `config.py`…) se importan de la forma habitual. `CHROMA_DB_PATH` (definido en `config.py` / `1.1_config_limpio.py`) es `/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db`. Además se añadió `load_dotenv()` (con `from dotenv import load_dotenv`) tras las importaciones, para que `OpenAIEmbeddings` y `ChatOpenAI` lean `OPENAI_API_KEY` del `.env`: el original no lo cargaba. **Los números de línea de este documento se refieren al `rag_system.py` anterior a estos cambios.** ✅ Probado de punta a punta (build → consulta → app con `streamlit.testing`) con modelos simulados en la estructura `…/Tema3/Asistente_Legal_RAG` y `…/Tema3/contratos`; no con la API real.

---

## Antes de empezar

### Qué vas a aprender

1. Cómo se construye un sistema RAG con LangChain: base vectorial, retrievers, prompt, modelo y parser.
2. Qué es un `MultiQueryRetriever` y un `EnsembleRetriever`.
3. Cómo se escribe la cadena con LCEL (`|`) usando un diccionario.
4. Por qué este archivo **no funciona tal cual con LangChain 1.x** y qué hay que cambiar.

### Qué necesitas

- Documentos 1.3 de Tema_1 (LCEL), 3.1 de Tema_3 (`Document` y metadatos), 1.1 (constantes) y 1.2 (prompts).
- Una base Chroma ya creada con fragmentos de contratos y una clave de OpenAI para ejecutarlo de verdad.

### Mapa del tema

```text
pregunta ──► final_retriever ──► format_docs ──► {context}
   │              ▲                                   │
   │   híbrido: EnsembleRetriever([MMR+MultiQuery, similitud], pesos 0.7/0.3)
   │   si no:   MMR + MultiQuery                       ▼
   └──────────────────────────────► {question} ──► prompt (RAG_TEMPLATE) ──► gpt-4o ──► texto

query_rag(pregunta) → (respuesta, fragmentos para mostrar)    get_retriever_info() → dict
```

### Glosario

| Término | Significado |
|---|---|
| **RAG** | El modelo responde con apoyo en fragmentos recuperados de documentos. |
| **Retriever** | Objeto que, dada una pregunta, devuelve `Document`s relevantes. |
| **`MultiQueryRetriever`** | Usa un LLM para generar varias versiones de la consulta y recuperar con cada una. (Según la referencia oficial: genera variantes con el prompt indicado; cómo une resultados no lo describe la página consultada.) |
| **`EnsembleRetriever`** | Combina los resultados de varios retrievers con *weighted Reciprocal Rank Fusion* (referencia oficial). |
| **MMR** | Búsqueda que equilibra relevancia y diversidad (ver 1.1). |
| **`RunnablePassthrough`** | Paso de LCEL que devuelve su entrada sin cambios. |
| **`StrOutputParser`** | Convierte la respuesta del modelo (`AIMessage`) en un `str`. |
| **`st.cache_resource`** | Decorador de Streamlit que guarda el resultado de una función para reutilizarlo. 📘 ✅ |

---

## Tutorial

### Preparación

Este módulo no se ejecuta solo: lo importa `app.py`. Para ejecutarlo hace falta una base Chroma y `OPENAI_API_KEY`.

---

### Lección 1: importaciones (líneas 1 a 11)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st

from config import *
from prompts import *
```

| Línea | Qué hace | Estado con LangChain 1.x ✅ |
|---|---|---|
| `Chroma` de `langchain_chroma` | Base vectorial | **Cambiado respecto al original:** el original importaba `Chroma` de `langchain_community`, clase **deprecada** en 0.2.9 ✅. Con `langchain_chroma` (como en la documentación oficial) ✅ el retriever MMR y el de similitud funcionan igual y se abre una base creada con la clase antigua (probado con embeddings simulados). `pip install -U langchain-chroma`. `langchain-community` además avisa de que se está retirando. |
| `OpenAIEmbeddings, ChatOpenAI` | Embeddings y chat | Funcionan |
| `from langchain_core.prompts import PromptTemplate` | Plantillas | **Cambiado respecto al original** (`from langchain.prompts …`, 🔀 **`ModuleNotFoundError`** en 1.x ✅). Funciona ✅ |
| `from langchain_classic.retrievers.multi_query import MultiQueryRetriever` | Multi-consulta | **Cambiado respecto al original** (`langchain.retrievers.multi_query`, 🔀 `ModuleNotFoundError` en 1.x ✅). Requiere `pip install -U langchain-classic`. Funciona ✅ |
| `from langchain_classic.retrievers import EnsembleRetriever` | Combinación | **Cambiado respecto al original** (`langchain.retrievers`, 🔀 `ModuleNotFoundError` en 1.x ✅). Funciona ✅ |
| `RunnablePassthrough`, `StrOutputParser` | LCEL | Funcionan |
| `from config import *` / `from prompts import *` | Traen todas las constantes | Ver 1.1 y 1.2 |

**Conclusión** ✅: el archivo original estaba escrito para **LangChain 0.x**. Se actualizaron tres importaciones (más `Chroma` y la llamada a `.invoke`) y ahora funciona con LangChain 1.x. Requiere `pip install -U langchain-classic langchain-chroma`.

---

### Lección 2: función de inicialización y base vectorial (líneas 13 a 20)

```python
@st.cache_resource
def initialize_rag_system():

    # Vector Store
    vectorestore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),\
        persist_directory=CHROMA_DB_PATH
    )
```

| Parte | Qué hace |
|---|---|
| `@st.cache_resource` | La función se ejecuta **una vez**; las siguientes llamadas devuelven el mismo objeto. ✅ (`initialize_rag_system() is initialize_rag_system()` → `True`) |
| `Chroma(embedding_function=..., persist_directory=...)` | Abre la base guardada en disco con ese modelo de embeddings. |
| `\` al final de la línea 18 | Continuación de línea innecesaria (dentro de paréntesis); el archivo compila igualmente ✅. |

⚠️ La variable se llama `vectorestore` (falta una "r"): solo es un nombre, pero conviene saberlo al leer el resto del código.

---

### Lección 3: modelos (líneas 22 a 24)

```python
    # Modelos
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)
```

Dos modelos: uno barato para **reescribir consultas** (`gpt-4o-mini`) y otro para **redactar** (`gpt-4o`). `temperature=0` en ambos. 📘 (valores en 1.1)

---

### Lección 4: retrievers (líneas 26 a 50)

```python
    # Retriever MMR (Maximal Margin Relevance)
    base_retriever = vectorestore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    # Retriever adicional con similarity para comparar
    similarity_retriever = vectorestore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K}
    )

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )
```

| Objeto | Qué es |
|---|---|
| `base_retriever` | Búsqueda MMR (`k=2`, `fetch_k=20`, `lambda_mult=0.7`). |
| `similarity_retriever` | Búsqueda por similitud simple (`k=2`). |
| `multi_query_prompt` | `MULTI_QUERY_PROMPT` convertido en plantilla ✅ (variable `question`). |
| `mmr_multi_retriever` | MMR + generación de variantes de consulta con `llm_queries`. |

- 🔀 El comentario dice "Maximal **Margin** Relevance"; el nombre es *Maximal **Marginal** Relevance* ✅ (docstring de `Chroma.max_marginal_relevance_search` instalado).
- La documentación oficial de Chroma muestra `search_type="mmr"` con `k` y `fetch_k`; `lambda_mult` aparece en la documentación del método instalado ✅ y no en esa página. ⚠️
- La firma oficial de `MultiQueryRetriever.from_llm` es `(retriever, llm, prompt, parser_key=None, include_original=False)` ✅: coincide con el uso.

---

### Lección 5: retriever híbrido (líneas 52 a 61)

```python
    # Ensemble Retriever que combinar MMR y similarity
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

- Con `ENABLE_HYBRID_SEARCH = True`, `final_retriever` combina los dos retrievers con pesos 0.7 y 0.3 (los pesos están escritos a mano aquí, no en `config.py`).
- 🔀 `similarity_threshold` **no es un parámetro** de `EnsembleRetriever`: la referencia oficial lista `retrievers`, `weights`, `c` e `id_key`, y ✅ la clase instalada ignora los extras (`extra="ignore"`): no da error y **no tiene efecto**.

---

### Lección 6: formatear fragmentos y construir la cadena (líneas 63 a 94)

```python
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Funcion para formatear y preprocesar los documentos recuperados
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

**`format_docs`**: convierte la lista de `Document` en un texto con encabezados. ✅ Ejemplo observado: para un fragmento con `source="C:\x\c.pdf"` y `page=0`, produce

```text
[Fragmento 1] - Fuente: c.pdf - Pagina: 0
texto

[Fragmento 2]
otro
```

(la fuente se reduce al nombre del archivo solo si la ruta contiene `\`; el texto se recorta con `.strip()`; sin metadatos solo lleva `[Fragmento N]`). El número de página es el que traiga `metadata["page"]`; con `PyPDFLoader` empieza en 0 (documento 3.1).

**La cadena** (LCEL):

| Paso | Qué hace |
|---|---|
| `{"context": ..., "question": RunnablePassthrough()}` | Un diccionario dentro de una cadena crea dos ramas en paralelo ✅ (la cadena resultante es una `RunnableSequence`). |
| `"context": final_retriever \| format_docs` | La pregunta va al retriever y sus documentos se formatean a texto. |
| `"question": RunnablePassthrough()` | La pregunta pasa tal cual. |
| `\| prompt` | Rellena `RAG_TEMPLATE` con `context` y `question`. |
| `\| llm_generation` | `gpt-4o` redacta. |
| `\| StrOutputParser()` | Devuelve `str`. |

**Comprobado** ✅ (modelos simulados): `rag_chain.invoke("¿Cuál es la renta?")` devuelve un `str`.

⚠️ **La función devuelve `rag_chain` y `mmr_multi_retriever`**, no `final_retriever`. Con la búsqueda híbrida, la respuesta se genera con el *ensemble*, pero los fragmentos que `query_rag` muestra salen del retriever MMR + MultiQuery (ver Lección 7): pueden no coincidir con los usados para responder.

---

### Lección 7: `query_rag` (líneas 97 a 122)

```python
def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener respuesta
        response = rag_chain.invoke(question)

        # Obtener documentos para mostrarlos
        docs = retriever.invoke(question)

        # Formatear los documentos para mostrar
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

| Paso | Qué hace |
|---|---|
| `initialize_rag_system()` | Obtiene la cadena y el retriever (cacheados). |
| `rag_chain.invoke(question)` | Respuesta de texto. |
| `retriever.invoke(question)` | Segunda recuperación, **otra vez con MultiQuery** (otra llamada al modelo de consultas), solo para mostrar fragmentos. |
| `docs[:SEARCH_K]` | Muestra como máximo 2. |
| `doc_info` | Diccionario con las claves que `app.py` espera: `fragmento`, `contenido` (recortado a 1000 caracteres + `...`), `fuente` (nombre de archivo) y `pagina` (`'No especificada'` si falta). |
| `except Exception` | Cualquier error se devuelve **como texto de respuesta** con lista de fragmentos vacía. |

**Comprobado** ✅ (versión con `.invoke`, modelos simulados): `query_rag("¿renta?")` devolvió `('Respuesta', [{'fragmento': 1, 'contenido': ..., 'fuente': 'contratoC.pdf', 'pagina': 5}])`: coincide con lo que consume `app.py`.

🔀 **Cambio respecto al original: `get_relevant_documents` ya no existe.** El original lo llamaba; estaba marcado como obsoleto desde `langchain-core` 0.1.46 con eliminación en 1.0 (✅ visible en `langchain-core` 0.3.80) y en `langchain-core` 1.8.4 ✅ no existe. Con el original (más las tres importaciones corregidas) la cadena respondía, pero `query_rag` terminaba en el `except` y devolvía:

```text
Error al procesar la cosulta: 'MultiQueryRetriever' object has no attribute 'get_relevant_documents'
```

con la lista de fragmentos vacía: **el usuario veía ese texto como "respuesta"** (y el modelo se había llamado ya, con su coste). Ahora el código usa `retriever.invoke(question)` ✅: con modelos simulados y una base de prueba, `query_rag` devuelve respuesta y 2 fragmentos con las claves `fragmento`, `contenido`, `fuente` y `pagina`.

---

### Lección 8: `get_retriever_info` (líneas 124 a 132)

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

**Comprobado** ✅: con la configuración de `config.py` devuelve `{'tipo': 'MMR + MultiQuery + Hybrid', 'documentos': 2, 'diversidad': 0.7, 'candidatos': 20, 'umbral': 0.7}`. `app.py` solo usa `'tipo'`. El `umbral` se muestra aunque no tenga efecto real en la búsqueda (🔀 ver Lección 5).

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo hago que funcione con LangChain 1.x?

**Ya está aplicado** en `rag_system.py` y en sus versiones 1.4. Si trabajas con una copia del original, cambia (✅ probado) tres importaciones y una llamada:

```python
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
```

y `retriever.get_relevant_documents(question)` por `retriever.invoke(question)`. `Chroma` debe importarse de `langchain_chroma` (`pip install -U langchain-classic langchain-chroma`).

### ¿Cómo muestro los fragmentos realmente usados?

Devolver `final_retriever` en lugar de `mmr_multi_retriever` (⚠️ cambiaría el código del archivo; solo sugerencia).

### ¿Cómo evito pagar dos veces la reescritura de consultas?

Hoy se hace una vez dentro de la cadena y otra en `query_rag`. Una opción sería obtener los documentos una sola vez y reutilizarlos (sugerencia, no probada).

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: langchain.prompts` / `langchain.retrievers` (forma del original) | LangChain 1.x ✅ | Importaciones de `langchain_core` / `langchain_classic` (ya aplicadas) |
| `ModuleNotFoundError: langchain_classic` | Falta el paquete | `pip install -U langchain-classic` |
| La respuesta es `Error al procesar la cosulta: ... get_relevant_documents` (forma del original) | Método eliminado ✅ | `retriever.invoke(...)` (ya aplicado) |
| `LangChainDeprecationWarning` por `Chroma` (solo si se vuelve a importar de `langchain_community`) | Clase deprecada ✅ | Usar `langchain_chroma` |
| `ModuleNotFoundError: langchain_chroma` | Falta el paquete | `pip install -U langchain-chroma` |
| El umbral no tiene efecto | Parámetro inexistente en `EnsembleRetriever` ✅ | No confiar en él |
| Fragmentos mostrados distintos a los usados | `query_rag` usa otro retriever ✅ | Ver arriba |

---

## Referencia

### Funciones del módulo

| Función | Entrada | Salida | Notas |
|---|---|---|---|
| `initialize_rag_system()` | — | `(rag_chain, mmr_multi_retriever)` | Cacheada con `st.cache_resource` ✅ |
| `query_rag(question)` | `str` | `(respuesta: str, docs_info: list[dict])` | Nunca lanza excepciones: devuelve el error como texto |
| `get_retriever_info()` | — | `dict` (5 claves) | `app.py` usa `'tipo'` |

### Claves de `docs_info`

| Clave | Contenido |
|---|---|
| `fragmento` | Número desde 1 |
| `contenido` | Texto (máx. 1000 caracteres + `...`) |
| `fuente` | Nombre de archivo |
| `pagina` | `metadata["page"]` o `'No especificada'` |

### Componentes LangChain

| Componente | Paquete (1.x) | Notas |
|---|---|---|
| `Chroma` | `langchain_chroma` (usado ahora) / `langchain_community` (deprecado, forma del original) | |
| `PromptTemplate` | `langchain_core.prompts` | |
| `MultiQueryRetriever` | `langchain_classic.retrievers.multi_query` | `from_llm(retriever, llm, prompt, parser_key, include_original)` |
| `EnsembleRetriever` | `langchain_classic.retrievers` | `retrievers`, `weights`, `c`, `id_key` |
| `RunnablePassthrough`, `StrOutputParser` | `langchain_core` | |

---

## Explicación

### ¿Por qué un MultiQueryRetriever?

Para recuperar con varias formulaciones de la pregunta y reducir el riesgo de que una redacción concreta no encuentre el fragmento. 📘

### ¿Por qué combinar MMR y similitud?

La idea del curso es dar más peso (0.7) a MMR y algo a la similitud simple (0.3). 📘 No se midió el efecto.

### ¿Por qué `query_rag` captura todo?

Para que `app.py` nunca reciba una excepción. A cambio, los errores se muestran como si fueran la respuesta del asistente. ✅

### Lo que este documento no cubre

- Cómo se creó la base Chroma ni su contenido.
- La calidad de las respuestas o de la recuperación (no se llamó a OpenAI).
- Las entradas de `Document` reales de tus contratos.

### Observaciones sobre el archivo

- El original estaba escrito para LangChain 0.x; con 1.x fallaban tres importaciones y `get_relevant_documents`. 🔀 ✅ Corregido en esta versión.
- `similarity_threshold` se ignora. 🔀 ✅
- Los fragmentos mostrados proceden de un retriever distinto al usado para responder, con una segunda llamada MultiQuery.
- `Maximal Margin`, `vectorestore`, `cosulta`, "que combinar": erratas en comentarios o nombres.
- Los errores se devuelven como texto de respuesta.
- Los pesos `[0.7, 0.3]` están escritos a mano; el resto de parámetros está en `config.py`.

---

## Autoevaluación

1. ¿Qué devuelve `initialize_rag_system()` y por qué se cachea?
2. ¿Qué hace el diccionario al inicio de `rag_chain`?
3. ¿Qué dos modelos se usan y para qué?
4. ¿Qué pasa con `similarity_threshold`?
5. ¿Qué retriever se usa para mostrar fragmentos y cuál para responder (con búsqueda híbrida)?
6. ¿Qué ocurría con el original en LangChain 1.x porque `retriever.get_relevant_documents` no existe?
7. ¿Qué tres importaciones del original fallan en LangChain 1.x?
8. ¿Qué claves tiene cada elemento de `docs_info`?
9. ¿Por qué la respuesta de error aparece como respuesta del asistente?

### Respuestas

1. `(rag_chain, mmr_multi_retriever)`; se cachea para no recrear la base y los modelos en cada reejecución ✅.
2. Crea dos ramas: `context` (retriever + `format_docs`) y `question` (pregunta sin cambios).
3. `gpt-4o-mini` para reescribir consultas; `gpt-4o` para generar la respuesta.
4. Se ignora: `EnsembleRetriever` no tiene ese parámetro ✅.
5. Para mostrar, `mmr_multi_retriever`; para responder, el ensemble.
6. Salta el `except` y `query_rag` devuelve un texto de error con lista vacía ✅.
7. `langchain.prompts`, `langchain.retrievers.multi_query` y `langchain.retrievers` ✅.
8. `fragmento`, `contenido`, `fuente`, `pagina`.
9. Porque `query_rag` captura la excepción y la devuelve como primer valor `(texto, [])`.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Cadena RAG | `{context, question} \| prompt \| gpt-4o \| StrOutputParser` |
| MultiQuery + MMR | Variantes de consulta + búsqueda diversa |
| Ensemble | Combina MMR y similitud con pesos 0.7/0.3 |
| Compatibilidad | Escrito para 0.x; 1.x exige cambios ✅ |
| `query_rag` | `(respuesta, fragmentos)`; errores como texto |

---

## Fuentes consultadas

- Chroma: <https://docs.langchain.com/oss/python/integrations/vectorstores/chroma>
- `EnsembleRetriever` (referencia): <https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever>
- `MultiQueryRetriever.from_llm` (referencia): <https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever/from_llm>
- Material del tema: `rag_system.py`, `1.4_rag_system.py`, `1.4_rag_system_limpio.py`, y (solo para ver los valores) `config.py` y `prompts.py`, en `Tema_3/Asistente_Legal_RAG/`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
