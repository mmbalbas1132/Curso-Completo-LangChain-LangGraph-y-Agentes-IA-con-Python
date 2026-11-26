# 3.8 MultiQueryRetriever: buscar con varias reformulaciones de la pregunta

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.8_multi_query_retriever.py` (versión comentada; su código es idéntico al de `3.8_multi_query_retriever_limpio.py`, ✅ comprobado) y de la documentación oficial de LangChain. El original se conserva en `3.8_multi_query_retriever_v1.py` (ya incluía `load_dotenv()` y la ruta de 3.6). **Dos cambios deliberados sobre el original** (pedidos para actualizar el script): `Chroma` se importa de `langchain_chroma` y `MultiQueryRetriever` de `langchain_classic.retrievers.multi_query`. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = lo observado o la documentación difiere del archivo; ✅ *(ejecutado o comprobado en el código instalado)* = con `langchain` 1.5.2, `langchain-classic` 1.1.0, `langchain-core` 1.8.4, `langchain-chroma` 1.2.0 y `chromadb` 1.5.9.
>
> **Alcance y privacidad.** El script necesita la API de OpenAI y la base `chroma_db` de 3.6 (con datos personales de los contratos). Yo lo ejecuté con **modelos simulados** (embeddings y modelo de chat falsos) y una base **temporal** con textos de prueba (✅); tu `chroma_db` no se tocó. **Pendiente: la salida real con la API**; la lección 6 marca dónde irá.

---

## Antes de empezar

### Qué vas a aprender

1. Por qué la redacción de la pregunta afecta a la búsqueda por similitud.
2. Qué hace `MultiQueryRetriever`: reescribir la pregunta con un LLM, buscar con cada versión y unir los resultados.
3. Cómo construirlo con `from_llm` a partir de un retriever base y un modelo de chat.
4. Qué cambió en las importaciones con langchain 1.x.
5. Qué cuesta (llamadas a la API) y qué no hace (no busca la pregunta original por defecto).

### Qué necesitas

- Haber visto 3.6 (vector stores) y 3.7 (retrievers), y haber **ejecutado 3.6** para que exista `chroma_db`.
- `pip install -U langchain-chroma langchain-classic langchain-openai chromadb python-dotenv`.
- `OPENAI_API_KEY` en `.env` (sirve para embeddings y para el modelo de chat).

### Mapa del tema

```text
consulta (texto)
   └─ LLM (gpt-4o-mini) ─► N preguntas alternativas        ← 1 llamada al modelo de chat
        ├─ pregunta 1 ─► embedding ─► base Chroma ─► 2 docs
        ├─ pregunta 2 ─► embedding ─► base Chroma ─► 2 docs
        └─ pregunta 3 ─► embedding ─► base Chroma ─► 2 docs
                     └─ unión sin repetidos ─► lista de Document (hasta 6)
```

### Glosario

| Término | Significado |
|---|---|
| **MultiQueryRetriever** | Retriever que genera varias versiones de la pregunta con un LLM, busca con cada una y devuelve la unión de resultados. |
| **Retriever base** | El retriever que hace la búsqueda real (aquí, el de 3.7 con `k=2`). |
| **LLM / modelo de chat** | Aquí `ChatOpenAI`, que reescribe la pregunta. |
| **Unión única** | Lista con los documentos de todas las búsquedas, sin repetir los que son iguales (mismo contenido y metadatos ✅). |
| **`include_original`** | Parámetro de `from_llm`; si es `True`, la pregunta original también se busca. Por defecto `False` ✅. |
| **`langchain-classic`** | Paquete donde viven ahora los retrievers "clásicos" que ya no están en `langchain` 1.x. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-chroma langchain-classic langchain-openai chromadb python-dotenv
```

Ejecuta antes 3.6 (crea `chroma_db`) y comprueba que `.env` tiene `OPENAI_API_KEY`.

### Lección 1: importaciones

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
```

- 🔀 **`Chroma` de `langchain_chroma`** (el original: `langchain_community`, obsoleta ✅; ver 3.6).
- 🔀 **`MultiQueryRetriever` de `langchain_classic`.** El original importaba de `langchain.retrievers.multi_query`; con `langchain` 1.5.2 ✅ ese módulo no existe (`ModuleNotFoundError: No module named 'langchain.retrievers'`). La página de referencia lo documenta como `langchain_classic.retrievers.multi_query.MultiQueryRetriever`. ⚠️ La página consultada no dice si `langchain-classic` está o no en proceso de retirada.
- `OpenAIEmbeddings` y `ChatOpenAI` se importan en dos líneas del mismo paquete; es válido (ver sugerencia 1).

### Lección 2: cargar el `.env`

```python
from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()
```

Ya venía en el original. Hace que `OpenAIEmbeddings` y `ChatOpenAI` encuentren la clave (ver 3.3).

### Lección 3: abrir la base

```python
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)
```

Igual que en 3.7: **abre** la base de 3.6 con el mismo modelo de embeddings. ✅ Con una ruta inexistente, Chroma crea una carpeta vacía y las búsquedas devuelven `[]` sin error.

### Lección 4: el modelo de chat

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

| Código | Qué hace |
|---|---|
| `model="gpt-4o-mini"` | Modelo de OpenAI que reescribirá la pregunta. |
| `temperature=0` | Reduce la aleatoriedad de la respuesta. ⚠️ No garantiza preguntas idénticas entre ejecuciones (no comprobado con la API). |

### Lección 5: el retriever múltiple

```python
base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
```

`from_llm` (✅ según el código instalado):

- monta la cadena *plantilla de prompt → modelo → analizador de líneas*;
- usa por defecto `DEFAULT_QUERY_PROMPT`, que pide **3 versiones distintas** de la pregunta, separadas por saltos de línea (la plantilla está en inglés);
- el analizador divide la respuesta por líneas y descarta las vacías (✅ con una respuesta falsa `"A\n\nB\nC\n"` salieron 3 preguntas);
- `include_original=False` por defecto: **la pregunta original no se busca** (✅ solo se buscaron las 3 generadas).

La página oficial de `from_llm` documenta `retriever`, `llm`, `prompt` (por defecto `DEFAULT_QUERY_PROMPT`), `parser_key` y `include_original` (sin descripción las dos últimas), y no avisa de ninguna obsolescencia del método.

### Lección 6: consultar y mostrar

```python
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

print("Top documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
```

Qué ocurre en `invoke` (✅ comprobado con modelos simulados):

1. 1 llamada al modelo de chat → N preguntas.
2. 1 embedding por pregunta (con 3 preguntas, 3).
3. 2 documentos por pregunta (`k=2`) → hasta 6; se eliminan los repetidos (✅ en la prueba, 6 → 5).

El título ya no dice «Top 2»: el número de resultados varía.

⚠️ **Salida real: pendiente.** Cuando ejecutes el script, pásame la salida y la añado aquí (sin reproducir datos personales).

---

## Guía práctica

### ¿Cómo ejecuto el script?

1. Ejecuta 3.6 si no existe `chroma_db`.
2. Comprueba la ruta de `persist_directory` y la clave en `.env`.
3. Ejecuta. Coste: 1 llamada al modelo de chat + ~3 embeddings (más que 3.7, que hace 1).

### ¿Cómo veo las preguntas que genera el modelo?

El retriever tiene `verbose=True` por defecto y registra las preguntas con `logger.info` (✅), pero solo se ven si activas el logging en nivel INFO. Antes de `invoke`:

```python
import logging
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
```

✅ Con esto se imprimió `Generated queries: [...]`. Este código no está en el script (sugerencia 3).

### ¿Cómo hago que también se busque la pregunta original?

```python
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm, include_original=True)
```

✅ Con `include_original=True` se buscaron las 3 generadas **y** la original.

### ¿Cómo uso mi propia plantilla (por ejemplo, en español)?

`from_llm` acepta `prompt=` (página oficial). La plantilla debe usar la variable `{question}` (✅ es la que recibe la cadena). ⚠️ No se probó una plantilla propia.

### Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain.retrievers'` ✅ | Importación antigua con langchain 1.x | `from langchain_classic.retrievers.multi_query import MultiQueryRetriever` y `pip install -U langchain-classic`. |
| `ModuleNotFoundError: langchain_classic` | Falta el paquete | `pip install -U langchain-classic`. |
| No sale ningún documento ✅ | Ruta de `chroma_db` equivocada (carpeta vacía creada) | Corrige la ruta (ver 3.7). |
| `OpenAIError: Missing credentials…` | Falta `OPENAI_API_KEY` | Ponla en `.env`. |
| `AuthenticationError 401` | Clave no válida o caducada, o una `OPENAI_API_KEY` vieja ya definida | Clave nueva y revisar el entorno. |
| No veo las preguntas generadas | El logging no está en INFO ✅ | Ver «¿Cómo veo las preguntas…?». |

---

## Referencia

### Importaciones

| Importación | Paquete | Para qué |
|---|---|---|
| `Chroma` | `langchain_chroma` | Base vectorial |
| `OpenAIEmbeddings` | `langchain_openai` | Embeddings |
| `ChatOpenAI` | `langchain_openai` | Modelo de chat |
| `MultiQueryRetriever` | `langchain_classic.retrievers.multi_query` (🔀 antes `langchain.retrievers.multi_query`) | Retriever múltiple |
| `load_dotenv` | `dotenv` | Cargar el `.env` |

### `MultiQueryRetriever.from_llm`

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `retriever` | obligatorio | Retriever base que hace la búsqueda |
| `llm` | obligatorio | Modelo que genera las preguntas |
| `prompt` | `DEFAULT_QUERY_PROMPT` | Plantilla para generar las versiones de la pregunta |
| `parser_key` | `None` | Obsoleto ✅ (no se usa; el código instalado lo marca como deprecated) |
| `include_original` | `False` | Si `True`, también se busca la pregunta original |

Atributo `verbose` (por defecto `True` ✅): registra las preguntas generadas con `logger.info`.

### Comparativa 3.7 / 3.8

| | 3.7 (`as_retriever`) | 3.8 (`MultiQueryRetriever`) |
|---|---|---|
| Búsquedas por consulta | 1 | 1 por cada pregunta generada (~3) |
| Llamadas a embeddings | 1 | ~3 |
| Llamada al modelo de chat | No | 1 |
| Nº de resultados | `k` (2) | Hasta `k × nº de preguntas`, sin repetidos |
| Busca la pregunta original | Sí | No (salvo `include_original=True`) ✅ |
| Paquete | `langchain_chroma` | `langchain_classic` |

---

## Explicación

### ¿Por qué reformular la pregunta?

La plantilla por defecto lo explica: generar varias perspectivas de la pregunta busca ayudar a superar limitaciones de la búsqueda por similitud basada en distancia (✅ texto de `DEFAULT_QUERY_PROMPT`). Una pregunta con la redacción poco afortunada puede dejar fuera fragmentos relevantes que otra redacción sí encontraría.

### ¿Por qué no busca la pregunta original?

Es el valor por defecto de `include_original` (✅ `False`). Si la original ya era buena y las reformulaciones se alejan de ella, podrías perder resultados: por eso existe el parámetro. ⚠️ Que convenga activarlo depende del caso; no se ha comprobado con tus datos.

### ¿Qué coste tiene?

Más llamadas a la API que 3.7: 1 al modelo de chat y ~3 de embeddings por consulta (✅ con modelos simulados). Además añade latencia (las búsquedas se hacen una tras otra en la versión síncrona ✅ según el código).

### ¿Qué enfoque usar en cada caso?

| Situación | Enfoque |
|---|---|
| Preguntas claras, base pequeña | `as_retriever` simple (3.7) |
| Preguntas ambiguas o redactadas de formas muy variadas | `MultiQueryRetriever` |
| Quieres que la pregunta original cuente | `include_original=True` |
| Necesitas preguntas en español/adaptadas | `prompt=` propio |

### Lo que este documento no cubre

- La calidad real de las reformulaciones y de los resultados (falta la salida real).
- Plantillas propias y `include_original=True` con la API real.
- Otros retrievers (ensemble, etc.) y la cadena RAG completa: proyecto `Asistente_Legal_RAG`.
- La versión asíncrona (`ainvoke`).
- El contenido de los contratos.

### Observaciones sobre el archivo

1. 🔀 El original importaba `MultiQueryRetriever` de `langchain.retrievers.multi_query`, que ya no existe en langchain 1.x ✅; con `langchain_classic` funciona.
2. 🔀 El original importaba `Chroma` de `langchain_community` (obsoleta ✅).
3. El original ya cargaba el `.env` y usaba la ruta de 3.6 (arreglos previos del alumno).
4. La plantilla por defecto está en inglés ✅; el modelo puede responder igualmente en español, pero no se ha comprobado.
5. El script depende de haber ejecutado 3.6 y no lo comprueba.
6. Dos importaciones separadas de `langchain_openai`; variable `i` sin usar; «mas» sin tilde.
7. El código (salvo las dos importaciones) se respetó: comentado y limpio tienen el mismo AST ✅.

---

## Autoevaluación

1. ¿Qué problema de la búsqueda por similitud intenta resolver `MultiQueryRetriever`?
2. ¿Qué hace `from_llm` con el retriever y el modelo que le pasas?
3. ¿Cuántas versiones de la pregunta pide la plantilla por defecto y quién decide cuántas salen realmente?
4. Con `k=2` y 3 preguntas, ¿cuál es el máximo de documentos devueltos y por qué puede ser menor?
5. ¿Se busca la pregunta original por defecto? ¿Cómo se cambia?
6. ¿Cuántas llamadas a la API hace una consulta, aproximadamente?
7. ¿Qué error da la importación antigua con langchain 1.x y cómo se arregla?
8. ¿Cómo puedes ver las preguntas generadas?
9. ¿Por qué el título del script ya no dice «Top 2»?

### Respuestas

1. Que el resultado dependa demasiado de cómo está redactada la pregunta.
2. Monta una cadena (plantilla → modelo → analizador de líneas) para generar preguntas, y las usa con el retriever base.
3. Pide 3, pero el número real lo decide el modelo.
4. 6; puede ser menor porque los documentos repetidos se eliminan.
5. No (`include_original=False`); se activa con `include_original=True`.
6. 1 al modelo de chat y ~3 de embeddings (una por pregunta generada).
7. `ModuleNotFoundError: No module named 'langchain.retrievers'`; se importa de `langchain_classic.retrievers.multi_query`.
8. Activando el logging INFO del logger `langchain_classic.retrievers.multi_query`.
9. Porque el número de resultados varía según las preguntas generadas y los repetidos.

---

## Resumen

| Concepto | Idea clave |
|---|---|
| `MultiQueryRetriever` | LLM reformula la pregunta → varias búsquedas → unión sin repetidos |
| `from_llm(retriever, llm)` | Construye el retriever con la plantilla por defecto (3 versiones, en inglés) |
| `include_original` | `False` por defecto: la pregunta original no se busca |
| Coste | 1 llamada al LLM + ~3 embeddings por consulta |
| Importación | `langchain_classic.retrievers.multi_query` (la antigua falla en langchain 1.x ✅) |
| Pendiente | Salida real con la API |

---

## Fuentes consultadas

- `MultiQueryRetriever`: <https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever>
- `MultiQueryRetriever.from_llm`: <https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever/from_llm>
- Archivos de origen: `3.8_multi_query_retriever.py` (comentado), `3.8_multi_query_retriever_limpio.py` (limpio) y `3.8_multi_query_retriever_v1.py` (original). Comprobaciones de la plantilla, el analizador, `include_original`, `verbose` y `parser_key` en el código instalado de `langchain-classic` (✅).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
