# 3.7 Retrievers: consultar la base vectorial con una interfaz estándar

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.7_retrievers_langchain.py` (versión comentada; su código es idéntico al de `3.7_retrievers_langchain_limpio.py`, ✅ comprobado) y de la documentación oficial de LangChain. El original se conserva en `3.7_retrievers_langchain_v1.py`. **Tres cambios deliberados sobre el original** (pedidos para actualizar el script): (1) `Chroma` se importa de `langchain_chroma` en lugar de `langchain_community`; (2) se añade `load_dotenv()`; (3) la ruta de la base pasa a ser la de 3.6, porque la original era una carpeta de Windows de otro usuario. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = lo observado o la documentación difiere del archivo; ✅ *(ejecutado)* = comportamiento observado con `langchain-core` 1.8.4, `langchain-chroma` 1.2.0, `langchain-openai` 1.8.5, `chromadb` 1.5.9 y `python-dotenv` 1.2.3.
>
> **Alcance y privacidad.** El script necesita la API de OpenAI y la base `chroma_db` de 3.6, que contiene datos personales de los contratos. Yo lo ejecuté con **embeddings simulados** y una base **temporal** con textos de prueba (✅); tu `chroma_db` no se tocó. La **salida real** con la API de OpenAI la ejecutó el alumno (código de salida 0) y se resume en la lección 5 sin reproducir el contenido de los contratos.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un *retriever* y en qué se diferencia de un vector store.
2. Abrir una base Chroma que ya existe (sin volver a crearla).
3. Convertirla en retriever con `as_retriever` y consultarla con `invoke`.
4. Un fallo silencioso típico: una ruta equivocada devuelve 0 resultados sin error.

### Qué necesitas

- Haber visto 3.5 (embeddings) y 3.6 (vector stores), y haber **ejecutado 3.6** para que exista `chroma_db`.
- `pip install -U langchain-chroma langchain-openai chromadb python-dotenv`.
- `OPENAI_API_KEY` en un archivo `.env` (el script usa `load_dotenv()`).

### Mapa del tema

```text
chroma_db (creada en 3.6)
   └─ Chroma(embedding_function, persist_directory)      ← abre la base
        └─ as_retriever(search_type="similarity", k=2)   ← la convierte en retriever
             └─ retriever.invoke(consulta)               ← 1 llamada a la API (vector de la consulta)
                  └─ [Document, Document]                ← los 2 más parecidos
```

### Glosario

| Término | Significado |
|---|---|
| **Retriever** | Interfaz que recibe una consulta (texto) y devuelve una lista de `Document`. La página oficial dice que es más general que un vector store. |
| **Vector store** | Base de datos de vectores (ver 3.6). Todos los vector stores pueden convertirse en retrievers (página oficial). |
| **`as_retriever`** | Método del vector store que crea un retriever a partir de él. |
| **`search_type`** | Tipo de búsqueda del retriever: ✅ `similarity` (por defecto), `similarity_score_threshold` o `mmr`. |
| **`search_kwargs`** | Diccionario de argumentos de la búsqueda; aquí `{"k": 2}`. |
| **`invoke`** | Método para ejecutar un retriever (y cualquier `Runnable`) con una entrada. |
| **Runnable** | Componente estándar de LangChain con `invoke`; ✅ un retriever lo es. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-chroma langchain-openai chromadb python-dotenv
```

Pon `OPENAI_API_KEY=tu_clave` en `.env` (ver 3.5) y ejecuta antes 3.6 para crear la carpeta `chroma_db`.

### Lección 1: importaciones y carga del `.env`

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()
```

- 🔀 **`Chroma` viene ahora de `langchain_chroma`.** El original la importaba de `langchain_community`, donde ✅ la clase está obsoleta (aviso `deprecated in LangChain 0.2.9` al activar todos los avisos). La página oficial de Chroma usa `langchain_chroma`.
- ✅ Al no importar nada de `langchain_community`, este script **no** muestra el aviso de retirada de ese paquete (a diferencia de 3.6, que sí importa un cargador de ahí). Confirmado en la ejecución real del alumno: la salida empieza directamente con la cabecera, sin avisos.
- `load_dotenv()` (añadido) lee `OPENAI_API_KEY` del `.env` sin sobrescribir variables ya definidas (ver 3.3).

### Lección 2: abrir la base existente

```python
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)
```

| Código | Qué hace |
|---|---|
| `Chroma(...)` | **Abre** la carpeta; no añade documentos (en 3.6 lo hacía `from_documents`). Usa la colección `langchain` por defecto ✅. |
| `embedding_function=` | Modelo para convertir **la consulta** en vector. Aquí el parámetro se llama `embedding_function`; en `from_documents` (3.6) se llamaba `embedding`. |
| `model="text-embedding-3-large"` | Debe ser el **mismo** modelo con el que se creó la base (3.6). |
| `persist_directory=…` | Carpeta de la base. 🔀 El original apuntaba a `C:\Users\santiago\…`, que no es tu equipo; ahora es la ruta de 3.6. |

⚠️ **Dos fallos silenciosos (✅ comprobados):**

1. Si la ruta **no existe**, Chroma **crea una carpeta vacía** y el retriever devuelve `[]` **sin ningún error**. El script imprime solo la cabecera.
2. Con una ruta de Windows (`C:\Users\…`) en Linux se crea una carpeta con ese nombre literal en el directorio actual, y pasa lo mismo: 0 resultados.

Si usas un modelo con otro tamaño de vector que el de la base, Chroma falla (✅ `Collection expecting embedding with dimension of 64, got 32`, con vectores simulados).

### Lección 3: crear el retriever

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
```

- `as_retriever` devuelve ✅ un `VectorStoreRetriever`, que es un `Runnable`.
- `search_type="similarity"`: los más parecidos. Es el valor por defecto ✅.
- `search_kwargs={"k": 2}`: devuelve 2 documentos (sin `k`, ✅ devuelve 4).
- ✅ Con `similarity` y el mismo `k`, el resultado es el mismo que `vectorstore.similarity_search(consulta, k=2)` (como en 3.6, con `k=3`).

### Lección 4: la consulta

```python
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)
```

- `invoke` recibe el texto y devuelve una lista de `Document` (✅ con `k=2`, 2). Hace **una** llamada a la API de embeddings (la de la consulta).
- 🔀 Esta consulta sí cierra con `?`; la de 3.6 no.
- Es la forma actual. El método antiguo `get_relevant_documents` quedó obsoleto (el mismo cambio se aplicó en `1.4_rag_system.py`).

### Lección 5: mostrar los resultados

```python
print("Top 2 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
```

- ✅ Con la base simulada imprime la cabecera y 2 bloques `Contenido / Metadatos`.
- ✅ **Salida real (ejecución del alumno con la API de OpenAI, código de salida 0).** Imprime la cabecera y exactamente **2** bloques (descritos sin reproducir datos personales):

| N.º | Procedencia | Qué contiene | ¿Responde a la pregunta? |
|---|---|---|---|
| 1 | Primera página de `CONTRATO DE ARRENDAMIENTO DE LOCAL DE NEGOCIO 2.pdf` | Identificación de las partes y descripción del local (incluida su ubicación) | **Sí** |
| 2 | Tercera página del mismo PDF | Cierre con firmas y anexos; aparece el nombre de la persona, no la ubicación | Solo indirectamente |

- ✅ Son los **mismos dos primeros resultados, en el mismo orden**, que dio `similarity_search(k=3)` en 3.6. Falta el tercero (el de otro contrato) porque aquí `k=2`: no falta ningún documento, es lo que pide `search_kwargs={"k": 2}`.
- ✅ También confirma que la ruta es correcta: la base tiene datos (con una ruta equivocada saldría solo la cabecera).
- Recuerda la lección de 3.6: los `k` más parecidos no tienen por qué ser todos relevantes; con `k=2` desaparece el resultado flojo, pero el 2.º ya no responde directamente.
- El «Top 2» está escrito a mano: si cambias `k`, el texto no se actualiza.

---

## Guía práctica

### ¿Cómo ejecuto el script?

1. Ejecuta 3.6 (crea `chroma_db`).
2. Comprueba que `persist_directory` apunta a esa carpeta.
3. Ejecuta. Coste: 1 embedding (la consulta).

### ¿Cómo sé que he abierto la base correcta?

Antes de crear el retriever, cuenta los vectores guardados (✅ `get()` es público):

```python
print(len(vectorstore.get()["ids"]))
```

Con la base de 3.6 deberías ver el número de chunks que mostró 3.6 (39 con tu carpeta, o más si hubo duplicados). Si sale 0, la ruta está mal. ⚠️ Este código no está en el script (ver *Sugerencias* en la versión comentada).

### ¿Cómo uso otro tipo de búsqueda?

Cambia `search_type`. La página oficial de Chroma muestra `mmr` con `k` y `fetch_k`:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
```

✅ `similarity_score_threshold` exige además un `score_threshold` de tipo `float` en `search_kwargs`; si falta, falla con `ValueError`.

### ¿Cómo hago la ruta portable?

Usa una ruta relativa o `pathlib` a partir de la ubicación del script (ver 3.3). ⚠️ Forma habitual de Python.

### Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| Solo sale la cabecera, sin resultados ✅ | Ruta inexistente o base vacía: Chroma crea una carpeta nueva sin avisar | Corrige `persist_directory` y comprueba `len(vectorstore.get()["ids"])`. |
| Aparece una carpeta con nombre `C:\Users\…` en tu proyecto ✅ | Ruta de Windows usada en Linux | Bórrala tú y usa la ruta de Linux. |
| `InvalidArgumentError … dimension` ✅ | Modelo de embeddings distinto al de la base | Usa el mismo modelo que 3.6. |
| `OpenAIError: Missing credentials…` | Falta `OPENAI_API_KEY` | Ponla en `.env`. |
| `AuthenticationError 401` | Clave no válida o caducada, o una `OPENAI_API_KEY` vieja ya definida (no la sobrescribe `load_dotenv()`) | Clave nueva y revisar el entorno. |
| `ModuleNotFoundError: langchain_chroma` | Falta el paquete | `pip install -U langchain-chroma`. |
| Resultados repetidos | La base tiene duplicados (ver 3.6) | Reconstruye la base. |

---

## Referencia

### Importaciones

| Importación | Paquete | Para qué |
|---|---|---|
| `Chroma` | `langchain_chroma` (la de `langchain_community.vectorstores` está obsoleta ✅) | Base vectorial |
| `OpenAIEmbeddings` | `langchain_openai` | Embeddings |
| `load_dotenv` | `dotenv` | Cargar el `.env` |

### Métodos y parámetros usados

| Elemento | Detalle |
|---|---|
| `Chroma(embedding_function, persist_directory, collection_name)` | Abre la base; la página oficial muestra los tres argumentos. `collection_name` por defecto `langchain` ✅ |
| `vectorstore.as_retriever(search_type, search_kwargs)` | Devuelve un `VectorStoreRetriever` ✅ |
| `search_type` | `similarity` (por defecto) · `similarity_score_threshold` · `mmr` ✅ |
| `search_kwargs["k"]` | Nº de documentos; por defecto 4 ✅ |
| `retriever.invoke(consulta)` | `list[Document]` ✅ |

### Comparativa con 3.6

| | 3.6 | 3.7 |
|---|---|---|
| Qué hace con la base | La crea (`from_documents`) | La abre (`Chroma(...)`) |
| Cómo busca | `vectorstore.similarity_search(consulta, k=3)` | `retriever.invoke(consulta)` con `k=2` |
| Parámetro del modelo | `embedding=` | `embedding_function=` |
| Coste en la API | Todos los chunks + la consulta | Solo la consulta |
| Aviso de `langchain-community` | Sí (cargador de PDF) | No ✅ |

---

## Explicación

### ¿Por qué un retriever si ya tenemos `similarity_search`?

Porque el retriever es un componente estándar (un `Runnable`): tiene siempre la misma interfaz (texto → lista de `Document`), sea cual sea la fuente de datos. La página oficial dice que los retrievers pueden crearse desde vector stores, pero también desde otras fuentes. Así se puede cambiar el almacén sin cambiar el resto del programa y conectarlo en cadenas, como hace el proyecto `Asistente_Legal_RAG`. 📘⚠️ Esta es la explicación del curso; la página oficial consultada no desarrolla los detalles.

### ¿Por qué abrir la base en lugar de recrearla?

Porque calcular los embeddings de todos los chunks cuesta dinero y tiempo, y duplicaría los vectores (ver 3.6). Se crea una vez (3.6) y aquí solo se consulta.

### ¿Por qué el modelo de embeddings debe coincidir?

Los vectores de la consulta y los de la base deben ser comparables. Con tamaños distintos, ✅ Chroma da error; con el mismo tamaño pero modelos distintos ⚠️ no habría error, pero los resultados no tendrían sentido (explicación general, no comprobada).

### ¿Qué enfoque usar en cada caso?

| Situación | Enfoque |
|---|---|
| Crear la base | 3.6 (`from_documents`) o `build_vectorstore.py` del proyecto |
| Consultar rápido en un script | `vectorstore.similarity_search` |
| Integrar en una cadena RAG | `as_retriever` + `invoke` |
| Resultados más variados | `search_type="mmr"` |
| Descartar resultados poco parecidos | `similarity_score_threshold` con `score_threshold` |

### Lo que este documento no cubre

- La calidad de las búsquedas más allá de este único ejemplo (una consulta, `k=2`).
- `mmr` y `similarity_score_threshold` en funcionamiento (solo se citan).
- Otros tipos de retrievers (multi-consulta, ensemble…) y la cadena RAG completa: proyecto `Asistente_Legal_RAG`.
- El contenido de los contratos.

### Observaciones sobre el archivo

1. 🔀 El original importaba `Chroma` de `langchain_community` (obsoleta ✅); se cambió a `langchain_chroma`.
2. 🔀 La ruta original era de Windows y de otro usuario (`C:\Users\santiago\…`); con ella el script no falla, simplemente no devuelve nada ✅. Se cambió a la ruta de 3.6.
3. El original no cargaba el `.env`; sin él, `OpenAIEmbeddings` no encuentra la clave si no está en el entorno. Se añadió `load_dotenv()`.
4. El script depende de haber ejecutado 3.6; no lo comprueba.
5. El texto «Top 2» está fijo y «mas» sin tilde.
6. La variable `i` no se usa.
7. El original tenía saltos de línea de Windows (CRLF); no afecta al funcionamiento.
8. El código (salvo los tres cambios) se respetó: comentado y limpio tienen el mismo AST ✅.

---

## Autoevaluación

1. ¿Qué recibe y qué devuelve un retriever?
2. ¿Qué diferencia hay entre `Chroma.from_documents` (3.6) y `Chroma(...)` (3.7)?
3. ¿Qué pasa si `persist_directory` apunta a una carpeta que no existe?
4. ¿Por qué debe usarse el mismo modelo de embeddings que al crear la base?
5. ¿Qué hace `search_kwargs={"k": 2}` y cuántos documentos devuelve si se omite `k`?
6. ¿Qué tipos de `search_type` admite el retriever?
7. ¿Cuántas llamadas a la API de embeddings hace este script y para qué?
8. ¿Por qué ya no aparece el aviso de retirada de `langchain-community` en este script?
9. ¿Cuál es la forma actual de ejecutar un retriever y cuál quedó obsoleta?

### Respuestas

1. Recibe una consulta en texto y devuelve una lista de `Document`.
2. `from_documents` crea la base: calcula los vectores y los guarda. `Chroma(...)` abre una base que ya existe y no añade nada.
3. Chroma la crea vacía y el retriever devuelve una lista vacía, sin error.
4. Para que los vectores de la consulta y los de la base sean comparables; con tamaños distintos Chroma da error.
5. Pide 2 resultados. Sin `k` devuelve 4 (✅).
6. `similarity`, `similarity_score_threshold` y `mmr` (✅ según el código instalado).
7. Una: la de la consulta. Los vectores de los documentos ya están guardados.
8. Porque no importa nada de `langchain_community`: `Chroma` viene de `langchain_chroma`.
9. `retriever.invoke(consulta)` es la actual; `get_relevant_documents` quedó obsoleta.

---

## Resumen

| Concepto | Idea clave |
|---|---|
| Retriever | Texto → lista de `Document`; interfaz estándar (`Runnable`) |
| `Chroma(...)` | Abre la base creada en 3.6; no la rellena |
| `as_retriever` | Convierte la base en retriever; `similarity` y `k=2` |
| `invoke` | Ejecuta la búsqueda; 1 llamada a la API (la consulta) |
| Ruta equivocada | 0 resultados sin error ✅ |
| `langchain_chroma` + `load_dotenv()` | Importación vigente y clave desde `.env` |

---

## Fuentes consultadas

- Chroma: <https://docs.langchain.com/oss/python/integrations/vectorstores/chroma>
- Retrievers: <https://docs.langchain.com/oss/python/integrations/retrievers>
- Archivos de origen: `3.7_retrievers_langchain.py` (comentado), `3.7_retrievers_langchain_limpio.py` (limpio) y `3.7_retrievers_langchain_v1.py` (original). Comprobaciones de `search_type` y `k` en el código instalado de `langchain-core` (✅).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
