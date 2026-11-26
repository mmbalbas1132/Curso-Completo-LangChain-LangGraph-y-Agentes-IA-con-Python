# 3.6 Vector stores: guardar embeddings en Chroma y buscar por significado

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.6_vector_stores.py` (versión con comentarios didácticos; su código es idéntico al de `3.6_vector_stores_limpio.py` y al original, ✅ comprobado) y de la documentación oficial de LangChain. El original ya incluía `load_dotenv()` (añadido por el alumno). **Cambio sobre el original:** `Chroma` se importa ahora de `langchain_chroma` (el original usaba `langchain_community`, donde la clase está obsoleta); es la única línea de código distinta. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = lo observado o la documentación difiere del archivo; ✅ *(ejecutado)* = comportamiento observado con `langchain-community` 0.4.2, `langchain-core` 1.7.7, `langchain-text-splitters` 1.1.3, `chromadb` 1.5.9 y `pypdf` 7.1.0.
>
> **Alcance y privacidad.** La carpeta `contratos` contiene documentos personales: **solo se comprobó su estructura** (número de PDF, páginas, claves de metadatos, tamaños); su contenido no se reproduce aquí. Yo ejecuté el script con **embeddings simulados** y una carpeta de base de datos **temporal** (tu `chroma_db` no se tocó). La **salida real** con la API de OpenAI la ejecutó el alumno (código de salida 0) y se resume en la lección 6 sin reproducir el contenido de los contratos.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un *vector store* (base de datos vectorial) y qué resuelve.
2. Cargar todos los PDF de una carpeta con `PyPDFDirectoryLoader`.
3. Trocear los documentos y guardarlos con sus vectores en Chroma (`Chroma.from_documents`), persistidos en disco.
4. Buscar los fragmentos más parecidos a una pregunta con `similarity_search`.
5. Tres advertencias reales: duplicados al re-ejecutar, clase obsoleta y rutas absolutas.

### Qué necesitas

- Haber visto 3.1 (loaders), 3.3-3.4 (splitters) y 3.5 (embeddings).
- `pip install -U langchain-community langchain-chroma langchain-openai langchain-text-splitters pypdf chromadb python-dotenv`.
- `OPENAI_API_KEY` en un archivo `.env` (el script usa `load_dotenv()`).
- Una carpeta con PDFs (aquí, `contratos`).

### Mapa del tema

```text
carpeta de PDF ─ PyPDFDirectoryLoader.load() ─► [Document × 15]   (una por página; 5 PDF × 3 páginas)
      └─ RecursiveCharacterTextSplitter(1000, 200).split_documents ─► [Document × 39]  (chunks)
            └─ Chroma.from_documents(chunks, OpenAIEmbeddings, persist_directory)
                    · vector por chunk (llamadas a OpenAI) ─► base en disco
                         └─ similarity_search(consulta, k=3) ─► 3 Document más parecidos
```

### Glosario

| Término | Significado |
|---|---|
| **Vector store** | Base de datos que guarda textos junto a sus vectores (embeddings) y busca por similitud. |
| **Chroma** | Vector store de código abierto que puede guardar en disco o en memoria. |
| **Colección** | Conjunto de vectores dentro de Chroma. Por defecto, `from_documents` usa el nombre `langchain` ✅. |
| **`persist_directory`** | Carpeta donde Chroma guarda los datos para conservarlos entre ejecuciones. |
| **Chunk** | Fragmento de texto resultante del splitter (ver 3.4). |
| **Búsqueda por similitud** | Convertir la consulta en vector y devolver los chunks cuyos vectores están más cerca. |
| **`k`** | Cuántos resultados devolver. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-community langchain-chroma langchain-openai langchain-text-splitters pypdf chromadb python-dotenv
```

Pon `OPENAI_API_KEY=tu_clave` en `.env` (ver 3.5). Coloca los PDF en la carpeta que indica el script.

### Lección 1: importaciones y carga del `.env`

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv

# Carga OPENAI_API_KEY desde el archivo .env (no sobrescribe variables ya definidas en el entorno).
load_dotenv()
```

- `import os` **no se usa** en el script (✅ no aparece en otra línea).
- 🔀 **`Chroma` ahora viene de `langchain_chroma`.** El original la importaba de `langchain_community`, donde ✅ la clase está obsoleta (al activar todos los avisos de Python aparece `LangChainDeprecationWarning: The class Chroma was deprecated in LangChain 0.2.9 …`; con la configuración por defecto no se muestra). La página oficial de Chroma usa `from langchain_chroma import Chroma`. ✅ Con embeddings simulados, la nueva clase se comporta igual (`from_documents`, `similarity_search`, colección `langchain`) y **abre bases creadas con la clase antigua**. ✅ El alumno ejecutó el script completo con `langchain_chroma` y la API real (código de salida 0) y obtuvo los mismos 3 resultados que con la clase antigua (lección 6). El aviso de retirada de `langchain-community` seguirá saliendo, porque `PyPDFDirectoryLoader` se importa de ese paquete.

### Lección 2: cargar los PDF de una carpeta

```python
loader = PyPDFDirectoryLoader("/home/mmbal/proyectos/curso-langchain/Tema3/contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio contratos.")
```

`PyPDFDirectoryLoader(ruta)` lee todos los PDF de la carpeta y `.load()` devuelve una lista de `Document`, uno por página. ✅ Con tu carpeta: **5 PDF de 3 páginas = 15 `Document`**, ninguno vacío (de 94 a 2.569 caracteres). Claves de `metadata`: `author`, `creationdate`, `creator`, `moddate`, `page`, `page_label`, `producer`, `source`, `total_pages`.

⚠️ La página oficial solo muestra el uso con la ruta y no documenta otros parámetros del cargador. La ruta del script es una ruta absoluta de Linux de tu equipo.

### Lección 3: trocear en chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")
```

Como en 3.4, `split_documents` trocea cada página por separado y conserva sus metadatos. ✅ Resultado: **15 páginas → 39 chunks** (de 94 a 998 caracteres, ninguno de más de 1000). Aquí sí se parte el texto, porque las páginas (hasta 2.569) superan `chunk_size=1000`.

### Lección 4: crear la base vectorial

```python
vectorstore = Chroma.from_documents(
    docs_split,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
)
```

| Código | Qué hace |
|---|---|
| `Chroma.from_documents(docs_split, …)` | Convierte cada chunk en un vector y los guarda con su texto y metadatos. |
| `embedding=OpenAIEmbeddings(model="text-embedding-3-large")` | Modelo que calcula los vectores (ver 3.5). Hace llamadas de pago a la API. |
| `persist_directory=…` | Carpeta donde se guarda la base. ✅ Los datos siguen en disco al terminar el programa. Sin este parámetro la base sería solo en memoria (la página oficial de Chroma lo indica). |

✅ Con embeddings simulados la colección se llama `langchain` y contiene **39 vectores**.

🔀 **Duplicados.** ✅ Si ejecutas el script **otra vez** con la misma carpeta, los 39 chunks se añaden de nuevo: **39 → 78 vectores**, porque no se pasan ids ni se borra la colección. ✅ **Caso real:** tras varias ejecuciones sin borrar la carpeta, los 3 resultados de `similarity_search(k=3)` eran **el mismo chunk repetido** (3 copias). Al borrar la carpeta de la base (`persist_directory`) y ejecutar una sola vez, volvieron a salir 3 chunks distintos. ✅ Reproducido también con embeddings simulados: con ids estables (parámetro `ids` de `from_documents`) la base no crece al re-ejecutar (10 vectores tras 3 ejecuciones en la prueba, en vez de 30).

⚠️ Con una clave no válida ✅ la API responde `401 AuthenticationError` en esta llamada; los dos `print` anteriores ya se habrán mostrado.

### Lección 5: la consulta

```python
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos"
```

Es una pregunta en lenguaje natural sobre los contratos. La cadena abre con `¿` pero no cierra con `?` (se respeta el código).

### Lección 6: buscar y mostrar

```python
resultados = vectorstore.similarity_search(consulta, k=3)

print("Tres documentos más coincidentes con la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
```

| Código | Qué hace |
|---|---|
| `similarity_search(consulta, k=3)` | Convierte la consulta en vector (otra llamada a la API) y devuelve los 3 `Document` más parecidos (✅ con embeddings simulados devuelve 3). |
| `enumerate(resultados, start=1)` | Recorre la lista numerando desde 1 (`i` no se imprime). |
| `doc.page_content` / `doc.metadata` | Texto del chunk y sus metadatos (`source`, `page`…). |

✅ Con embeddings simulados el script imprime 3 bloques "Contenido / Metadatos", pero el orden no tiene significado.

✅ **Salida real (ejecución del alumno con la API de OpenAI).** Mostró 15 documentos, 39 chunks y estos 3 resultados (descritos sin reproducir datos personales):

| N.º | Procedencia | Qué contiene | ¿Responde a la pregunta? |
|---|---|---|---|
| 1 | Primera página de `CONTRATO DE ARRENDAMIENTO DE LOCAL DE NEGOCIO 2.pdf` | Identificación de las partes y descripción del local (incluida su ubicación) | **Sí** |
| 2 | Tercera página del mismo PDF | Cierre con firmas y anexos; aparece el nombre de la persona, no la ubicación del local | Solo indirectamente |
| 3 | Primera página de `CONTRATO DE ARRENDAMIENTO DE LOCAL DE NEGOCIO.pdf` | Otro contrato de local de negocio, de otras partes y otra ciudad | **No** |

Lecciones: (a) `similarity_search` devuelve siempre los `k` más parecidos, sean o no relevantes; (b) se parece por significado general (contrato de local de negocio), no filtra por la persona; (c) un chunk puede cortarse a mitad de una frase (el 1.º termina en mitad de un punto del contrato), efecto del `chunk_size=1000`; (d) el orden de las claves de `metadata` varía de un resultado a otro (es un diccionario, no importa); (e) ✅ con `langchain_chroma` y la base recreada, la salida volvió a ser la misma que con la clase antigua.

---

## Guía práctica

### ¿Cómo ejecuto el script?

1. Instala los paquetes y pon la clave en `.env`.
2. Comprueba que la carpeta `contratos` existe en la ruta del script (ruta absoluta de Linux).
3. Ejecuta. Coste: embeddings de los 39 chunks + 1 de la consulta.

### ¿Cómo evito duplicados al re-ejecutar?

Opciones (⚠️ no ejecutadas con la API real):

- Borrar la carpeta de la base antes de ejecutar (✅ es lo que resolvió el caso real).
- Pasar ids estables a `from_documents` (✅ la firma admite `ids`).
- Abrir la base existente en lugar de recrearla.

### ¿Cómo vuelvo a la forma antigua (o migro otro código)?

Ya está migrado en este tema. Para migrar otro código que use la clase antigua:

```bash
pip install -U langchain-chroma
```

Cambia la importación `from langchain_community.vectorstores import Chroma` por `from langchain_chroma import Chroma`. ✅ Con embeddings simulados, `from_documents`, `similarity_search` y los retrievers MMR y de similitud siguen funcionando y se abre una base creada con la clase antigua. ⚠️ La página oficial crea el almacén con `Chroma(collection_name=…, embedding_function=…, persist_directory=…)` y añade documentos con `add_documents`; `from_documents` también existe en la clase nueva ✅.

### ¿Cómo hago la ruta portable?

Usa una ruta relativa o `pathlib` a partir de la ubicación del script (ver 3.3). ⚠️ Forma habitual de Python.

### Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `LangChainDeprecationWarning: The class Chroma was deprecated…` ✅ (solo con todos los avisos activados) | Se usa `Chroma` de `langchain_community` (forma del original) | Usar `langchain_chroma`. |
| `ModuleNotFoundError: langchain_chroma` | Falta el paquete | `pip install -U langchain-chroma`. |
| `AuthenticationError 401` ✅ (clave falsa) | Clave no válida o caducada, o una `OPENAI_API_KEY` vieja ya definida (no la sobrescribe `load_dotenv()`) | Clave nueva y revisar el entorno. |
| `OpenAIError: Missing credentials…` ✅ | Falta `OPENAI_API_KEY` | Ponla en `.env`. |
| La base tiene el doble de vectores | Re-ejecución sin ids ✅ | Ver *¿Cómo evito duplicados?* |
| `ValueError … is not a valid file or url` (ruta inexistente) | La ruta absoluta no existe en tu equipo | Corrige la ruta. |
| `ModuleNotFoundError: chromadb` ✅ | Falta `chromadb` | `pip install chromadb`. |

---

## Referencia

### Importaciones

| Importación | Paquete | Para qué |
|---|---|---|
| `Chroma` | `langchain_chroma` (usada en este documento); la de `langchain_community.vectorstores` está obsoleta ✅ | Base vectorial |
| `OpenAIEmbeddings` | `langchain_openai` | Embeddings |
| `PyPDFDirectoryLoader` | `langchain_community.document_loaders` | PDFs de una carpeta |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Chunks |
| `load_dotenv` | `dotenv` | Cargar el `.env` |

### Métodos y parámetros usados

| Elemento | Detalle |
|---|---|
| `PyPDFDirectoryLoader(ruta).load()` | `list[Document]`, uno por página ✅ |
| `Chroma.from_documents(documents, embedding, ids, collection_name, persist_directory, …)` | Firma ✅ de la clase de `langchain_community`; `collection_name` por defecto `'langchain'` |
| `vectorstore.similarity_search(consulta, k)` | Lista de `Document`; la página oficial muestra `k=2` y un filtro opcional por metadatos |

### Medidas ✅

| Dato | Valor |
|---|---|
| PDF en `contratos` | 5 (3 páginas cada uno) |
| `Document` tras cargar | 15 |
| Chunks (1000/200) | 39 (94–998 caracteres) |
| Vectores tras 1ª ejecución / 2ª | 39 / 78 |
| Resultados de `similarity_search(k=3)` | 3 (en la ejecución real, 1 responde a la pregunta) |

---

## Explicación

### ¿Por qué un vector store?

Porque comparar la consulta con todos los textos uno a uno sería lento y repetitivo: la base guarda los vectores ya calculados y busca por cercanía. 📘⚠️ El archivo no lo explica; es la idea general vista en 3.5.

### ¿Por qué trocear antes de guardar?

Cada chunk pasa a ser una unidad de búsqueda: la consulta recupera fragmentos concretos y no páginas enteras. La página oficial de *chunking* lo explica (ver 3.4).

### ¿Por qué `persist_directory`?

Para no recalcular los vectores (y pagar los embeddings) en cada ejecución. Pero este script **recrea** la colección cada vez, así que persistir sin controlar los duplicados los acumula.

### ¿Qué enfoque usar en cada caso?

| Situación | Enfoque |
|---|---|
| Prueba rápida | Base en memoria (sin `persist_directory`) |
| Reutilizar la base | Persistir y abrirla sin volver a añadir los documentos |
| Código nuevo | `langchain_chroma` en lugar de `langchain_community` |

### Lo que este documento no cubre

- La calidad de las búsquedas más allá de este único ejemplo (una consulta, `k=3`).
- Filtros por metadatos, otras métricas, otros vector stores.
- Retrievers y RAG completo (proyecto del asistente legal, `Asistente_Legal_RAG`).
- El contenido de los contratos.

### Observaciones sobre el archivo

1. 🔀 El original importaba `Chroma` de `langchain_community` (obsoleta ✅); se cambió a `langchain_chroma`.
2. 🔀 Re-ejecutar duplica los vectores ✅.
3. `import os` sin usar.
4. Rutas absolutas de un equipo concreto.
5. La consulta no cierra la interrogación.
6. El script no gestiona errores (clave, rutas).
7. El código se respetó: las tres versiones tienen el mismo AST ✅.

---

## Autoevaluación

1. ¿Qué devuelve `PyPDFDirectoryLoader.load()`?
2. ¿Cuántos `Document` y cuántos chunks salen con la carpeta `contratos`?
3. ¿Qué hace `Chroma.from_documents`?
4. ¿Para qué sirve `persist_directory`?
5. ¿Qué pasa al ejecutar el script dos veces con la misma carpeta?
6. ¿Qué devuelve `similarity_search(consulta, k=3)`?
7. ¿Por qué se recomienda `langchain_chroma`?
8. ¿Por qué la búsqueda con embeddings simulados no demuestra nada sobre la calidad?
9. ¿Qué llamadas de pago hace este script?
10. En la salida real, ¿cuántos de los 3 resultados responden a la pregunta y qué enseña eso sobre `k`?

### Respuestas

1. Una lista de `Document`, uno por página de cada PDF.
2. 15 `Document` y 39 chunks.
3. Calcula el vector de cada chunk con el modelo de embeddings y lo guarda con su texto y metadatos en una colección de Chroma.
4. Guardar la base en esa carpeta para conservarla entre ejecuciones.
5. Los 39 chunks se añaden de nuevo (78 vectores).
6. Una lista con los 3 `Document` más parecidos a la consulta.
7. Porque la clase de `langchain_community` está obsoleta (✅ aviso) y la documentación usa `langchain_chroma`.
8. Porque los vectores simulados no reflejan el significado; el orden es arbitrario.
9. Embeddings de los 39 chunks y de la consulta.
10. Solo el primero responde directamente; el segundo es del mismo contrato pero no incluye la ubicación; el tercero es de otro contrato. `similarity_search` devuelve siempre `k` resultados, relevantes o no.

---

## Resumen

| Concepto | Idea clave |
|---|---|
| `PyPDFDirectoryLoader` | Una `Document` por página de todos los PDF de la carpeta |
| Splitter 1000/200 | 15 páginas → 39 chunks |
| `Chroma.from_documents` | Vectores + textos + metadatos en una colección |
| `persist_directory` | Guarda en disco; re-ejecutar duplica ✅ |
| `similarity_search(k)` | Los k chunks más parecidos (en la salida real, 1 de 3 respondía directamente) |
| `langchain_chroma` | Paquete recomendado; la clase de `community` está obsoleta |

---

## Fuentes consultadas

- Chroma: <https://docs.langchain.com/oss/python/integrations/vectorstores/chroma>
- `PyPDFDirectoryLoader`: <https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfdirectory>
- Embeddings: <https://docs.langchain.com/oss/python/integrations/embeddings>
- Chunking: <https://docs.langchain.com/oss/python/langchain/chunking>
- Archivos de origen: `3.6_vector_stores.py` (comentado) y `3.6_vector_stores_limpio.py` (limpio); carpeta `contratos` (solo estructura).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
