# ESTUDIO 1.3 — `build_vectorstore.py`: construir la base vectorial del Asistente Legal

> ⚠️ **Aviso de procedencia.** Este script es **nuevo**: no formaba parte del curso original. Es una reconstrucción mía a partir de `3.6_vector_stores.py` y de `config.py`. Lo probé con **embeddings simulados** (`DeterministicFakeEmbedding`) y PDF de prueba, **no con la API real de OpenAI ni con tu `chroma_db` real**. Antes de fiarte, ejecútalo una vez (ver «Cómo ejecutarlo»).

Archivos de esta lección: `1.3_build_vectorstore.py` (comentado), `1.3_build_vectorstore_limpio.py` (limpio) y `build_vectorstore.py` (el ejecutable del proyecto). Los tres tienen **el mismo código** (AST idéntico).

---

## 1. Tutorial — de cero a una base funcionando

Objetivo: tener la carpeta `chroma_db` que `rag_system.py` abre al responder preguntas.

1. Instala dependencias: `pip install -U langchain-community langchain-chroma langchain-openai langchain-text-splitters pypdf python-dotenv`.
2. Comprueba que el `.env` (cerca del proyecto) contiene `OPENAI_API_KEY`.
3. Coloca los PDF en la carpeta `contratos` (junto a la carpeta del proyecto) o decide otra ruta.
4. Ejecuta:

```bash
python build_vectorstore.py
```

Salida esperada (con 15 páginas de prueba):

```text
Se cargaron 15 documentos desde .../contratos.
Se crearon 39 chunks de texto.
La base contiene 39 vectores en <CHROMA_DB_PATH>.
```

5. Lanza la aplicación: `streamlit run app.py`.

## 2. Guía práctica (how-to)

| Necesito… | Comando |
|---|---|
| Usar otra carpeta de PDF | `python build_vectorstore.py ruta/a/los/pdf` |
| Borrar y crear de cero (PDF cambiados, otro troceado) | `python build_vectorstore.py --reconstruir` |
| Ambas cosas | `python build_vectorstore.py ruta/a/los/pdf --reconstruir` |
| Añadir PDF nuevos | Ponlos en la carpeta y ejecuta sin `--reconstruir` |
| Quitar un PDF de la base | `--reconstruir` (los ids de PDF borrados no se eliminan solos) |

## 3. Referencia

### 3.1 Importaciones y configuración

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
```

- 📘 `from config import EMBEDDING_MODEL, CHROMA_DB_PATH`: el modelo con el que **construyes** la base debe ser el mismo con el que **consultas** (`rag_system.py` también lee `EMBEDDING_MODEL`). Si difieren, las búsquedas no tienen sentido (dimensiones o espacios vectoriales distintos).
- ✅ `Chroma` viene de `langchain_chroma` (la versión de `langchain_community` está deprecada).
- ⚠️ `PyPDFDirectoryLoader` sigue viniendo de `langchain_community`: seguirá saliendo el aviso de retirada de ese paquete. Es inofensivo.

### 3.2 Parámetros

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CARPETA_PDF_POR_DEFECTO = Path(__file__).resolve().parent.parent / "contratos"
```

- `1000/200` son los valores de `3.6`. ⚠️ Si tu base real se creó con otros, cámbialos: afectan a la calidad de la recuperación.
- ⚠️ La carpeta por defecto (`../contratos`) es una **suposición mía**; pasa otra ruta como argumento si no es la tuya.

### 3.3 Ids estables

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

- Formato del id: `archivo|página|huella|n`.
- Solo el **nombre** del archivo (no la ruta): el id no cambia si mueves la carpeta (✅ comprobado ejecutando desde otra ruta).
- `n` evita ids repetidos si dos chunks de una misma página tuvieran texto idéntico.
- 🔀 **Contraste con 3.6**: allí `from_documents` se llamaba sin `ids` y cada ejecución duplicaba los vectores (lo viste: 3 resultados idénticos). Con ids, Chroma **actualiza** en lugar de duplicar.

### 3.4 Construcción

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

- La colección se llama `langchain` (la de por defecto), la misma que abre `rag_system.py`.
- El borrado con `--reconstruir` se hace **después** de cargar y trocear: si falla la lectura de los PDF, no pierdes la base antigua.
- Sin PDF en la carpeta, el script termina con un mensaje claro (✅ comprobado).

### 3.5 Punto de entrada

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye (o actualiza) la base vectorial Chroma del asistente legal.")
    parser.add_argument("carpeta_pdf", nargs="?", default=str(CARPETA_PDF_POR_DEFECTO), help="Carpeta con los PDF")
    parser.add_argument("--reconstruir", action="store_true", help="Borra la colección existente antes de crearla")
    args = parser.parse_args()
    construir(Path(args.carpeta_pdf), args.reconstruir)
```

### 3.6 Resultados comprobados (embeddings simulados)

| Prueba | Resultado |
|---|---|
| 1.ª ejecución | 39 vectores |
| Repetir ejecución | 39 vectores (no crece) |
| Ejecutar desde otra ruta | 39 vectores |
| `--reconstruir` | 39 vectores |
| Carpeta vacía | Error claro, sin traza larga |
| `rag_system.query_rag` sobre la base creada | Abre la base y devuelve fragmentos con las claves que espera `app.py` |

## 4. Explicación (por qué así)

- **Separar construir de consultar.** Calcular embeddings cuesta dinero y tiempo; no debe hacerse cada vez que arranca la app. Se construye una vez y `rag_system.py` solo abre la base.
- **Ids estables = idempotencia, no ahorro.** Re-ejecutar no duplica, pero ⚠️ **los embeddings se recalculan todos** en cada ejecución (✅ 39 textos embebidos cada vez): sigue habiendo coste de API. Si no cambiaron los PDF, no hace falta volver a ejecutarlo.
- **Límite de los ids.** Si cambia el texto de un chunk, cambia su huella y por tanto su id; el chunk antiguo se queda en la base. Para PDF modificados o un troceado distinto, usa `--reconstruir`.
- **Datos personales.** Los contratos contienen datos personales: la base `chroma_db` los contiene también (en texto y metadatos). No la subas a repositorios públicos.

## 5. Sugerencias (no aplicadas)

1. Pasar `CHUNK_SIZE`, `CHUNK_OVERLAP` y la carpeta de PDF a `config.py`.
2. Evitar recalcular embeddings de ids ya existentes (⚠️ no probado).
3. Eliminar de la base los chunks cuyo id ya no corresponde a ningún PDF.
4. Mostrar un resumen por PDF.

## 6. Autoevaluación

1. ¿Por qué un id que incluye la ruta completa sería peor que uno con solo el nombre del archivo?
2. Ejecutas el script dos veces sin cambiar los PDF: ¿cuántos vectores hay? ¿Se llama dos veces a la API de embeddings?
3. Cambias una frase de un PDF: ¿qué pasa sin `--reconstruir`?
4. ¿Qué ocurre si construyes con `text-embedding-3-large` y consultas con otro modelo?

**Respuestas:** 1) al mover la carpeta cambiarían todos los ids y se duplicaría todo. 2) El mismo número; sí, se embeben todos los chunks de nuevo. 3) El chunk modificado se añade con id nuevo y el antiguo permanece. 4) Los vectores no son comparables; los resultados serían incorrectos.
