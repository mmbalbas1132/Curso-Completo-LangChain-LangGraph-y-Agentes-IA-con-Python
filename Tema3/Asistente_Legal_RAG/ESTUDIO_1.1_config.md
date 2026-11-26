# 1.1 Configuración central del asistente RAG (`config.py`)

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `config.py` (original; la única línea de código cambiada es `CHROMA_DB_PATH`, lección 2) y sus versiones `1.1_config.py` (comentada) y `1.1_config_limpio.py` (limpia), con código idéntico (✅ comprobado con el árbol sintáctico). Marcas: 📘 *(material del curso)*; ⚠️ *(no contrastado)*; 🔀 *(discrepancia)*; ✅ *(ejecutado o comprobado)*. `config.py` solo define constantes y no usa LangChain; para saber **dónde se usa cada constante** se leyó `rag_system.py` (documento 1.4), y varias comprobaciones ✅ se hicieron ejecutándolo con modelos simulados (`langchain-core` 1.8.4, `langchain-classic` 1.1.0, `chromadb` 1.5.9). `config.py` y las dos versiones 1.1 tienen la misma ruta de WSL en `CHROMA_DB_PATH`.

---

## Antes de empezar

### Qué vas a aprender

1. Para qué sirve un archivo de configuración y por qué se separan los valores del código.
2. Qué significa cada constante (modelos, base vectorial, búsqueda MMR, búsqueda híbrida).
3. Dónde se usa cada una en el proyecto.

### Qué necesitas

- Tener una idea de qué es un RAG. Este es el primer documento del proyecto: no necesitas haber visto los demás (1.2 prompts, 1.3 construcción de la base, 1.4 motor RAG y 1.5 interfaz vienen después).

### Mapa del tema

```text
config.py ──from config import *──► rag_system.py
 modelos ───────────────► OpenAIEmbeddings / ChatOpenAI
 ruta de la base ───────► Chroma(persist_directory=...)
 MMR (k, fetch_k, λ) ───► as_retriever(search_type="mmr", search_kwargs=...)
 híbrido + umbral ──────► EnsembleRetriever(...)  y  get_retriever_info()
```

### Glosario

| Término | Significado |
|---|---|
| **Embedding** | Representación numérica de un texto; se usa para buscar fragmentos parecidos. |
| **Vector store (Chroma)** | Base de datos que guarda los embeddings de los fragmentos. |
| **MMR** | *Maximal Marginal Relevance*: búsqueda que equilibra parecido con la consulta y **diversidad** entre resultados. ✅ (descrita así en la documentación del propio método instalado). |
| **`k`** | Número de fragmentos finales. |
| **`fetch_k`** | Candidatos que se recuperan antes de elegir los `k` finales. |
| **`lambda_mult`** | Número entre 0 y 1: 0 = máxima diversidad, 1 = mínima. ✅ (docstring de `Chroma.max_marginal_relevance_search` instalado). |
| **Retriever híbrido** | Aquí, combinación de dos retrievers (MMR + similitud). |

---

## Tutorial

### Lección 1: modelos (líneas 1 a 4)

```python
# Configuración de modelos
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"
```

| Constante | Valor | Dónde se usa ✅ |
|---|---|---|
| `EMBEDDING_MODEL` | `"text-embedding-3-large"` | `OpenAIEmbeddings(model=...)` |
| `QUERY_MODEL` | `"gpt-4o-mini"` | Modelo del `MultiQueryRetriever` (reescribe la consulta) |
| `GENERATION_MODEL` | `"gpt-4o"` | Modelo que redacta la respuesta final |

Los nombres de los modelos de OpenAI no se contrastaron con documentación oficial. ⚠️ Coinciden con el texto de la barra lateral de `app.py` ("Consultas: GPT-4o-mini / Respuestas: GPT-4o"), que está escrito a mano en `app.py`.

### Lección 2: base vectorial (líneas 6 y 7)

```python
# Configuración del vector store
CHROMA_DB_PATH = "/home/mmbal/proyectos/curso-langchain/Tema3/chroma_db"
```

Carpeta donde `Chroma` guarda y lee los datos (`persist_directory`). 🔀 El original traía una ruta **absoluta de Windows de otro usuario/equipo** (`C:\Users\…\Tema 3\chroma_db`); se cambió a la ruta de tu WSL, la misma carpeta donde 3.6 y `1.3_build_vectorstore.py` crean la base. Si mueves el proyecto, habrá que cambiarla. La documentación oficial de Chroma indica que `persist_directory` sirve para guardar los datos entre ejecuciones.

### Lección 3: retriever MMR (líneas 9 a 13)

```python
# Configuración del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2
```

| Constante | Valor | Se pasa a ✅ | Significado |
|---|---|---|---|
| `SEARCH_TYPE` | `"mmr"` | `as_retriever(search_type=...)` | Tipo de búsqueda |
| `MMR_DIVERSITY_LAMBDA` | `0.7` | `search_kwargs["lambda_mult"]` | Cercano a 1 = poca diversidad |
| `MMR_FETCH_K` | `20` | `search_kwargs["fetch_k"]` | Candidatos previos |
| `SEARCH_K` | `2` | `search_kwargs["k"]` (ambos retrievers) y recorte de `docs[:SEARCH_K]` | Fragmentos finales |

La documentación oficial de Chroma muestra `as_retriever(search_type="mmr", search_kwargs={"k": ..., "fetch_k": ...})`; **no menciona `lambda_mult`**. ⚠️ (sí está en la documentación del método instalado ✅).

### Lección 4: búsqueda híbrida (líneas 15 a 17)

```python
# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70
```

- `ENABLE_HYBRID_SEARCH = True` activa el `EnsembleRetriever` en `rag_system.py` y añade `+ Hybrid` al texto de la barra lateral ✅.
- 🔀 `SIMILARITY_THRESHOLD` se pasa como `similarity_threshold=` al `EnsembleRetriever`, pero ese parámetro **no existe**: la referencia oficial no lo lista y ✅ la clase instalada ignora los argumentos extra (`extra="ignore"`). Efecto real sobre la búsqueda: ninguno; solo se muestra como `umbral` en `get_retriever_info()`.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo cambio el modelo de generación?

Edita `GENERATION_MODEL`. Recuerda que el texto de la barra lateral de `app.py` está escrito a mano y no se actualiza solo.

### ¿Cómo hago respuestas más variadas entre fragmentos?

Baja `MMR_DIVERSITY_LAMBDA` (0 = máxima diversidad, 1 = mínima). ✅

### ¿Cómo desactivo la búsqueda híbrida?

`ENABLE_HYBRID_SEARCH = False`: `rag_system.py` usa solo el retriever MMR + MultiQuery. ✅ (por el código).

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| La base vectorial no se encuentra o sale vacía | `CHROMA_DB_PATH` apunta a una carpeta que no es la de la base (si no existe, Chroma crea una vacía ✅) | Poner la ruta de la base creada por 3.6 o 1.3 |
| Cambiar `SIMILARITY_THRESHOLD` no cambia nada | El parámetro se ignora en `EnsembleRetriever` ✅ | No esperar efecto |
| Devuelve más/menos de 2 fragmentos | `SEARCH_K` | Ajustar `SEARCH_K` |

---

## Referencia

| Constante | Tipo | Valor | Uso |
|---|---|---|---|
| `EMBEDDING_MODEL` | `str` | `text-embedding-3-large` | Embeddings |
| `QUERY_MODEL` | `str` | `gpt-4o-mini` | Variantes de consulta |
| `GENERATION_MODEL` | `str` | `gpt-4o` | Respuesta final |
| `CHROMA_DB_PATH` | `str` | ruta de tu WSL | Base Chroma |
| `SEARCH_TYPE` | `str` | `mmr` | Tipo de búsqueda |
| `MMR_DIVERSITY_LAMBDA` | `float` | `0.7` | `lambda_mult` |
| `MMR_FETCH_K` | `int` | `20` | `fetch_k` |
| `SEARCH_K` | `int` | `2` | `k` |
| `ENABLE_HYBRID_SEARCH` | `bool` | `True` | Activa ensemble |
| `SIMILARITY_THRESHOLD` | `float` | `0.70` | Se ignora en la búsqueda ✅ |

---

## Explicación

### ¿Por qué un archivo de configuración?

Para cambiar modelos y parámetros sin tocar la lógica. 📘

### ¿Por qué `from config import *` es delicado?

Importa todos los nombres sin decir cuáles; es cómodo pero oculta de dónde viene cada constante. (Así lo hace `rag_system.py`.)

### Lo que este documento no cubre

- Cómo se construyó la base Chroma (no está en estos archivos).
- Los nombres de modelo de OpenAI y sus precios.

### Observaciones sobre el archivo

- 🔀 Ruta original de otro usuario/equipo (Windows): sustituida por la ruta de tu WSL. Sigue siendo absoluta.
- `SIMILARITY_THRESHOLD` no tiene efecto. 🔀 ✅
- Los pesos `[0.7, 0.3]` del ensemble no están aquí: están escritos a mano en `rag_system.py`.
- "Configuracion" sin tilde en un comentario del original.

---

## Autoevaluación

1. ¿Qué modelo hace los embeddings y cuál redacta la respuesta?
2. ¿Qué hace `QUERY_MODEL`?
3. ¿Qué significa `lambda_mult` cercano a 0?
4. ¿Qué diferencia hay entre `SEARCH_K` y `MMR_FETCH_K`?
5. ¿Por qué cambiar `SIMILARITY_THRESHOLD` no cambia las respuestas?
6. ¿Qué problema tiene `CHROMA_DB_PATH`?
7. ¿Qué activa `ENABLE_HYBRID_SEARCH`?

### Respuestas

1. `text-embedding-3-large` hace los embeddings; `gpt-4o` redacta la respuesta.
2. Genera variantes de la consulta para el `MultiQueryRetriever`.
3. Máxima diversidad entre los resultados ✅.
4. `SEARCH_K` son los fragmentos finales; `MMR_FETCH_K`, los candidatos previos.
5. Porque `EnsembleRetriever` no tiene ese parámetro y lo ignora ✅.
6. Es absoluta, así que depende del equipo: el original traía la de otro usuario y hubo que cambiarla por la de tu WSL.
7. El `EnsembleRetriever` que combina MMR + MultiQuery con similitud.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Modelos | Embeddings, consultas y generación |
| MMR | `k`, `fetch_k`, `lambda_mult` |
| Híbrido | Ensemble MMR + similitud |
| `SIMILARITY_THRESHOLD` | Sin efecto ✅ |

---

## Fuentes consultadas

- Chroma (LangChain): <https://docs.langchain.com/oss/python/integrations/vectorstores/chroma>
- Referencia de `EnsembleRetriever`: <https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever>
- Material del tema: `config.py`, `1.1_config.py`, `1.1_config_limpio.py` (en `Tema_3/Asistente_Legal_RAG/`) y `rag_system.py` (solo para ver dónde se usa cada constante).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
