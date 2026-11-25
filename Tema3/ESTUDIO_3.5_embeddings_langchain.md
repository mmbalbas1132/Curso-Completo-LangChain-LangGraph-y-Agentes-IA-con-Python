# 3.5 Embeddings con LangChain: del texto al vector y la similitud coseno

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.5_embeddings_langchain.py` (versión con comentarios didácticos; su código es idéntico al de `3.5_embeddings_langchain_limpio.py`, ✅ comprobado) y de la documentación oficial de LangChain. **Cambio sobre el original:** los dos archivos incluyen `from dotenv import load_dotenv` y `load_dotenv()` para leer la clave del `.env`; el resto del código es idéntico al original (que no tenía comentarios). Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = lo observado o la documentación difiere del archivo; ✅ *(ejecutado)* = comportamiento observado con `langchain-openai` 1.7.4, `langchain-core` 1.7.7 y `numpy` 2.2.6.
>
> **Alcance.** No se ejecutó nada contra la API real de OpenAI (no hay clave): la mecánica se comprobó con un modelo de embeddings simulado, de modo que **no se conoce el valor real de la similitud ni la dimensión real**; no se inventan.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un *embedding* y para qué sirve.
2. Crear un modelo de embeddings con `OpenAIEmbeddings`.
3. Convertir un texto en un vector con `embed_query`.
4. Calcular la similitud coseno entre dos vectores con NumPy e interpretarla.

### Qué necesitas

- Python básico (listas, f-strings).
- `pip install -U langchain-openai numpy python-dotenv`.
- Una clave de OpenAI en `OPENAI_API_KEY`, guardada en un archivo `.env`. El script la lee con `load_dotenv()`.
- Haber visto el tema de documentos (3.1–3.4) ayuda, pero no es imprescindible.

### Mapa del tema

```text
"La capital de Francia es París."          "París es un nombre común para mascotas."
            │ embed_query                               │ embed_query
            ▼                                           ▼
        vec1 = [0.01, -0.02, …]                     vec2 = [ … ]        (listas de floats)
            └───────────── len(vec1) ──► dimensión
            └──── cos_sim = (vec1·vec2) / (‖vec1‖·‖vec2‖) ────┘  ──► número entre -1 y 1
```

### Glosario

| Término | Significado |
|---|---|
| **Embedding** | Vector de números de longitud fija que representa el significado de un texto: textos con ideas parecidas quedan cerca en el espacio de vectores (definición de la documentación de LangChain). |
| **Vector** | Lista ordenada de números. |
| **Dimensión** | Cantidad de números del vector (`len(vector)`). |
| **`OpenAIEmbeddings`** | Clase de `langchain_openai` que usa los modelos de embeddings de OpenAI. |
| **`embed_query`** | Método que convierte **un** texto en su vector. |
| **`embed_documents`** | Método que convierte una **lista** de textos en una lista de vectores. |
| **Producto escalar** (`np.dot`) | Suma de multiplicar dos vectores número a número. |
| **Norma** (`np.linalg.norm`) | Longitud de un vector. |
| **Similitud coseno** | Medida del ángulo entre dos vectores: 1 = misma dirección, 0 = perpendiculares, -1 = opuestos. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-openai numpy python-dotenv
```

**La clave de API.** Guarda `OPENAI_API_KEY=tu_clave` en un archivo `.env` (sin comillas ni espacios alrededor del `=`). `load_dotenv()` lo busca desde la carpeta del script hacia arriba y no sobrescribe una variable ya definida en el entorno (✅ comprobado). Sin clave, ✅ `OpenAIEmbeddings(...)` falla con `OpenAIError: Missing credentials…`.

### Lección 1: importaciones y carga del `.env`

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()
```

`numpy` **no** es una dependencia de LangChain: se instala aparte. `load_dotenv()` es la línea añadida al original (ver *Procedencia*); debe ir antes de crear el modelo.

### Lección 2: crear el modelo

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

La página oficial de `OpenAIEmbeddings` usa este mismo nombre en su ejemplo. Para la familia `text-embedding-3` permite elegir el tamaño del vector con el parámetro opcional `dimensions` (aquí no se usa).

🔀 **Si omitieras `model`**, la clase usa por defecto otro modelo: ✅ en la versión instalada es `text-embedding-ada-002`. Por eso el script lo indica.

### Lección 3: los dos textos

```python
texto1 = "La capital de Francia es París."
texto2 = "París es un nombre común para mascotas."
```

Comparten la palabra "París" con sentidos distintos (ciudad / nombre de mascota). 📘⚠️ Parece pensado para ver si el vector refleja el significado y no solo las palabras; el script no lo explica.

### Lección 4: obtener los vectores

```python
vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)
```

`embed_query` devuelve una lista de números (`list[float]`; ✅ comprobado con un modelo simulado). Cada llamada es una petición a la API: **2 en total**.

### Lección 5: la dimensión

```python
print(f"Dimensión de los vectores: {len(vec1)}")
```

Imprime cuántos números tiene el vector. Para `text-embedding-3-large`, OpenAI indica 3072 por defecto ⚠️ (dato que no aparece en las páginas de LangChain consultadas y que no se comprobó contra la API). Ambos vectores tienen la misma dimensión, requisito para compararlos.

### Lección 6: similitud coseno

```python
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

| Parte | Qué hace |
|---|---|
| `np.dot(vec1, vec2)` | Producto escalar de los dos vectores. |
| `np.linalg.norm(vec1)` | Longitud de `vec1`. |
| `/ (norma1 * norma2)` | Divide para quitar el efecto de la longitud: solo cuenta la **dirección**. |

La página oficial de embeddings describe la similitud coseno como la medida del ángulo entre dos vectores y la lista junto con la distancia euclídea y el producto escalar.

✅ Comprobación con vectores sencillos:

| Caso | Resultado |
|---|---|
| Idénticos | 1.0 |
| Perpendiculares | 0.0 |
| Opuestos | -1.0 |
| Uno multiplicado por 5 | 1.0 (importa la dirección, no la longitud) |

### Lección 7: mostrar el resultado

```python
print(f"Similitud coseno entre vec1 y vec2: {cos_sim:.3f}")
```

`:.3f` muestra el número con 3 decimales. ✅ Con embeddings simulados el script imprime este formato (`Dimensión de los vectores: 3072` / `Similitud coseno entre vec1 y vec2: 0.009`), pero **esos números no significan nada**: son del simulador, no de OpenAI. El valor real solo lo obtendrás ejecutando con tu clave.

---

## Guía práctica

### ¿Cómo ejecuto el script?

1. Instala los paquetes y pon la clave en `.env` (ver *Preparación*).
2. Ejecuta el script desde cualquier carpeta: `load_dotenv()` busca el `.env` desde la carpeta del script hacia arriba.
3. Coste: 2 llamadas de embeddings por ejecución.

### ¿Cómo obtengo los vectores de varios textos a la vez?

```python
vectores = embeddings.embed_documents([texto1, texto2])
```

⚠️ Forma documentada en la página oficial (devuelve un vector por texto); no es el código del archivo y no se ejecutó con la API real.

### ¿Cómo reduzco el tamaño del vector?

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
```

⚠️ La página oficial lo muestra comentado (`# dimensions=1024`) y dice que se puede elegir el tamaño con la familia `text-embedding-3`. No se ejecutó.

### Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `OpenAIError: Missing credentials…` | No hay `OPENAI_API_KEY` en el entorno ✅ | Ponla en `.env` y revisa que `load_dotenv()` va antes de crear el modelo. |
| `ModuleNotFoundError: numpy` | Falta NumPy (no viene con LangChain) | `pip install numpy`. |
| `ModuleNotFoundError: langchain_openai` | Falta el paquete | `pip install langchain-openai`. |
| 401 `API key has expired` | Clave caducada, o una `OPENAI_API_KEY` vieja ya definida en el entorno que `load_dotenv()` no sobrescribe | Crea una clave nueva y revisa el entorno. |
| Los vectores tienen otra dimensión que la esperada | Se usó otro modelo o `dimensions` | Indica `model` y comprueba `len(vec1)`. |

---

## Referencia

### Importaciones

| Importación | Paquete | Para qué |
|---|---|---|
| `OpenAIEmbeddings` | `langchain_openai` | Modelo de embeddings de OpenAI |
| `numpy as np` | `numpy` | Producto escalar y norma |
| `load_dotenv` | `dotenv` (`python-dotenv`) | Cargar el `.env` |

### Métodos y parámetros

| Elemento | Detalle |
|---|---|
| `OpenAIEmbeddings(model=…)` | Modelo a usar. Sin `model`, valor por defecto `text-embedding-ada-002` ✅ (versión instalada) |
| `dimensions` | Opcional; para la familia `text-embedding-3`, tamaño del vector (documentación oficial) |
| `embed_query(texto)` | `str → list[float]` |
| `embed_documents(textos)` | `list[str] → list[list[float]]` |
| `np.dot(a, b)` | Producto escalar (admite listas de Python ✅) |
| `np.linalg.norm(v)` | Norma del vector |

### Interpretación de la similitud coseno

| Valor | Significado geométrico |
|---|---|
| 1 | Misma dirección |
| 0 | Perpendiculares |
| -1 | Direcciones opuestas |

---

## Explicación

### ¿Por qué se usa el coseno y no otra medida?

La documentación oficial lista tres medidas (coseno, distancia euclídea y producto escalar). El coseno compara el ángulo y no la longitud (✅ multiplicar un vector por 5 no cambia el resultado). El archivo usa el coseno y no discute las alternativas 📘.

### ¿Por qué `embed_query` para los dos textos?

La documentación define `embed_query` como "embeber una consulta" y `embed_documents` para una lista de documentos. El script usa `embed_query` para ambos textos. ⚠️ La documentación consultada no indica si hay diferencias prácticas entre ambos métodos con los modelos de OpenAI; no se afirma ninguna.

### ¿Qué enfoque usar en cada caso?

| Situación | Enfoque |
|---|---|
| Comparar dos textos puntuales | `embed_query` ×2 y similitud coseno (como el script) |
| Muchos textos | `embed_documents` (una llamada con una lista) |
| Buscar por significado en muchos documentos | Guardar los vectores en un *vector store* y buscar (visto en el proyecto del asistente legal) |
| Reducir tamaño/coste de almacenamiento | Parámetro `dimensions` (familia `text-embedding-3`) ⚠️ |

### Lo que este documento no cubre

- Vector stores y búsqueda por similitud (el script no los usa).
- Otros proveedores de embeddings.
- El valor real de la similitud entre los dos textos y la dimensión real (no se ejecutó con la API).
- Normalización de los vectores: la página de `OpenAIEmbeddings` no la menciona.

### Observaciones sobre el archivo

1. El original no tenía comentarios ni `load_dotenv()`; se añadió esto último (ver *Procedencia*).
2. ⚠️ No se conoce el resultado real: depende de la API.
3. `numpy` es una dependencia aparte.
4. El script no comprueba errores (clave ausente, red).
5. El código se respetó: ambas versiones tienen el mismo código ✅, salvo las dos líneas de `load_dotenv`.

---

## Autoevaluación

1. ¿Qué es un embedding?
2. ¿Qué devuelve `embed_query`?
3. ¿Qué indica `len(vec1)`?
4. ¿Para qué sirve la similitud coseno y en qué rango está?
5. ¿Qué significa un resultado cercano a 0?
6. ¿Por qué hay que indicar `model` explícitamente?
7. ¿Qué diferencia hay entre `embed_query` y `embed_documents`?
8. ¿Por qué `load_dotenv()` debe ir antes de crear `OpenAIEmbeddings` y qué pasa si ya hay otra `OPENAI_API_KEY` definida?

### Respuestas

1. Un vector de números de longitud fija que representa el significado de un texto.
2. Una lista de números (`list[float]`).
3. La dimensión del embedding (cuántos números tiene).
4. Mide cuánto se parecen dos vectores por su dirección; va de -1 a 1.
5. Vectores perpendiculares (poco parecidos en dirección).
6. Porque, sin `model`, la clase usa otro modelo por defecto (`text-embedding-ada-002` en la versión instalada), con otra dimensión.
7. `embed_query` convierte un texto en un vector; `embed_documents` convierte una lista de textos en una lista de vectores.
8. Porque el modelo lee la clave al crearse; sin clave falla con *Missing credentials*. Si ya hay otra definida, `load_dotenv()` no la sobrescribe y se usa esa.

---

## Resumen

| Concepto | Idea clave |
|---|---|
| Embedding | Vector que representa el significado de un texto |
| `OpenAIEmbeddings(model=…)` | Modelo de embeddings de OpenAI |
| `embed_query` | Un texto → un vector (`list[float]`) |
| `len(vec)` | Dimensión del embedding |
| Similitud coseno | `(a·b)/(‖a‖‖b‖)`, entre -1 y 1 |
| `load_dotenv()` | Carga la clave del `.env` antes de crear el modelo |

---

## Fuentes consultadas

- Integración `OpenAIEmbeddings`: <https://docs.langchain.com/oss/python/integrations/embeddings/openai>
- Modelos de embeddings (interfaz y similitud): <https://docs.langchain.com/oss/python/integrations/embeddings>
- Archivos de origen: `3.5_embeddings_langchain.py` (comentado) y `3.5_embeddings_langchain_limpio.py` (limpio).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
