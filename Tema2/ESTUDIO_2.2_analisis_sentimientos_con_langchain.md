# 2.2 Análisis de sentimientos con LCEL: ejecución en lote con `.batch()`

> **Cómo usar este texto.** Lee primero *Antes de empezar*, sigue el *Tutorial* en orden y consulta la *Guía práctica*, la *Referencia* y la *Explicación* cuando lo necesites. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.2_analisis_sentimientos_con_langchain.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* marca lo que procede solo del archivo y no está confirmado en la documentación oficial. ⚠️ *(no contrastado)* marca lo que **no se pudo confirmar** en las páginas oficiales consultadas. Cuando la documentación oficial **difiere** del archivo, se señala con 🔀 *(discrepancia)*.

---

## Antes de empezar

### Qué vas a aprender

1. Cargar la clave de API desde un archivo `.env` y comprobar que existe antes de usarla (*fail-fast*).
2. Reutilizar una cadena LCEL de análisis de sentimientos.
3. Ejecutar la cadena sobre **varios textos a la vez** con `.batch()`.
4. Controlar la concurrencia (`max_concurrency`) y los errores (`return_exceptions`).
5. Decidir cuándo conviene `.batch()` y cuándo no.

### Qué necesitas

- Python básico: funciones, diccionarios, *f-strings*, `try/except`, listas.
- Conocer `RunnableLambda`, `RunnableParallel` y el operador `|`. El archivo los da por conocidos de la "Parte 1" y solo los repasa.
- Una clave de API de OpenAI.

### Mapa del tema

```text
texto_entrada
    │
    ▼
[preprocessor]        limpia y recorta a 500 caracteres
    │
    ▼
[RunnableParallel]    ┬─ resumen
                      └─ sentimiento_data
    │
    ▼
[merger]              aplana el resultado
    │
    ▼
{resumen, sentimiento, razon}
```

**La novedad** respecto a la Parte 1 no está en la cadena (es la misma), sino en **cómo se invoca**:

| | Parte 1 | Parte 2 (este archivo) |
|---|---|---|
| Método | `chain.invoke(texto)` | `chain.batch([t1, t2, t3])` |
| Entradas → salidas | 1 → 1 | N → N |
| Extra | — | Carga de secretos con `load_dotenv` y verificación de `OPENAI_API_KEY` |

### Glosario

| Término | Significado en este tema |
|---|---|
| **Lote (batch)** | Lista de entradas que se procesan en una sola llamada. |
| **`.batch()`** | Método que ejecuta la cadena sobre una lista de entradas y devuelve una lista de resultados. |
| **Concurrencia** | Varias tareas en curso a la vez. |
| **`max_concurrency`** | Número máximo de llamadas en paralelo. |
| **`return_exceptions`** | Opción de `.batch()` para recibir los errores como valores en lugar de abortar. |
| **Rate limit (429)** | Código de respuesta del proveedor cuando recibe demasiadas peticiones. |
| **Fail-fast** | Principio: comprobar los requisitos al inicio y fallar con un mensaje claro. |
| **Fallback** | Valor de reserva que se devuelve cuando algo falla. |
| **`.env` / `load_dotenv()`** | Archivo con claves `CLAVE=valor` y función que las carga en `os.environ`. |
| **Runnable** | Componente ejecutable y encadenable de LangChain. |
| **`RunnableLambda` / `RunnableParallel`** | Envuelve una función / agrupa ramas con nombre. |
| **`os.getenv("X")`** | Lee una variable de entorno; devuelve `None` si no existe. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-openai langchain-core python-dotenv
```

Archivo `.env` en la carpeta de trabajo:

```bash
OPENAI_API_KEY=tu_clave
```

Ejecución del script:

```bash
python 2.2_analisis_sentimientos_con_langchain.py
```

---

### Lección 1: secretos y modelo (secciones 1–3 del archivo)

**Objetivo.** Cargar la clave de forma segura, comprobarla y crear el modelo.

```python
import os
import json

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

| Bloque | Qué hace y por qué |
|---|---|
| `import os`, `import json` | `os` para leer variables de entorno; `json` para interpretar la respuesta del modelo. Las importaciones siguen el orden: biblioteca estándar → terceros. 📘 |
| `load_dotenv()` | Carga el `.env` en `os.environ`. Según el archivo: busca `.env` en el directorio actual y en los padres, y **solo escribe las claves que no existan ya** (para que un `.env` antiguo no pise secretos inyectados por la plataforma). Para forzar la sobrescritura existe `override=True`. 📘 ⚠️ *(`python-dotenv` no es de LangChain; no está en las fuentes permitidas)* |
| `if not os.getenv(...)`: `raise RuntimeError(...)` | **Fail-fast.** Si la clave falta, el programa se detiene aquí con un mensaje claro, en lugar de fallar más adelante con un error de autenticación difícil de localizar. 📘 |
| `ChatOpenAI(model="gpt-4o-mini", temperature=0)` | Crea el modelo. Lee `OPENAI_API_KEY` del entorno automáticamente, por lo que no hace falta pasar `api_key=...`. |

**Detalles que aporta el archivo.** 📘

- `os.getenv("X")` devuelve `None` si no existe; `os.environ["X"]` lanzaría `KeyError`. Se prefiere `getenv` para lanzar un mensaje propio.
- No imprimas la clave en *logs*, ni siquiera truncada.
- Añade `.env` a `.gitignore` antes del primer *commit*.
- `load_dotenv()` se puede llamar varias veces sin efectos colaterales.

**Punto clave.** `load_dotenv()` va **antes** de usar la clave, y la comprobación debe hacerse antes de la primera llamada a la API.

---

### Lección 2: la cadena (secciones 4–8 del archivo)

**Objetivo.** Montar la misma cadena que en la Parte 1. El archivo indica que no cambia: *"cada eslabón es independiente"* y `.batch()` ejecuta la cadena completa por cada elemento. 📘

#### 2.1 Preprocesador

```python
def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud."""
    return text.strip()[:500]

preprocessor = RunnableLambda(preprocess_text)
```

- Elimina espacios en los extremos y recorta a 500 caracteres.
- En un lote, **cada texto se recorta por separado**.
- `preprocessor` ya es un Runnable, no una función: no se llama como `preprocessor(texto)`, sino con `.invoke()`, `.batch()`, `.stream()` o combinándolo con `|`. 📘

#### 2.2 Rama de resumen

```python
def generate_summary(text):
    """Genera un resumen conciso del texto."""
    prompt = f"Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content

summary_branch = RunnableLambda(generate_summary)
```

Devuelve el texto del `AIMessage` (`response.content`). El archivo describe el prompt como "Rol + Tarea + Restricción" (R-T-C) y remite a la Parte 1 para su explicación; ese detalle no está desarrollado en este archivo. 📘

#### 2.3 Rama de sentimiento

```python
def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado."""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}

    Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}

sentiment_branch = RunnableLambda(analyze_sentiment)
```

| Parte | Qué hace |
|---|---|
| Prompt | Pide **únicamente** un JSON con las claves `sentimiento` y `razon`. Las llaves dobles `{{ }}` producen llaves literales dentro de un *f-string*. |
| `json.loads(...)` | Convierte la respuesta en un diccionario. |
| `except json.JSONDecodeError` | Devuelve un valor de reserva con la misma estructura, para que el resultado conserve su forma aunque el JSON falle. |

**Por qué el `try/except` es crítico en un lote.** Según el archivo: sin él, `json.loads` lanzaría una excepción que se propagaría por `RunnableLambda` → `RunnableParallel` → `chain.batch`, y se perderían **todos** los resultados del lote, no solo el del texto problemático. 📘

#### 2.4 Unificador

```python
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado."""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"],
    }

merger = RunnableLambda(merge_results)
```

`data` tiene la forma `{"resumen": str, "sentimiento_data": {"sentimiento": ..., "razon": ...}}`. La función aplana ese anidamiento. Si renombras las claves en el `RunnableParallel`, debes actualizarlas aquí. No hace llamadas al modelo, por lo que su coste es despreciable. 📘

#### 2.5 Paralelo y cadena final

```python
parallel_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch,
})

chain = preprocessor | parallel_analysis | merger
```

El archivo indica que el operador `|` construye un `RunnableSequence`, que a su vez es un Runnable; por eso `chain` ofrece los mismos métodos que cualquier Runnable. 📘 ⚠️ *(no contrastado: las páginas oficiales consultadas no describen `RunnableSequence`, `RunnableParallel` ni el operador `|`)*

**Puntos clave de la Lección 2.**

- La cadena es idéntica a la de la Parte 1.
- Cada rama captura sus propios errores con un valor de reserva.
- El nombre de las ramas es el contrato con `merge_results`.

---

### Lección 3: ejecutar en lote (sección 9 del archivo)

**Objetivo.** Procesar tres reseñas con una sola llamada.

```python
reviews_batch = [
    "Excelente producto, muy satisfecho con la compra",
    "Terrible calidad, no lo recomiendo para nada",
    "Está bien, cumple su función básica pero nada especial",
]

resultado_batch = chain.batch(reviews_batch)

print(resultado_batch)
```

#### Qué devuelve `.batch()`

Una **lista del mismo tamaño que la entrada, en el mismo orden**: con 3 reseñas, 3 diccionarios, cada uno con `resumen`, `sentimiento` y `razon`. El orden se conserva aunque internamente los elementos terminen desordenados. 📘 ⚠️ *(la documentación consultada no afirma que `.batch()` conserve el orden; sí indica que `batch_as_completed()` puede devolver resultados desordenados e incluye el índice de cada entrada)*

#### Comparación con un bucle `.invoke()`

| | Bucle con `.invoke()` | `.batch()` |
|---|---|---|
| Ejecución | Secuencial (una tras otra) | Concurrente |
| Latencia | Suma de las latencias | Aproximadamente la de la más lenta (con límites) |
| Control de concurrencia | No | `max_concurrency` |
| Errores | Rompen el bucle | `return_exceptions` los encapsula |
| Código | Imperativo (`for`) | Declarativo (una línea) |
| Coste (tokens) | Igual | Igual |

Todo lo anterior procede del archivo 📘. La documentación oficial confirma que `.batch()` **paraleliza las llamadas al modelo en el cliente** y que el procesamiento en paralelo puede mejorar el rendimiento.

#### Control de la concurrencia: `max_concurrency`

Número máximo de elementos procesados a la vez. Según el archivo:

- Por defecto no hay límite (todos a la vez), con riesgo de errores 429.
- Se recomienda un valor de 5 a 10 para OpenAI.
- Con lotes de miles de elementos es obligatorio poner un límite. 📘

🔀 *(discrepancia)* El archivo lo escribe así:

```python
resultados_batch_5 = chain.batch(reviews_batch, max_concurrency=5)
```

La documentación oficial lo muestra dentro del diccionario `config` (`RunnableConfig`):

```python
resultados = chain.batch(reviews_batch, config={"max_concurrency": 5})
```

La referencia oficial de `.batch()` indica que `max_concurrency` es una clave de `config` "para controlar cuánto trabajo se hace en paralelo". Usa la forma oficial.

#### Control de errores: `return_exceptions`

- Con `return_exceptions=False` (valor por defecto): si **cualquier** elemento lanza una excepción, `.batch()` la lanza y pierdes todos los resultados. 📘
- Con `return_exceptions=True`: los elementos fallidos devuelven la **excepción como valor** y los demás su resultado normal. 📘

```python
res = chain.batch(items, return_exceptions=True)
for r in res:
    if isinstance(r, Exception):
        print("Falló:", r)
    else:
        print(r["sentimiento"])
```

La documentación oficial lista `return_exceptions` como parámetro de `.batch()` con valor por defecto `False`, pero las páginas consultadas no describen su comportamiento. ⚠️ *(no contrastado)*

**Puntos clave de la Lección 3.**

- `.batch()` recibe una lista y devuelve una lista.
- Hay dos mecanismos de seguridad distintos: el *fallback* dentro de cada rama y `return_exceptions` en el lote.
- En producción, limita siempre la concurrencia.

---

### Lección 4: medir bucle frente a `.batch()` (sección 10 del archivo)

**Objetivo.** Comprobar con tiempos reales la diferencia entre las dos formas de ejecutar.

El bloque está desactivado con `if False:`. Cambia a `True` para ejecutarlo.

```python
if False:  # Cambia a True para probar
    import time

    # --- OPCIÓN A: bucle con .invoke (secuencial) ---
    t0 = time.perf_counter()
    resultados_loop = [chain.invoke(r) for r in reviews_batch]
    t_loop = time.perf_counter() - t0

    # --- OPCIÓN B: .batch (concurrente) ---
    t0 = time.perf_counter()
    resultados_batch = chain.batch(reviews_batch)
    t_batch = time.perf_counter() - t0

    # --- OPCIÓN C: .batch con límite de concurrencia ---
    t0 = time.perf_counter()
    # resultados_batch_5 = chain.batch(reviews_batch, max_concurrency=5)
    resultados_batch_5 = chain.batch(reviews_batch, config={"max_concurrency": 5})
    t_batch_5 = time.perf_counter() - t0

    # --- OPCIÓN D: .batch con errores encapsulados ---
    t0 = time.perf_counter()
    resultados_batch_exc = chain.batch(reviews_batch, return_exceptions=True)
    t_batch_exc = time.perf_counter() - t0

    print(f"Bucle .invoke:      {t_loop:.2f}s")
    print(f".batch():           {t_batch:.2f}s  ← ganancia ≈ {t_loop/t_batch:.1f}x")
    print(f".batch(max=5):      {t_batch_5:.2f}s")
    print(f".batch(exc=True):   {t_batch_exc:.2f}s")
    print(f"Resultados idénticos?: {resultados_loop == resultados_batch}")
```

> 🔀 La opción C usa `max_concurrency=5` como argumento directo, tal como está en el archivo. Consulta la discrepancia de la Lección 3.

**Cómo interpretar el ejemplo de latencia del archivo** (3 elementos, "~2 s por elemento"; es una **estimación didáctica** del archivo, no una medición) 📘:

| Configuración | Tiempo estimado | Razón |
|---|---|---|
| Bucle `.invoke()` | ~6 s | Secuencial |
| `.batch()` | ~2 s | Paralelo, si no hay *throttling* |
| `.batch(max=1)` | ~6 s | Una a la vez |
| `.batch(max=2)` | ~4 s | 2 + 1 |
| `.batch(max=5)` | ~2 s | Los 3 elementos caben |

La latencia depende del mínimo entre el número de elementos y `max_concurrency`.

**Nota sobre la comparación de resultados.** El archivo señala que `resultados_loop == resultados_batch` puede ser `True` o `False`: con `temperature=0` debería ser `True`, pero la API no garantiza un determinismo perfecto. 📘

🔀 *(discrepancia interna del archivo)* En la sección 3 el archivo afirma que `temperature=0` "garantiza que la misma entrada produce la misma salida en cualquier momento"; en la sección 10 afirma lo contrario. Este documento adopta la versión prudente: **no está garantizado**.

---

### Ejercicios propuestos en el archivo

| | Ejercicio |
|---|---|
| A | Añade un cuarto texto al lote. ¿Se mantiene el orden? ¿Cuánto tarda frente a los tres primeros? |
| B | Añade un texto vacío (`""`). ¿Qué sentimiento detecta y por qué? Pista del archivo: piensa en lo que hace `preprocess_text`. |
| C | Activa el bloque de la Lección 4 y explica la diferencia entre las cuatro opciones. |
| D | Prueba `max_concurrency=1`, `2` y `5` y dibuja una tabla de tiempo frente a `max_concurrency`. |
| E | Añade un texto con muchos `{` y `}`. ¿Cae en el `except` interno o rompe `.batch()`? Justifica el resultado. |
| F | (Avanzado) Sustituye `.batch` por `await chain.abatch(...)` dentro de un `async def` con `asyncio.run(...)`. Compara los tiempos. |
| G | Comenta `OPENAI_API_KEY` en el `.env` y ejecuta: deberías ver el `RuntimeError` antes de cualquier llamada a la API. |

---

## Guía práctica

### ¿Cómo evito que el programa falle con un error críptico por falta de clave?

```python
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )
```

### ¿Cómo analizo varios textos a la vez?

```python
resultado_batch = chain.batch(reviews_batch)
```

### ¿Cómo limito cuántas peticiones se lanzan a la vez?

Con la forma oficial (clave `max_concurrency` en `config`):

```python
resultado = chain.batch(reviews_batch, config={"max_concurrency": 5})
```

### ¿Cómo evito perder todo el lote por un solo error?

Dos mecanismos que se pueden combinar:

1. **Fallback dentro de la rama** (`try/except` con un valor por defecto), que mantiene el formato uniforme. El archivo lo prefiere porque así `.batch()` no necesita `return_exceptions`. 📘
2. **`return_exceptions=True`** en la llamada, y comprobar `isinstance(r, Exception)` en cada resultado.

### ¿Cómo añado un dato común a todo el lote (por ejemplo un identificador)?

Después del `.batch()`, no dentro de `merge_results` (que se ejecutaría una vez por elemento) 📘:

```python
[{**r, "job_id": jid} for r in res]
```

### ¿Cómo obtengo los resultados según van terminando?

El archivo menciona `.batch_as_completed()` (devuelve los resultados a medida que se completan, no en el orden original). La documentación oficial lo describe para los modelos de chat: devuelve cada salida al terminar, puede llegar desordenada e incluye el índice de la entrada para reconstruir el orden.

### ¿Cómo mido si `.batch()` compensa?

Compara con `time.perf_counter()` el bucle `.invoke()` y `.batch()` (Lección 4). El archivo recomienda esta función en lugar de `time.time()`. 📘

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `RuntimeError: Falta OPENAI_API_KEY` | La variable no está en el `.env` ni en el entorno | Definirla y volver a ejecutar |
| Se pierden todos los resultados del lote | Una excepción se propagó por la cadena | *Fallback* en la rama y/o `return_exceptions=True` |
| Errores 429 en lotes grandes | Demasiadas peticiones simultáneas | Limitar con `config={"max_concurrency": N}`; subir `max_retries` del modelo |
| Los resultados llegan en orden distinto | Se usó `batch_as_completed()` | Usar el índice incluido en cada resultado, o usar `.batch()` |
| Un texto vacío da un sentimiento inesperado | El preprocesador no valida el contenido | Ver ejercicio B |
| Un texto muy largo se analiza solo en parte | El preprocesador recorta a 500 caracteres | Ajustar el límite en `preprocess_text` |

Sobre `max_retries`: la documentación oficial indica que su valor por defecto es 6, con *backoff* exponencial y *jitter*, y que reintenta errores de red, 429 y 5xx, pero no errores de cliente como 401 o 404.

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `import os` | Biblioteca estándar | Variables de entorno |
| `import json` | Biblioteca estándar | Interpretar la respuesta JSON |
| `from dotenv import load_dotenv` | `python-dotenv` | Cargar `.env` |
| `from langchain_core.runnables import RunnableLambda, RunnableParallel` | `langchain-core` | Componentes de la cadena |
| `from langchain_openai import ChatOpenAI` | `langchain-openai` | Modelo de chat |

### Componentes de la cadena

| Nombre | Tipo | Entrada | Salida |
|---|---|---|---|
| `preprocessor` | `RunnableLambda(preprocess_text)` | `str` | `str` (máx. 500 caracteres) |
| `summary_branch` | `RunnableLambda(generate_summary)` | `str` | `str` |
| `sentiment_branch` | `RunnableLambda(analyze_sentiment)` | `str` | `dict` (`sentimiento`, `razon`) |
| `parallel_analysis` | `RunnableParallel({...})` | `str` | `dict` (`resumen`, `sentimiento_data`) |
| `merger` | `RunnableLambda(merge_results)` | `dict` | `dict` (`resumen`, `sentimiento`, `razon`) |
| `chain` | `preprocessor \| parallel_analysis \| merger` | `str` | `dict` |

### Métodos de un Runnable citados en el archivo

| Método | Descripción según el archivo | Confirmado en documentación oficial |
|---|---|---|
| `.invoke(x)` | 1 elemento | Sí (modelos de chat: devuelve un `AIMessage`) |
| `.batch([x, y, z])` | N elementos, concurrente | Sí, paraleliza en el cliente (modelos de chat) |
| `.stream(x)` | 1 elemento, token a token | Sí (modelos de chat) |
| `.ainvoke(x)` | 1 elemento, asíncrono | Solo la firma en la referencia |
| `.abatch([...])` | N elementos, asíncrono y concurrente | Solo la firma; acepta `config` con `max_concurrency` |
| `.astream(x)` | 1 elemento, asíncrono y en streaming | No consultado |
| `.batch_as_completed([...])` | Devuelve según terminan | Sí (modelos de chat) |

### Parámetros de `.batch()`

| Parámetro | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `inputs` | lista | — | Entradas del lote |
| `config` | `RunnableConfig` (diccionario) | `None` | Admite, entre otras, la clave `max_concurrency` |
| `return_exceptions` | `bool` | `False` | Según el archivo, devuelve las excepciones como valores 📘 |

### Parámetros del modelo citados

| Parámetro | Valor | Notas |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Parámetro usado en la documentación oficial |
| `temperature` | `0` | Según el archivo, busca determinismo en clasificación 📘 |
| `max_retries` | Ejemplo: `3` | Documentación oficial: por defecto 6 |

### Claves de los diccionarios

| Diccionario | Claves |
|---|---|
| Salida de `analyze_sentiment` | `sentimiento`, `razon` |
| Salida de `parallel_analysis` | `resumen`, `sentimiento_data` |
| Salida final de `chain` | `resumen`, `sentimiento`, `razon` |
| Valor de reserva | `{"sentimiento": "neutro", "razon": "Error en análisis"}` |

### Patrones reutilizables (resumen del archivo)

| # | Patrón | Idea |
|---|---|---|
| 1 | Secretos vía `.env` | `load_dotenv()` + *fail-fast* con `os.getenv` |
| 2 | Mismo Runnable, múltiples métodos | `invoke`, `batch`, `stream`, `ainvoke`, `abatch`, `astream` sin tocar la cadena |
| 3 | Batch con concurrencia controlada | Limitar siempre en producción |
| 4 | Batch con errores encapsulados | `return_exceptions=True` + `isinstance(r, Exception)` |
| 5 | *Fallback* en la rama, no en el batch | El contrato de datos se mantiene uniforme |
| 6 | Medir antes de optimizar | `time.perf_counter()` |
| 7 | Orden preservado | Para orden de llegada, `batch_as_completed()` |

---

## Explicación

### ¿Por qué `.batch()` en lugar de un bucle?

Porque ejecuta los elementos a la vez en lugar de uno tras otro, con menos código y con control de concurrencia y de errores. El coste en tokens **no cambia**: se paga lo mismo. 📘 La documentación oficial respalda que la ejecución en paralelo puede mejorar el rendimiento.

### ¿Por qué la misma cadena sirve para `.invoke()`, `.batch()` y `.stream()`?

El archivo lo llama "contrato Runnable" o "polimorfismo de Runnable": todas las cadenas heredan los mismos métodos, así que se aprende `.invoke` una vez y los demás funcionan igual. Cambiar de método no obliga a tocar la definición de la cadena. 📘

### ¿Por qué se comprueba la clave al inicio (*fail-fast*)?

Si falta, el error aparecería varias capas más abajo como un error de autenticación que no indica dónde está el problema. Comprobarlo al inicio lo convierte en un mensaje accionable. 📘

### ¿Por qué `load_dotenv()` no sobrescribe variables ya definidas?

En producción los secretos suelen venir inyectados por la plataforma; un `.env` obsoleto no debe pisarlos. 📘 ⚠️ *(comportamiento de `python-dotenv`, no de LangChain)*

### ¿Por qué el *fallback* va dentro de la rama?

Mantiene un resultado uniforme (siempre un diccionario con las mismas claves) y evita que un texto "raro" rompa el lote. Así `.batch()` casi no necesita `return_exceptions`. 📘

### ¿Cómo puede ser concurrente si las funciones usan `llm.invoke` (síncrono)?

El archivo afirma que `RunnableLambda` ejecuta las funciones síncronas en un *thread pool*, de modo que cada llamada bloquea su propio hilo pero no el bucle de eventos principal, y que con `async def` + `await llm.ainvoke()` + `chain.abatch(...)` se evita el coste de los hilos. También afirma que `.batch()` es internamente asíncrono. 📘 ⚠️ *(no contrastado: la documentación oficial consultada no describe estos mecanismos internos; trátalos como la explicación del curso, no como un hecho verificado)*

### ¿Cuándo **no** conviene `.batch()`? (según el archivo) 📘

- Si importa la latencia de cada elemento más que el orden → `batch_as_completed()`.
- Si necesitas streaming por elemento → `astream` en un bucle.
- Si los textos dependen unos de otros.
- Si la API o el modelo local no soportan bien muchas peticiones concurrentes.

### Riesgo de lotes grandes

El archivo advierte de que `.batch()` no avisa si el lote es enorme: con 10 000 elementos intentaría lanzar 10 000 peticiones a menos que se limite la concurrencia, con riesgo de agotar cuotas o colgar la máquina. 📘 ⚠️ *(no contrastado; la documentación oficial no indica cuál es el límite por defecto)*

### Lo que este documento no cubre

- La **Parte 1** del análisis de sentimientos (`.invoke()`, prompt R-T-C, justificación de `temperature=0`): el archivo la menciona pero no es fuente de este documento.
- La Parte 3 ("streaming"), que el archivo anuncia como futura.
- Una descripción oficial de `RunnableLambda`, `RunnableParallel`, `RunnableSequence` y el operador `|`.
- El comportamiento oficial de `return_exceptions` y de los mecanismos internos de concurrencia.

### Observaciones sobre el archivo

- `merge_results` accede directamente a `["sentimiento"]` y `["razon"]`. El valor de reserva cubre un JSON **inválido**; un JSON válido sin esas claves no tendría protección en el código.
- El archivo atribuye a la ausencia de llaves dobles un `KeyError`. No se reproduce esa afirmación porque no se pudo verificar con las fuentes permitidas.
- El archivo titula "silent truncation" a la advertencia sobre lotes enormes; describe un exceso de peticiones, no un truncado.

---

## Autoevaluación

1. ¿Qué cambia entre la Parte 1 y la Parte 2: la cadena o la forma de invocarla?
2. ¿Qué hace `load_dotenv()` y por qué va antes de comprobar la clave?
3. ¿Qué es el *fail-fast* y qué ventaja aporta?
4. ¿Qué devuelve `chain.batch(reviews_batch)` si la lista tiene 3 elementos?
5. ¿Qué ocurre, por defecto, si un elemento del lote lanza una excepción?
6. ¿Qué hace `return_exceptions=True`?
7. ¿Cómo se limita la concurrencia según la documentación oficial?
8. ¿Cambia el coste en tokens al usar `.batch()` en lugar de un bucle?
9. ¿Por qué el *fallback* se coloca dentro de cada rama?
10. ¿Cuándo es preferible `batch_as_completed()`?

### Respuestas

1. Solo la forma de invocarla: la cadena es idéntica.
2. Carga el `.env` en `os.environ` (sin sobrescribir las variables ya definidas, según el archivo). Va antes porque la comprobación lee esa variable.
3. Comprobar los requisitos al inicio y fallar con un mensaje claro; evita errores de autenticación difíciles de localizar.
4. Una lista de 3 diccionarios, en el mismo orden, cada uno con `resumen`, `sentimiento` y `razon`.
5. `.batch()` lanza la excepción y se pierden todos los resultados.
6. Devuelve las excepciones como valores en las posiciones fallidas y el resultado normal en el resto.
7. Con la clave `max_concurrency` dentro de `config`, por ejemplo `config={"max_concurrency": 5}`.
8. No: se paga lo mismo por token; cambia la latencia.
9. Para mantener el resultado uniforme y que un texto problemático no rompa el lote.
10. Cuando importa recibir cada resultado en cuanto termina (por ejemplo, para mostrar progreso), aceptando que puede llegar fuera de orden.

---

## Resumen del tema

| Concepto | En una línea |
|---|---|
| `.batch()` | Ejecuta la cadena sobre una lista y devuelve una lista en el mismo orden |
| `max_concurrency` | Limita las llamadas simultáneas; se pasa en `config` |
| `return_exceptions` | Devuelve las excepciones como valores en lugar de abortar |
| *Fallback* en la rama | Mantiene el formato aunque falle el JSON |
| `load_dotenv()` + `os.getenv` | Carga la clave y la comprueba al inicio |
| *Fail-fast* | Fallar pronto con un mensaje claro |
| Contrato Runnable | Una cadena ofrece `invoke`, `batch`, `stream` y sus versiones asíncronas |
| `batch_as_completed()` | Resultados según terminan, posiblemente desordenados |
| `time.perf_counter()` | Forma recomendada de medir tiempos |

---

## Fuentes consultadas

- Documentación oficial de `ChatOpenAI` (variable `OPENAI_API_KEY`, parámetro `model`): <https://python.langchain.com/docs/integrations/chat/openai/> (redirige a <https://docs.langchain.com/oss/python/integrations/chat/openai>)
- Documentación oficial de modelos (`batch`, `batch_as_completed`, `max_concurrency`, `max_retries`): <https://docs.langchain.com/oss/python/langchain/models>
- Referencia de API de `Runnable.batch` (parámetros `config` y `return_exceptions`): <https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/batch>
- Referencia de API de `Runnable.batch_as_completed` y `Runnable.abatch` (solo firmas y parámetros): <https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/batch_as_completed>, <https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/abatch>
- Material del tema: `2.2_analisis_sentimientos_con_langchain.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
