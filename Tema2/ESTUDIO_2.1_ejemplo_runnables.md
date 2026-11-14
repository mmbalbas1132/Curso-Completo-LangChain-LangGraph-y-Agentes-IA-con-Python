# 2.1 Introducción a LCEL y Runnables con `RunnableLambda`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.1_ejemplo_runnables.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un **Runnable** y para qué sirve `RunnableLambda`.
2. Cómo convertir una función normal o una `lambda` en un eslabón de una cadena.
3. Cómo encadenar eslabones con el operador `|` (LCEL).
4. Cómo ejecutar la cadena con `.invoke()`.

### Qué necesitas

- Python básico: funciones, `lambda`, *f-strings* y listas.
- `langchain-core` y `python-dotenv` instalados. Este ejemplo **no llama a ningún modelo**, así que no necesita clave de API.

### Mapa del tema

```text
43  →  paso1 (lambda)  →  "Numero 43"  →  paso2 (def)  →  ["Numero 43", "Numero 43"]
        RunnableLambda                      RunnableLambda
        └──────────────── cadena = paso1 | paso2 ────────────────┘
```

### Glosario

| Término | Significado |
|---|---|
| **Runnable** | Tipo de componente que LangChain puede ejecutar y encadenar. Según el archivo, prompts, modelos y *parsers* deben ser Runnables para formar parte de una cadena. 📘 |
| **`RunnableLambda`** | "Adaptador" que convierte una función de Python (o una `lambda`) en un Runnable. 📘 |
| **LCEL** | *LangChain Expression Language*: forma de componer Runnables con el operador `\|`. 📘 |
| **Operador `\|` (tubería)** | Conecta componentes: la salida de uno es la entrada del siguiente, de izquierda a derecha. 📘 |
| **Cadena** | Resultado de unir Runnables con `\|`. |
| **`.invoke(x)`** | Ejecuta un Runnable con la entrada `x`. |
| **`lambda`** | Función anónima de una línea de Python. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-core python-dotenv
```

Ejecución:

```bash
python 2.1_ejemplo_runnables.py
```

---

### Lección 1: importaciones y entorno (sección 1 del archivo)

```python
from dotenv import load_dotenv
import os  # (Nota: 'os' no se usa en este ejemplo, pero se importa por costumbre)

load_dotenv()

from langchain_core.runnables import RunnableLambda
```

| Línea | Qué hace |
|---|---|
| `from dotenv import load_dotenv` / `load_dotenv()` | Carga las variables de un archivo `.env`. Aquí no se usa ninguna clave de API; el archivo lo mantiene "por buena práctica" al inicio de proyectos con LangChain. 📘 |
| `import os` | No se usa en este ejemplo; el propio archivo lo reconoce. |
| `from langchain_core.runnables import RunnableLambda` | Importa el adaptador que convierte funciones en Runnables. |

**Punto clave.** `RunnableLambda` **no ejecuta** la función al crearse: solo la envuelve. La función se ejecuta cuando llamas a `.invoke()` (o `.batch()`, `.stream()`) sobre la cadena. El archivo lo resume así: "estamos construyendo una receta, no cocinando todavía". 📘 ⚠️ *(no contrastado)*

---

### Lección 2: los dos pasos (sección 2 del archivo)

#### Paso 1: con una función `lambda`

```python
paso1 = RunnableLambda(lambda x: f"Numero {x}")
```

- Recibe un valor `x` (aquí será el número `43`).
- Devuelve un `str` con formato: `"Numero 43"`.
- Cambia el tipo: entra un `int` y sale un `str`. Si recibiera `"hola"`, devolvería `"Numero hola"`. 📘

#### Paso 2: con una función clásica (`def`)

```python
def duplicar_texto(texto):
    return [texto] * 2

paso2 = RunnableLambda(duplicar_texto)
```

- Recibe un `str` y devuelve una lista con ese texto repetido dos veces: `"Numero 43"` → `["Numero 43", "Numero 43"]`.
- Podría escribirse como `lambda`, pero `def` documenta mejor la intención y permite reutilizar la función fuera de la cadena. 📘

**Punto clave.** Tanto una `lambda` como una función `def` se envuelven igual con `RunnableLambda`.

---

### Lección 3: la cadena con `|` (sección 3 del archivo)

```python
cadena = paso1 | paso2
```

El símbolo `|` funciona como la tubería de Linux: la salida de `paso1` se convierte en la entrada de `paso2`. El flujo es **estrictamente de izquierda a derecha**. 📘

```text
entrada  →  paso1  →  salida de paso1 (= entrada de paso2)  →  paso2  →  resultado
```

Según el archivo, `paso1 | paso2` devuelve internamente un `RunnableSequence`, que también es un Runnable. Por eso se puede seguir encadenando (`paso1 | paso2 | paso3 | ...`) y `cadena` ofrece `.invoke()`, `.batch()` y `.stream()` igual que cualquier Runnable. 📘 ⚠️ *(no contrastado: las páginas oficiales consultadas no describen `RunnableSequence`)*

---

### Lección 4: ejecutar y ver el resultado (secciones 4 y 5 del archivo)

```python
resultado = cadena.invoke(43)

print(resultado)
```

**Qué ocurre:**

1. `43` entra en `paso1` → `"Numero 43"`.
2. Ese texto entra en `paso2` → `["Numero 43", "Numero 43"]`.
3. `print` muestra el resultado final.

**Salida esperada** (según el archivo):

```text
['Numero 43', 'Numero 43']
```

**Puntos clave de la lección.**

- Todos los Runnables tienen `.invoke()`.
- El valor que pasas a `.invoke()` es la entrada del **primer** eslabón.
- El resultado es la salida del **último** eslabón.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo convierto una función en un eslabón de una cadena?

```python
paso = RunnableLambda(mi_funcion)
```

### ¿Cómo uso una `lambda` como eslabón?

```python
paso1 = RunnableLambda(lambda x: f"Numero {x}")
```

### ¿Cómo encadeno varios pasos?

```python
cadena = paso1 | paso2
```

### ¿Cómo ejecuto la cadena?

```python
resultado = cadena.invoke(43)
```

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Un eslabón recibe un tipo inesperado | La salida del paso anterior no es el tipo que espera el siguiente | Revisar el tipo de entrada y salida de cada paso |
| La función "no se ejecuta" al crear `RunnableLambda` | `RunnableLambda` solo la envuelve | Llamar a `.invoke()` sobre la cadena 📘 |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `from dotenv import load_dotenv` | `python-dotenv` | Cargar `.env` (no imprescindible en este ejemplo) |
| `import os` | Biblioteca estándar | Sin uso en el ejemplo |
| `from langchain_core.runnables import RunnableLambda` | `langchain-core` | Envolver funciones |

### Pasos de la cadena

| Nombre | Tipo | Entrada | Salida |
|---|---|---|---|
| `paso1` | `RunnableLambda(lambda x: ...)` | cualquier valor (aquí `int`) | `str` |
| `duplicar_texto` | Función `def` | `str` | `list[str]` |
| `paso2` | `RunnableLambda(duplicar_texto)` | `str` | `list[str]` |
| `cadena` | `paso1 \| paso2` | `int` | `list[str]` |

### Métodos citados de un Runnable

| Método | Descripción | Confirmado en documentación oficial |
|---|---|---|
| `.invoke(x)` | Ejecuta el Runnable | La referencia lo describe como "Invoke this `Runnable` synchronously" |
| `.batch([...])` | Varias entradas | Sí, para modelos de chat (paraleliza en el cliente) |
| `.stream(x)` | Salida por fragmentos | Sí, para modelos de chat |

### `RunnableLambda`: lo que muestra la referencia oficial

| Elemento | Qué indica la referencia |
|---|---|
| Clase base | `Runnable[Input, Output]` |
| Parámetro `func` | Acepta varios tipos de callable (síncronos, iteradores, asíncronos, con `RunnableConfig`) |
| Parámetro `afunc` | Variantes asíncronas |
| Parámetro `name` | `str` opcional |
| Descripción de la clase | No incluida en la página (solo firmas) |

### `Runnable.pipe`

La referencia oficial de `pipe` describe su parámetro `others` como "Otros `Runnable` u objetos similares a `Runnable` para componer", con tipo `Runnable | Callable`. No menciona el operador `|` ni `RunnableSequence`. ⚠️ *(la relación entre `|` y `pipe` no se pudo confirmar en las páginas consultadas)*

---

## Explicación

### ¿Por qué hace falta `RunnableLambda`?

Porque, según el archivo, LangChain exige que todos los componentes de una cadena sean Runnables. Una función normal de Python no lo es; `RunnableLambda` la adapta. 📘 La referencia oficial es coherente con esto: `pipe` acepta "Runnable u objetos similares a Runnable", incluidos callables, aunque no describe la conversión.

### ¿Por qué se compara con una "receta"?

Porque crear la cadena solo **describe** los pasos; no los ejecuta. La ejecución ocurre con `.invoke()`. Esto permite construir la cadena una vez y usarla muchas veces. 📘

### ¿Por qué la tubería se lee de izquierda a derecha?

Porque la salida de cada componente es la entrada del siguiente, como en la tubería de Linux. 📘

### ¿Cuándo usar `lambda` y cuándo `def`?

| Caso | Recomendación |
|---|---|
| Transformación breve de una línea | `lambda` |
| Lógica reutilizable o que conviene documentar | `def` |

### Lo que este documento no cubre

- Modelos de chat, plantillas o *parsers* dentro de la cadena.
- `.batch()` y `.stream()` en la práctica (solo se citan).
- Una descripción oficial de `RunnableLambda`, `RunnableSequence` y del operador `|`: las páginas consultadas solo traen firmas.

### Observaciones sobre el archivo

- `os` se importa y no se usa (el archivo lo indica).
- `load_dotenv()` se llama aunque el ejemplo no necesita claves de API.

---

## Autoevaluación

1. ¿Qué es un Runnable?
2. ¿Qué hace `RunnableLambda`?
3. ¿Se ejecuta la función al crear un `RunnableLambda`?
4. ¿Qué hace el operador `|`?
5. ¿Qué devuelve `cadena.invoke(43)` y por qué?
6. ¿Qué tipo de dato recibe y cuál devuelve `paso1`?
7. ¿Cuándo es preferible `def` a `lambda`?
8. ¿Necesita este ejemplo una clave de API? ¿Por qué?

### Respuestas

1. Un componente que LangChain puede ejecutar y encadenar.
2. Convierte una función de Python o una `lambda` en un Runnable.
3. No: solo la envuelve; se ejecuta con `.invoke()`.
4. Conecta componentes: la salida de uno pasa como entrada al siguiente, de izquierda a derecha.
5. `['Numero 43', 'Numero 43']`: `paso1` convierte `43` en `"Numero 43"` y `paso2` lo repite dos veces.
6. Recibe un valor (aquí un `int`) y devuelve un `str`.
7. Cuando la lógica conviene documentarla o reutilizarla fuera de la cadena.
8. No: no llama a ningún modelo.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Runnable | Componente ejecutable y encadenable |
| `RunnableLambda` | Convierte funciones en Runnables |
| `\|` | Encadena de izquierda a derecha |
| `.invoke()` | Ejecuta la cadena |
| Cadena | "Receta" que se ejecuta al invocarla |

---

## Fuentes consultadas

- Referencia de `RunnableLambda` (solo firmas): <https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda>
- Referencia de `Runnable.pipe` (parámetro `others`): <https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/pipe>
- Documentación oficial de modelos (`batch`, `stream`): <https://docs.langchain.com/oss/python/langchain/models>
- Material del tema: `2.1_ejemplo_runnables.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
