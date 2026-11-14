# 2.8–2.9 Salida estructurada: de `BaseModel` de Pydantic a `with_structured_output`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento único elaborado a partir de las dos partes `2.8_output_parsers_parte1.py` y `2.9_output_parsers_parte2.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo de los archivos; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice. ✅ *(ejecutado)* = comportamiento observado al ejecutar el código (Pydantic 2.14.3). **Pydantic no es de LangChain** y no está entre las fuentes permitidas: sus comportamientos se confirmaron ejecutándolos, no con documentación. La parte 2 **no se pudo ejecutar** porque necesita una clave de OpenAI; su salida real no se muestra.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un **esquema de datos** y cómo definirlo con `BaseModel` de Pydantic.
2. Cómo Pydantic **valida y convierte** los datos de entrada y asigna valores por defecto.
3. Cómo describir cada campo con `Field(description=...)`.
4. Cómo pedir a un modelo de chat que devuelva un **objeto tipado** en lugar de texto libre con `with_structured_output`.
5. Cómo leer y serializar el resultado (`model_dump_json`).

### Qué necesitas

- Python básico (clases, anotaciones de tipo, diccionarios).
- Conocer `ChatOpenAI` y `.invoke()` (documentos 2.2 y 2.3).
- Para la **parte 1** no necesitas clave de API. Para la **parte 2** necesitas `OPENAI_API_KEY`.

### Mapa del tema

```text
PARTE 1 (2.8) — solo Pydantic
  dict {"id": "123", "nombre": "Ana"}  ──►  Usuario(**data)  ──►  objeto validado  ──►  JSON
                                           (convierte "123" → 123, añade activo=True)

PARTE 2 (2.9) — Pydantic + LangChain
  AnalisisTexto (esquema con Field)  ──►  llm.with_structured_output(AnalisisTexto)
        ──►  structured_llm.invoke("Analiza...")  ──►  instancia de AnalisisTexto  ──►  JSON
```

### Qué aporta cada parte

| | Parte 1 (2.8) | Parte 2 (2.9) |
|---|---|---|
| Archivo | `2.8_output_parsers_parte1.py` | `2.9_output_parsers_parte2.py` |
| Idea | Qué hace Pydantic con los datos | Usar un esquema Pydantic como **contrato de salida** del modelo |
| Librerías | `pydantic` | `pydantic`, `langchain_openai`, `dotenv` |
| ¿Llama a un modelo? | No | Sí (`gpt-4o-mini`) |
| Novedad | `BaseModel`, validación, valor por defecto, `model_dump_json()` | `Field`, `with_structured_output`, resultado tipado |

> **Nota sobre el nombre.** Los archivos se llaman *output parsers*, pero **ninguno usa una clase de *output parser* de LangChain**: la parte 1 es Pydantic puro y la parte 2 usa `with_structured_output`.

### Glosario

| Término | Significado |
|---|---|
| **Esquema** | Descripción de qué campos forman un objeto, de qué tipo son y qué significan. |
| **Pydantic / `BaseModel`** | Biblioteca y clase base para definir esquemas que **validan** los datos. |
| **Validación** | Comprobar (y, si es posible, convertir) que los datos cumplen el esquema. |
| **Campo opcional** | Campo con valor por defecto (`activo: bool = True`). |
| **`ValidationError`** | Error que lanza Pydantic cuando los datos no cumplen el esquema. |
| **`Field(description=...)`** | Añade metadatos a un campo; aquí, una descripción que el modelo puede leer. |
| **JSON Schema** | Descripción en JSON de la estructura de un objeto. |
| **Salida estructurada** | Respuesta del modelo con una forma fija (un objeto) en lugar de texto libre. |
| **`with_structured_output(Esquema)`** | Envuelve el modelo para que devuelva instancias del esquema. |
| **`model_dump_json()`** | Método de Pydantic v2 que serializa el objeto a un `str` JSON. |
| **`model_dump()`** | Método de Pydantic v2 que devuelve un `dict`. |

---

## Tutorial

### Preparación

```bash
pip install -U pydantic langchain-openai python-dotenv
```

Archivo `.env` (solo para la parte 2):

```bash
OPENAI_API_KEY=tu_clave
```

---

### Lección 1 (parte 1, `2.8`): tu primer esquema con Pydantic

**Objetivo.** Ver cómo Pydantic valida y convierte los datos.

```python
from pydantic import BaseModel

class Usuario(BaseModel):
    id: int
    nombre: str
    activo: bool = True

data = {"id": "123", "nombre": "Ana"}

usuario = Usuario(**data)

print(usuario.model_dump_json())
```

| Pieza | Qué hace |
|---|---|
| `from pydantic import BaseModel` | Importa la clase fundamental de Pydantic. |
| `class Usuario(BaseModel)` | Heredar de `BaseModel` convierte la clase en un "validador de datos": define la estructura y **fuerza** que los datos cumplan las reglas. 📘 |
| `id: int` | Exige un entero. Si llega el texto `"123"`, lo convierte. 📘 ✅ |
| `nombre: str` | Exige un texto. |
| `activo: bool = True` | Booleano **opcional** porque tiene valor por defecto. 📘 |
| `data = {"id": "123", "nombre": "Ana"}` | Datos de entrada con una "trampa": `id` es un **texto** y falta `activo`. |
| `Usuario(**data)` | `**data` desempaqueta el diccionario; equivale a `Usuario(id="123", nombre="Ana")`. |
| `usuario.model_dump_json()` | Convierte el objeto validado de nuevo a un `str` JSON. |

**Qué ocurre dentro de `Usuario(**data)`:**

1. Pydantic ve que `id` es un `str` y esperaba un `int`: **hace la conversión**.
2. Pydantic ve que `activo` no está: le asigna el valor por defecto (`True`).

**Salida real** ✅:

```text
{"id":123,"nombre":"Ana","activo":true}
```

Fíjate en que `id` ya es el número `123` (sin comillas) y que `activo` aparece como `true`.

**Comportamientos adicionales comprobados al ejecutar** ✅:

| Entrada | Resultado |
|---|---|
| `Usuario(id="abc", nombre="x")` | `ValidationError` (no se puede convertir a entero) |
| `Usuario(nombre="x")` (falta `id`, que no tiene valor por defecto) | `ValidationError` por campo ausente |
| `usuario.model_dump()` | Un `dict` de Python: `{'id': 123, 'nombre': 'Ana', 'activo': True}` |

**Puntos clave de la Lección 1.**

- Un esquema Pydantic **valida y convierte**; no es solo una descripción.
- Un campo con valor por defecto es opcional; uno sin él es obligatorio.
- `model_dump_json()` devuelve un `str`; `model_dump()` devuelve un `dict`.

---

### Lección 2 (parte 2, `2.9`): importaciones y entorno

```python
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

load_dotenv()
```

| Línea | Qué hace |
|---|---|
| `import os` | El archivo explica que las librerías leen `OPENAI_API_KEY` del entorno; `os` no se usa directamente en este script. 📘 |
| `load_dotenv()` | Carga el `.env` en el entorno. Debe llamarse **antes** de crear el cliente; sin la clave, `ChatOpenAI` da un error de autenticación. 📘 |
| `BaseModel, Field` | `Field` añade descripciones y reglas a cada campo. |
| `ChatOpenAI` | Clase de LangChain sobre el cliente de OpenAI, con la misma interfaz (`invoke`, `stream`, `batch`) que el resto de proveedores. 📘 |

---

### Lección 3 (parte 2): el esquema como contrato de salida

```python
class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")

    sentimiento: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo)")
```

| Campo | Tipo | Descripción que ve el modelo |
|---|---|---|
| `resumen` | `str` | "Resumen breve del texto." |
| `sentimiento` | `str` | "Sentimiento del texto (Positivo, neutro o negativo)" |

- Los dos campos son **obligatorios** (no tienen valor por defecto). 📘
- Si el modelo intentara poner un número en un campo `str`, Pydantic lanza `ValidationError`. 📘 ✅ *(comprobado al ejecutar: un entero en un campo `str` falla)*
- **La `description` no es decorativa: es prompt.** Según el archivo, esa frase viaja dentro del JSON Schema que se envía al modelo y este la lee como instrucción; cuanto más clara, mejor la extracción. 📘 ⚠️ *(la documentación oficial muestra `Field(description=...)` pero no explica cómo se usa en tiempo de ejecución; sí se comprobó que Pydantic incluye la descripción en el JSON Schema del esquema* ✅*)*

**JSON Schema que genera Pydantic para este esquema** ✅:

```json
{"properties": {"resumen": {"description": "Resumen breve del texto.", "title": "Resumen", "type": "string"}, "sentimiento": {"description": "Sentimiento del texto (Positivo, neutro o negativo)", "title": "Sentimiento", "type": "string"}}, "required": ["resumen", "sentimiento"], "title": "A", "type": "object"}
```

*(En la comprobación la clase se llamó `A`; con `AnalisisTexto` el `title` sería ese nombre.)*

**Valores del sentimiento.** El tipo es `str`, no un `Enum`: la pista de valores está solo en la descripción. El archivo lo justifica como flexibilidad (admitir "Positivo" o "positivo") y sugiere, sin aplicarla, usar `Literal["Positivo", "neutro", "negativo"]` para forzar valores exactos. 📘

---

### Lección 4 (parte 2): modelo y `with_structured_output`

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

structured_llm = llm.with_structured_output(AnalisisTexto)
```

**El modelo.** `model="gpt-4o-mini"` es el identificador del modelo (uno inexistente daría un error 404, según el archivo 📘). `temperature=0.6`: para extracción suele preferirse `0`, pero el archivo considera aceptable `0.6` porque el esquema "ancla" la salida. 📘 ⚠️ *(no contrastado)*

**`with_structured_output`.** Envuelve el modelo: en lugar de texto libre, `structured_llm` devuelve un objeto tipado. Qué ocurre por dentro según el archivo 📘:

1. LangChain convierte `AnalisisTexto` en un JSON Schema.
2. Lo envía al modelo como la definición de una "herramienta" (*function/tool*).
3. Pide al modelo que llame a esa herramienta con los argumentos correctos.
4. LangChain recoge la respuesta, la valida con Pydantic y devuelve una **instancia de `AnalisisTexto`**.

**Contraste con la documentación oficial:**

| Afirmación del archivo | Documentación oficial |
|---|---|
| `with_structured_output(Esquema)` devuelve una instancia del esquema | Confirmado: con un modelo Pydantic devuelve una instancia (ejemplo `Movie(...)`) |
| Se basa en una llamada a herramienta | Confirmado: el método `'function_calling'` "deriva la salida estructurada forzando una llamada a herramienta que sigue el esquema". La página de `ChatOpenAI` indica que, si no se especifica otro, el método por defecto usa *function calling* |
| Pydantic valida el resultado | Confirmado: "Pydantic valida automáticamente" |
| La interfaz sigue siendo `.invoke`, `.stream`, `.batch` | ⚠️ No contrastado |

**Otras opciones que muestra la documentación oficial** y que el archivo no usa:

| Opción | Descripción |
|---|---|
| `method="json_schema"` | Usa la función nativa de salida estructurada del proveedor (en `ChatOpenAI`) |
| `method="json_mode"` | "Genera JSON válido, pero el esquema debe describirse en el prompt" |
| `include_raw=True` | Devuelve también el `AIMessage` sin procesar junto al resultado |
| Otros tipos de esquema | `TypedDict` o JSON Schema; con ellos el resultado es un `dict` y la validación es manual |

---

### Lección 5 (parte 2): ejecución y resultado

```python
texto_prueba = "Me encantó la nueva película de acción, tiene muchos efectos especiales y emoción."

resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

print(resultado.model_dump_json())
```

| Elemento | Qué hace |
|---|---|
| `texto_prueba` | Texto a analizar (en un caso real vendría de un usuario, un fichero o una base de datos) |
| `structured_llm.invoke(f"...")` | Envía el prompt y devuelve **un objeto `AnalisisTexto`**, no un `str` |
| `resultado.model_dump_json()` | Serializa el objeto a un `str` JSON (útil para logs, HTTP o guardar en un fichero) |

**Entrada de `.invoke`.** El archivo dice que se puede pasar un `str` o una lista de mensajes (`[HumanMessage(...)]`) y que LangChain lo normaliza. 📘 La documentación oficial indica que `invoke()` acepta un mensaje o una lista de mensajes y que pasar una cadena es un atajo de un único `HumanMessage`.

**Formas de usar el resultado** 📘:

| Expresión | Qué da |
|---|---|
| `resultado.resumen` | El campo `resumen`, ya tipado |
| `resultado.sentimiento` | El campo `sentimiento` |
| `resultado.model_dump()` | Un `dict` de Python |
| `resultado.model_dump_json()` | Un `str` JSON |

**Forma de la salida** (los valores los genera el modelo y varían; esta parte no se ejecutó):

```text
{"resumen":"<resumen generado por el modelo>","sentimiento":"<valor generado por el modelo>"}
```

**Puntos clave de las lecciones 2–5.**

- El esquema Pydantic es el **contrato** de salida: campos, tipos y descripciones.
- `with_structured_output` cambia el tipo de retorno de un `AIMessage` a un objeto Pydantic.
- `resultado` **no es un texto**: es una instancia del esquema.

### Ejercicios

Estos archivos no incluyen ejercicios propuestos.

---

## Guía práctica

### ¿Cómo valido y convierto datos con Pydantic?

```python
class Usuario(BaseModel):
    id: int
    nombre: str
    activo: bool = True

usuario = Usuario(**data)
```

### ¿Cómo hago opcional un campo?

Dale un valor por defecto: `activo: bool = True`.

### ¿Cómo convierto el objeto a JSON o a diccionario?

```python
usuario.model_dump_json()   # str JSON
usuario.model_dump()        # dict
```

### ¿Cómo describo un campo para que el modelo lo entienda?

```python
resumen: str = Field(description="Resumen breve del texto.")
```

### ¿Cómo pido al modelo una respuesta con forma fija?

```python
structured_llm = llm.with_structured_output(AnalisisTexto)
resultado = structured_llm.invoke("Analiza el siguiente texto: ...")
```

### ¿Cómo fuerzo valores exactos en un campo?

El archivo sugiere `Literal["Positivo", "neutro", "negativo"]` en lugar de `str`. 📘 No se aplica en el archivo.

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ValidationError` al crear `Usuario(...)` | Un valor no se puede convertir al tipo, o falta un campo obligatorio | Revisar los datos de entrada y los tipos del esquema ✅ |
| Error de autenticación en la parte 2 | `load_dotenv()` no se ejecutó antes de crear `ChatOpenAI`, o falta la clave | Llamar a `load_dotenv()` primero y definir `OPENAI_API_KEY` 📘 |
| Error 404 al llamar al modelo | Identificador de modelo inexistente | Revisar `model="..."` 📘 |
| El campo `sentimiento` devuelve "positivo" o "Positivo" indistintamente | El tipo es `str`, no un enumerado | Usar `Literal[...]` si necesitas valores exactos 📘 |
| `resultado` no tiene `.content` | Es un objeto del esquema, no un `AIMessage` | Usar `resultado.resumen`, `resultado.sentimiento`, etc. |

---

## Referencia

### Importaciones

| Importación | Paquete | Parte | Uso |
|---|---|---|---|
| `from pydantic import BaseModel` | `pydantic` | 1 y 2 | Clase base de los esquemas |
| `from pydantic import Field` | `pydantic` | 2 | Metadatos de un campo |
| `import os` | Biblioteca estándar | 2 | Sin uso directo en el script |
| `from dotenv import load_dotenv` | `python-dotenv` | 2 | Cargar `.env` |
| `from langchain_openai import ChatOpenAI` | `langchain-openai` | 2 | Modelo de chat |

### Esquemas del archivo

| Clase | Campos | Obligatorio | Valor por defecto |
|---|---|---|---|
| `Usuario` | `id: int` | Sí | — |
| | `nombre: str` | Sí | — |
| | `activo: bool` | No | `True` |
| `AnalisisTexto` | `resumen: str` | Sí | — |
| | `sentimiento: str` | Sí | — |

### Métodos de un modelo Pydantic v2 usados

| Método | Devuelve | Notas |
|---|---|---|
| `Clase(**datos)` | Instancia validada | Convierte tipos cuando es posible ✅ |
| `.model_dump_json()` | `str` JSON | El archivo indica que en Pydantic v1 el equivalente era `.json()`, obsoleto 📘 ⚠️ |
| `.model_dump()` | `dict` | ✅ |

### `with_structured_output`

| Elemento | Detalle | Fuente |
|---|---|---|
| Llamada | `modelo.with_structured_output(Esquema)` | Archivo y documentación oficial |
| Retorno con Pydantic | Instancia del esquema | Documentación oficial |
| Retorno con `TypedDict` / JSON Schema | `dict` | Documentación oficial |
| Validación | Automática con Pydantic; manual con `TypedDict` y JSON Schema | Documentación oficial |
| Parámetro `method` | `'json_schema'`, `'function_calling'`, `'json_mode'` | Documentación oficial |
| Parámetro `include_raw` | `True` devuelve también el `AIMessage` | Documentación oficial |
| Método por defecto en `ChatOpenAI` | *Function calling* | Documentación oficial (página de `ChatOpenAI`) |

### Parámetros del modelo

| Parámetro | Valor | Notas |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Parámetro usado en la documentación oficial |
| `temperature` | `0.6` | El archivo señala que para extracción suele preferirse `0` 📘 |

### Variables de entorno

| Variable | Clase | Confirmada en documentación oficial |
|---|---|---|
| `OPENAI_API_KEY` | `ChatOpenAI` | Sí |

### Tipo de retorno: comparación

| Llamada | Tipo devuelto |
|---|---|
| `llm.invoke(...)` | `AIMessage` |
| `llm.with_structured_output(AnalisisTexto).invoke(...)` | Instancia de `AnalisisTexto` |

---

## Explicación

### ¿Por qué usar un esquema en vez de pedir "responde en JSON"?

El esquema da un **contrato** con campos, tipos y descripciones, y devuelve un objeto Python validado en lugar de un texto que habría que interpretar. 📘 La documentación oficial indica que, con Pydantic, la validación es automática, mientras que con `TypedDict` o JSON Schema es manual.

### ¿Por qué es útil la parte 1 antes de la parte 2?

Porque la parte 2 usa Pydantic como base: entender que `BaseModel` valida, convierte y asigna valores por defecto explica por qué el resultado de `with_structured_output` es fiable y tipado.

### ¿Por qué la `description` "es prompt"?

Porque, según el archivo, se incluye en el esquema que recibe el modelo, que la lee como instrucción. Que Pydantic la incluya en el JSON Schema se comprobó al ejecutar ✅; que el modelo la use no se pudo contrastar con la documentación oficial. ⚠️

### ¿Por qué `sentimiento` es `str` y no un enumerado?

El archivo lo prefiere por flexibilidad ante variaciones de mayúsculas, a costa de no forzar los valores. Con `Literal[...]` se obtendría un control exacto. 📘

### ¿Cuándo usar cada enfoque?

| Situación | Enfoque |
|---|---|
| Validar datos que ya tienes (sin modelo) | Esquema Pydantic (parte 1) |
| Necesitas que el modelo devuelva campos concretos | `with_structured_output(Esquema)` |
| Solo necesitas texto libre | `llm.invoke(...)` |
| Necesitas el `AIMessage` además del objeto | `include_raw=True` (documentación oficial) |

### Lo que este documento no cubre

- Clases de *output parser* de LangChain (por ejemplo, los *parsers* de texto o JSON): no aparecen en los archivos pese al nombre.
- La **salida real** de la parte 2: requiere una clave de API y no se ejecutó.
- `Literal`, `Enum`, esquemas anidados y validadores personalizados de Pydantic (solo se cita `Literal` como sugerencia).
- Las opciones `method` e `include_raw`: se citan de la documentación oficial, pero los archivos no las usan.

### Observaciones sobre los archivos

- **Nombre vs contenido.** Los archivos se llaman *output parsers*, pero tratan de Pydantic y de salida estructurada.
- La descripción del campo `sentimiento` escribe "Positivo" con mayúscula y "neutro" y "negativo" en minúscula; el archivo reconoce que no fuerza valores exactos.
- El comentario sobre la temperatura es ambivalente: dice que para extracción suele preferirse `0` y a la vez que `0.6` es aceptable. No se pudo contrastar con las fuentes permitidas.
- `os` se importa pero no se usa directamente en la parte 2.

---

## Autoevaluación

1. ¿Qué hace `class Usuario(BaseModel)` además de definir campos?
2. ¿Qué ocurre con `id: int` cuando llega `"123"`?
3. ¿Por qué `activo` es opcional y `id` no?
4. ¿Qué imprime la parte 1?
5. ¿Qué diferencia hay entre `model_dump()` y `model_dump_json()`?
6. ¿Para qué sirve `Field(description=...)` según el archivo?
7. ¿Qué devuelve `llm.with_structured_output(AnalisisTexto).invoke(...)`?
8. ¿Qué cuatro pasos describe el archivo que ocurren "por dentro"?
9. ¿Qué ventaja e inconveniente tiene que `sentimiento` sea `str`?
10. ¿Qué parte necesita clave de API y cuál no?

### Respuestas

1. La convierte en un validador de datos: fuerza que los datos cumplan las reglas.
2. Pydantic lo convierte al entero `123`.
3. Porque `activo` tiene valor por defecto (`= True`); `id` no.
4. `{"id":123,"nombre":"Ana","activo":true}`.
5. `model_dump()` devuelve un `dict`; `model_dump_json()` devuelve un `str` JSON.
6. Añade una descripción que viaja en el JSON Schema y que el modelo lee como instrucción.
7. Una instancia de `AnalisisTexto`, no un `AIMessage`.
8. Convertir el esquema a JSON Schema; enviarlo como herramienta; pedir al modelo que la llame; parsear y validar la respuesta con Pydantic.
9. Ventaja: flexibilidad ante variaciones de mayúsculas. Inconveniente: no fuerza los valores exactos.
10. La parte 2 necesita clave de API; la parte 1 no.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `BaseModel` | Esquema que valida y convierte datos |
| Campo con valor por defecto | Opcional |
| `ValidationError` | Datos que no cumplen el esquema |
| `model_dump_json()` / `model_dump()` | `str` JSON / `dict` |
| `Field(description=...)` | Describe el campo para el modelo |
| `with_structured_output(Esquema)` | El modelo devuelve un objeto tipado |
| Retorno con Pydantic | Instancia del esquema, no `AIMessage` |
| `Literal[...]` | Forma de forzar valores exactos (sugerencia del archivo) |

---

## Fuentes consultadas

- Documentación oficial de modelos (sección de salida estructurada): <https://docs.langchain.com/oss/python/langchain/models>
- Documentación oficial de `ChatOpenAI` (salida estructurada, método por defecto, `method="json_schema"`): <https://python.langchain.com/docs/integrations/chat/openai/> (redirige a <https://docs.langchain.com/oss/python/integrations/chat/openai>)
- Documentación oficial de mensajes: <https://docs.langchain.com/oss/python/langchain/messages>
- Material del tema: `2.8_output_parsers_parte1.py` y `2.9_output_parsers_parte2.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
