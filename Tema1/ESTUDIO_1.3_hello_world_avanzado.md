# 1.3 Plantillas y cadenas: `PromptTemplate` + operador `|` (LCEL)

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `hello_world_avanzado.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice; ✅ *(ejecutado)* = comportamiento observado al ejecutar código con `langchain-core` 1.7.7. El script necesita una clave de API, por lo que **no se ejecutó con un modelo real**: el mecanismo (plantilla, cadena, `AIMessage`) se comprobó con un modelo simulado (`FakeListChatModel`) y la respuesta de un modelo real nunca se reproduce.

---

## Antes de empezar

### Qué vas a aprender

1. Crear una plantilla de texto con huecos (`PromptTemplate`).
2. Unir plantilla y modelo en una cadena con el operador `|`.
3. Ejecutar la cadena con `.invoke()` pasando un diccionario.

### Qué necesitas

- Documentos 1.1 y 1.2 (modelo, `invoke`, `AIMessage`).
- Clave `DEEPSEEK_API_KEY` en el `.env`.

### Mapa del tema

```text
{"nombre": "Juan"} ──► PromptTemplate ──► texto con el hueco relleno ──► ChatDeepSeek ──► AIMessage
                        (plantilla)                                       (chat)           └─ .content
                        └──────────── cadena = plantilla | chat ────────────┘
```

### Glosario

| Término | Significado |
|---|---|
| **`PromptTemplate`** | Texto fijo con huecos `{variable}` que se rellenan después. |
| **`input_variables`** | Lista de los nombres de los huecos. |
| **LCEL** | LangChain Expression Language: forma de encadenar pasos con `\|`. |
| **Cadena** | Objeto resultante de unir pasos con `\|`; también se ejecuta con `.invoke()`. |
| **`LLMChain`** | Forma antigua de crear cadenas; el archivo dice que `\|` la sustituye. 📘 |

---

## Tutorial

### Preparación

```bash
pip install -qU langchain-deepseek langchain-core python-dotenv
python hello_world_avanzado.py
```

---

### Lección 1: importaciones y modelo (secciones 1 a 3)

```python
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate

load_dotenv()

chat = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
```

| Línea | Qué hace y por qué |
|---|---|
| `from langchain_core.prompts import PromptTemplate` | Importa la plantilla desde `langchain_core`. El archivo avisa de que la ruta antigua `langchain.prompts` está obsoleta y puede dar error de importación. 📘 |
| `load_dotenv()` | Carga la clave; debe ir antes de crear el modelo. |
| `ChatDeepSeek(...)` | Mismo modelo que en el documento 1.2; aquí la variable se llama `chat` en lugar de `llm`. |

---

### Lección 2: la plantilla (sección 4)

```python
plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre. \nNombre: {nombre}\nAsistente:"
)
```

| Parte | Significado |
|---|---|
| `input_variables=["nombre"]` | Declara qué huecos tiene el texto. |
| `template="..."` | Texto con el hueco `{nombre}`; `\n` son saltos de línea. |

**Comprobado** ✅ (sin modelo): con `nombre="Juan"` la plantilla produce exactamente este texto:

```text
Saluda al usuario con su nombre. 
Nombre: Juan
Asistente:
```

Y si falta la variable, `plantilla.format()` lanza `KeyError: 'nombre'`.

---

### Lección 3: la cadena con `|` (sección 5)

```python
cadena = plantilla | chat
```

El operador `|` significa "el resultado del paso de la izquierda pasa al de la derecha": primero se rellena la plantilla y su resultado va al modelo. 📘

**Comprobado** ✅ (modelo simulado): `plantilla | modelo` produce una cadena que, al invocarse, devuelve un `AIMessage`.

El archivo afirma que esto sustituye a la antigua `LLMChain`. 📘 ⚠️ (no se contrastó en las páginas oficiales consultadas).

---

### Lección 4: ejecutar (sección 6)

```python
resultado = cadena.invoke({"nombre": "Juan"})

print("→ Respuesta de DeepSeek:", resultado.content)
```

| Paso | Qué hace |
|---|---|
| `{"nombre": "Juan"}` | Diccionario: clave = hueco, valor = texto a poner. |
| `cadena.invoke(...)` | Rellena la plantilla y envía el texto al modelo. |
| `resultado.content` | El resultado es un `AIMessage`; `.content` es su texto. |

La respuesta no se reproduce: la genera el modelo y varía.

**Puntos clave.**

- Plantilla = estructura; diccionario = datos.
- `plantilla | chat` es la cadena; se usa igual que un modelo (`.invoke`).

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo uso otro nombre?

Cambia el valor del diccionario: `cadena.invoke({"nombre": "Ana"})`.

### ¿Cómo añado otra variable?

Añade el hueco al texto, inclúyelo en `input_variables` y pásalo en el diccionario.

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `KeyError: 'nombre'` | Falta esa clave en el diccionario | Pasar todas las variables ✅ |
| Error de importación de `PromptTemplate` | Ruta antigua `langchain.prompts` | Usar `langchain_core.prompts` 📘 |

---

## Referencia

| Elemento | Descripción |
|---|---|
| `PromptTemplate(input_variables, template)` | Crea la plantilla. |
| `plantilla \| chat` | Crea la cadena (LCEL). |
| `cadena.invoke(dict)` | Ejecuta la cadena con un diccionario; devuelve `AIMessage`. ✅ |
| `resultado.content` | Texto de la respuesta. |

---

## Explicación

### ¿Por qué usar una plantilla y no concatenar texto?

Separa la estructura fija del dato variable: el mismo molde se reutiliza con otros valores y no se manipula el texto a mano. 📘

### ¿Por qué `|`?

Es la forma de LCEL de componer pasos; cada paso es un *runnable* (se puede invocar). Permite añadir más pasos después, como se verá en el Tema 2.

### Lo que este documento no cubre

- Plantillas de varios mensajes (`ChatPromptTemplate`, Tema 2).
- La respuesta real del modelo.

### Observaciones sobre el archivo

- El texto de la plantilla termina en `Asistente:` y se envía como un único mensaje; el archivo no explica esta elección (en 1.5 se repite el patrón). ⚠️
- El script no termina el archivo con salto de línea; no afecta a la ejecución.

---

## Autoevaluación

1. ¿Qué es una `PromptTemplate`?
2. ¿Qué hace `|` en `plantilla | chat`?
3. ¿Qué tipo de dato se pasa a `cadena.invoke`?
4. ¿Qué devuelve la cadena?
5. ¿Qué error aparece si falta una variable?
6. ¿De qué módulo se importa `PromptTemplate` según el archivo?

### Respuestas

1. Un texto con huecos que se rellenan después.
2. Pasa el resultado de la plantilla al modelo.
3. Un diccionario.
4. Un `AIMessage`.
5. `KeyError` con el nombre de la variable que falta ✅.
6. `langchain_core.prompts`.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `PromptTemplate` | Molde de texto con huecos |
| `\|` (LCEL) | Une pasos: salida de uno = entrada del siguiente |
| `cadena.invoke({...})` | Ejecuta con un diccionario |
| `AIMessage.content` | Texto de la respuesta |

---

## Fuentes consultadas

- Documentación oficial de modelos: <https://docs.langchain.com/oss/python/langchain/models>
- Documentación oficial de `ChatDeepSeek`: <https://docs.langchain.com/oss/python/integrations/chat/deepseek>
- Material del tema: `hello_world_avanzado.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
