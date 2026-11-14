# 2.5 `MessagesPlaceholder`: insertar un historial de conversación en una plantilla

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.5_message_playholders.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice. ✅ *(ejecutado)* = comportamiento observado al ejecutar el código con `langchain-core` 1.7.7; este archivo no llama a ningún modelo, así que su salida es reproducible.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un **historial de conversación** como lista de mensajes.
2. Para qué sirve `MessagesPlaceholder` dentro de un `ChatPromptTemplate`.
3. Por qué importa el **orden** `system` → historial → `human`.
4. Cómo rellenar la plantilla con `format_messages` y qué lista de mensajes produce.

### Qué necesitas

- Haber visto `ChatPromptTemplate`, `from_messages` y `format_messages` (documento 2.3).
- Conocer `HumanMessage` y `AIMessage`.
- No necesitas clave de API: el archivo **no llama a ningún modelo**.

### Mapa del tema

```text
chat_prompt = [ ("system", ...)                          ← cómo debe comportarse el modelo
                MessagesPlaceholder("historial")          ← lista de turnos previos
                ("human", "Usuario: {pregunta_actual}") ]  ← pregunta nueva

format_messages(historial=[...4 mensajes...], pregunta_actual="...")
        └──►  [System, Human, AI, Human, AI, Human]   (6 mensajes)
```

### Glosario

| Término | Significado |
|---|---|
| **Historial** | Lista de mensajes (`HumanMessage` / `AIMessage`) con los turnos anteriores. |
| **`MessagesPlaceholder`** | Pieza especial de una plantilla: no es un texto, es un "hueco" que se rellena con **una lista entera de mensajes**. 📘 |
| **`variable_name`** | Nombre del hueco; debe coincidir con la clave que pasas al formatear. |
| **`format_messages(**kwargs)`** | Rellena la plantilla y devuelve la lista de mensajes. |
| **`BaseMessage`** | Clase base de los mensajes (`SystemMessage`, `HumanMessage`, `AIMessage`…). |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-core
```

Ejecución (no necesita clave de API):

```bash
python 2.5_message_playholders.py
```

---

### Lección 1: importaciones y plantilla (secciones 1 y 2 del archivo)

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])
```

| Elemento | Qué hace |
|---|---|
| `HumanMessage`, `AIMessage` | Para construir el historial simulado |
| `ChatPromptTemplate`, `MessagesPlaceholder` | La plantilla y el hueco para el historial |
| `("system", "...")` | Cómo debe comportarse el modelo |
| `MessagesPlaceholder(variable_name="historial")` | Hueco donde irá la lista de turnos previos |
| `("human", "Usuario: {pregunta_actual}")` | El nuevo mensaje, con la variable `{pregunta_actual}` |

**El orden es crítico y tiene lógica conversacional** 📘:

1. `system`: cómo debe comportarse el modelo.
2. `historial`: los turnos previos (el contexto).
3. `human`: el mensaje nuevo del usuario.

**Punto clave.** `MessagesPlaceholder` no es texto: le dice a la plantilla "aquí va una lista entera de mensajes". 📘

---

### Lección 2: el historial simulado (sección 3 del archivo)

```python
historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.")
]
```

- Es una **lista de objetos**, no un texto. Cada objeto tiene un **tipo** (`HumanMessage` o `AIMessage`) que indica quién habló. Es lo que `MessagesPlaceholder` espera recibir. 📘
- Los prefijos `"Usuario:"` e `"IA:"` **no son roles mágicos**: son texto que el ejemplo añade para que se entienda mejor. En la práctica el modelo no necesita que los mensajes empiecen así. 📘 *(el archivo lo explica en un comentario)*

---

### Lección 3: formatear y ver el resultado (secciones 4 y 5 del archivo)

```python
mensajes = chat_prompt.format_messages(
    historial=historial_conversacion,
    pregunta_actual="¿Puedes decirme algo interesante de su arquitectura?"
)

for m in mensajes:
    print(m.content)
```

**Claves que se pasan a `format_messages`:**

| Clave | Debe coincidir con |
|---|---|
| `historial` | El `variable_name` del `MessagesPlaceholder` |
| `pregunta_actual` | El hueco `{pregunta_actual}` del mensaje `human` |

**Resultado.** `mensajes` es una lista de **6** objetos `BaseMessage`: 1 `SystemMessage` + 4 del historial + 1 `HumanMessage` (la pregunta nueva). 📘 ✅

```text
SystemMessage  → Eres un asistente útil que mantiene el contexto de la conversación.
HumanMessage   → Usuario: ¿Cuál es la capital de Francia?
AIMessage      → IA: La capital de Francia es París.
HumanMessage   → Usuario: ¿Y cuántos habitantes tiene?
AIMessage      → IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.
HumanMessage   → Usuario: ¿Puedes decirme algo interesante de su arquitectura?
```

**Salida real del script** (solo imprime `m.content`, es decir, el texto de cada mensaje) ✅:

```text
Eres un asistente útil que mantiene el contexto de la conversación.
Usuario: ¿Cuál es la capital de Francia?
IA: La capital de Francia es París.
Usuario: ¿Y cuántos habitantes tiene?
IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.
Usuario: ¿Puedes decirme algo interesante de su arquitectura?
```

Los tipos de la columna izquierda del primer bloque no los imprime el script: el archivo indica que `print(type(m))` mostraría `SystemMessage`, `HumanMessage` y `AIMessage`. 📘

**Punto clave.** El modelo recibirá la conversación estructurada: un mensaje de sistema, los turnos previos con su rol y la pregunta nueva.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo añado un historial a una plantilla de chat?

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])
```

### ¿Cómo construyo el historial?

Con una lista de `HumanMessage` y `AIMessage`, alternando según quién habló.

### ¿Cómo relleno la plantilla?

```python
mensajes = chat_prompt.format_messages(
    historial=historial_conversacion,
    pregunta_actual="¿Puedes decirme algo interesante de su arquitectura?"
)
```

### ¿Cómo veo qué se enviaría al modelo?

Recorre la lista e imprime `m.content` (o `type(m)` para ver el tipo de cada mensaje).

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `KeyError: 'historial'` | No se pasó el historial al formatear | Pasar `historial=[...]`, o declarar el hueco como opcional (ver Explicación) ✅ |
| `KeyError` con otro nombre | La clave no coincide con la variable de la plantilla | Usar exactamente `historial` y `pregunta_actual` 📘 |
| Los mensajes del modelo repiten "Usuario:" / "IA:" | Los prefijos forman parte del contenido | No son necesarios; el rol lo da el tipo de mensaje 📘 |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `from langchain_core.messages import HumanMessage, AIMessage` | `langchain-core` | Mensajes del historial |
| `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder` | `langchain-core` | Plantilla y hueco |

### `MessagesPlaceholder`: lo que muestra la referencia oficial

| Parámetro | Descripción oficial |
|---|---|
| `variable_name` | Nombre de la clave que aporta los mensajes |
| `optional` | Si la clave puede omitirse. El ejemplo oficial usa `optional=True` |
| `n_messages` | Atributo que aparece en un ejemplo oficial (`MessagesPlaceholder("history", n_messages=1)`); la página no lo describe en prosa |

Comportamientos que muestran los ejemplos oficiales:

| Caso | Resultado |
|---|---|
| Falta la clave y **no** es `optional` | `KeyError` (confirmado al ejecutar ✅) |
| Falta la clave y `optional=True` | Lista vacía |
| Se pasan tuplas `(rol, contenido)` | Se convierten en objetos de mensaje |

La referencia oficial incluye este ejemplo, equivalente al del archivo:

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
prompt.invoke(
    {
        "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
        "question": "now multiply that by 4",
    }
)
```

### Variables de la plantilla del archivo

| Variable | Origen |
|---|---|
| `historial` | `MessagesPlaceholder(variable_name="historial")` |
| `pregunta_actual` | Hueco en el mensaje `human` |

Al ejecutar, `chat_prompt.input_variables` devuelve ambas ✅.

### Métodos usados

| Método | Devuelve |
|---|---|
| `chat_prompt.format_messages(**kwargs)` | Lista de mensajes (`list[BaseMessage]`) |
| `chat_prompt.invoke({...})` | `ChatPromptValue` (confirmado en la referencia oficial y al ejecutar ✅); no se usa en el archivo |

### Composición del resultado del archivo

| Posición | Tipo | Origen |
|---|---|---|
| 1 | `SystemMessage` | `("system", ...)` |
| 2–5 | `HumanMessage`, `AIMessage`, `HumanMessage`, `AIMessage` | `historial_conversacion` |
| 6 | `HumanMessage` | `("human", "Usuario: {pregunta_actual}")` |

---

## Explicación

### ¿Por qué un historial como lista de mensajes y no como texto?

Porque cada objeto lleva su **tipo** (`HumanMessage` / `AIMessage`), que indica quién habló, y eso es lo que `MessagesPlaceholder` espera. 📘 La documentación oficial usa el mismo patrón: el historial se pasa como lista de mensajes bajo el nombre del hueco.

### ¿Por qué importa el orden?

Porque la conversación tiene lógica: primero las instrucciones, después el contexto previo y al final lo que acaba de decir el usuario. 📘

### ¿Qué pasa si no hay historial?

Sin `optional=True`, formatear sin la clave del historial lanza `KeyError`; con `optional=True`, el hueco se rellena con una lista vacía. Esto procede de los ejemplos de la referencia oficial y se confirmó al ejecutar la primera variante ✅. El archivo **no** usa `optional`, por lo que siempre hay que pasar el historial, aunque sea una lista vacía.

### ¿Cuándo usar este enfoque?

| Situación | Enfoque |
|---|---|
| Chatbot con memoria de turnos previos | `MessagesPlaceholder` |
| Una sola pregunta sin contexto | Un `ChatPromptTemplate` con `system` y `human` |
| Quieres ver el prompt sin llamar al modelo | `format_messages(...)` |

### Lo que este documento no cubre

- La llamada a un modelo: el archivo no la hace.
- Cómo se obtiene y se actualiza el historial en una aplicación real.
- `n_messages` (solo aparece en un ejemplo oficial sin descripción).

### Observaciones sobre el archivo

- Los prefijos `"Usuario:"` e `"IA:"` en el contenido de los mensajes y en la plantilla son redundantes: el rol ya lo indica el tipo de mensaje. El propio archivo lo aclara.
- El nombre del archivo escribe `playholders` (no `placeholders`). Es solo el nombre del archivo.

---

## Autoevaluación

1. ¿Qué es `MessagesPlaceholder` y en qué se diferencia de un mensaje `("human", "...")`?
2. ¿Por qué el historial es una lista de objetos y no un texto?
3. ¿Qué dos claves se pasan a `format_messages` en este archivo y con qué deben coincidir?
4. ¿Cuántos mensajes tiene `mensajes` y de qué tipo?
5. ¿Por qué el orden `system` → historial → `human` es importante?
6. ¿Son `"Usuario:"` e `"IA:"` roles de LangChain?
7. ¿Qué ocurre si no se pasa `historial` al formatear?
8. ¿Qué imprime el bucle final?

### Respuestas

1. Un hueco que se rellena con una lista entera de mensajes; un mensaje `human` es un único texto con rol.
2. Porque cada objeto indica con su tipo quién habló; es lo que espera el `MessagesPlaceholder`.
3. `historial` (el `variable_name` del hueco) y `pregunta_actual` (el hueco del mensaje `human`).
4. Seis: un `SystemMessage`, cuatro del historial y un `HumanMessage`.
5. Porque refleja la lógica conversacional: instrucciones, contexto previo y pregunta nueva.
6. No: son texto del ejemplo; el rol lo da el tipo de mensaje.
7. Se lanza `KeyError`, salvo que el hueco se declare opcional.
8. El texto (`m.content`) de cada uno de los 6 mensajes, uno por línea.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `MessagesPlaceholder` | Hueco para una lista de mensajes |
| `variable_name` | Nombre del hueco; coincide con la clave al formatear |
| Orden | `system` → historial → `human` |
| `format_messages` | Devuelve la lista final de mensajes |
| Historial | Lista de `HumanMessage` / `AIMessage` |
| `optional=True` | Permite omitir el historial (según la referencia oficial) |

---

## Fuentes consultadas

- Referencia de `MessagesPlaceholder`: <https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder>
- Referencia de `ChatPromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate>
- Documentación oficial de mensajes: <https://docs.langchain.com/oss/python/langchain/messages>
- Material del tema: `2.5_message_playholders.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
