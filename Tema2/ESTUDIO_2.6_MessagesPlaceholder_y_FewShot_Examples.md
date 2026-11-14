# 2.6 `MessagesPlaceholder` y ejemplos *few-shot*

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.6_MessagesPlaceholder_y_FewShot_Examples.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice. ✅ *(ejecutado)* = comportamiento observado al ejecutar el código con `langchain-core` 1.7.7; este archivo no llama a ningún modelo, así que su salida es reproducible.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es el **aprendizaje en contexto** (*In-Context Learning*) y qué son los ejemplos ***few-shot***.
2. Cómo usar `MessagesPlaceholder` para inyectar **ejemplos** en un prompt (y no solo un historial).
3. Cómo estructurar un prompt de clasificación de sentimientos con ejemplos `Human` / `AI`.
4. Qué buenas prácticas seguir al elegir los ejemplos.

### Qué necesitas

- Haber visto `ChatPromptTemplate`, `MessagesPlaceholder` y `format_messages` (documentos 2.3 y 2.5).
- No necesitas clave de API: el archivo **no llama a ningún modelo**; solo construye y muestra el prompt.

### Mapa del tema

```text
El MISMO mecanismo (MessagesPlaceholder) sirve para dos propósitos:

  memoria   →  inyectar una conversación pasada       (historial)
  ejemplos  →  inyectar un "entrenamiento" con ejemplos (few-shot)   ← lo nuevo

Estructura de un prompt few-shot:
  system        instrucciones generales
  ejemplo 1     Human (entrada)  →  AI (salida esperada)
  ejemplo 2     Human (entrada)  →  AI (salida esperada)
  ...
  human         la pregunta real; el modelo debe producir la salida
```

### Glosario

| Término | Significado |
|---|---|
| **In-Context Learning (ICL)** | Enseñar al modelo mediante ejemplos incluidos en el propio prompt. 📘 |
| **Few-shot examples** | De 2 a 5 ejemplos (entrada → salida) que guían al modelo. 📘 |
| **`MessagesPlaceholder`** | Hueco de la plantilla que se rellena con una lista de mensajes; aquí aloja los ejemplos. |
| **Par Human/AI** | Un ejemplo: un `HumanMessage` con la entrada y un `AIMessage` con la salida esperada. |
| **Clasificación de sentimientos** | Tarea del ejemplo: etiquetar un texto como `POSITIVO`, `NEGATIVO` o `NEUTRO`. |
| **NER** | *Named Entity Recognition*: extraer entidades (nombres, empresas…) de un texto. Se cita en el Ejemplo A. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-core
```

Ejecución (no necesita clave de API):

```bash
python 2.6_MessagesPlaceholder_y_FewShot_Examples.py
```

---

### Lección 1: importaciones (sección 1 del archivo)

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

`HumanMessage` y `AIMessage` sirven para construir tanto el historial como los ejemplos; `ChatPromptTemplate` y `MessagesPlaceholder` forman la plantilla.

---

### Lección 2: código base, el historial (sección 2 del archivo)

El archivo recupera el patrón de la lección anterior (documento 2.5) como referencia, para comparar cómo **el mismo concepto** sirve a dos propósitos: memoria y ejemplos. 📘

```python
chat_prompt_historial = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])

historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.")
]

mensajes_historial = chat_prompt_historial.format_messages(
    historial=historial_conversacion,
    pregunta_actual="¿Puedes decirme algo interesante de su arquitectura?"
)
```

Aquí se inyecta una "conversación pasada": el modelo usa esa memoria para entender que "su arquitectura" se refiere a París. 📘

> **Importante.** En esta sección el script solo imprime un encabezado; **no imprime** `mensajes_historial` ✅. Para ver su contenido, recorre la lista con un bucle (el resultado equivale al del documento 2.5).

---

### Lección 3: ejemplos *few-shot* (sección 3 del archivo)

**Objetivo.** Clasificar el sentimiento de un texto usando ejemplos en lugar de explicarlo con palabras.

#### 3.1 La plantilla

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto a analizar: {texto_usuario}")
])
```

| Pieza | Función |
|---|---|
| `("system", ...)` | Instrucción general: clasificar como `POSITIVO`, `NEGATIVO` o `NEUTRO` |
| `MessagesPlaceholder(variable_name="ejemplos")` | Hueco para los ejemplos (la **estructura** es idéntica a la del historial) |
| `("human", "Texto a analizar: {texto_usuario}")` | La pregunta real |

#### 3.2 Los ejemplos

```python
ejemplos_sentimientos = [
    HumanMessage(content="Texto a analizar: Me encanta este producto, es increíble"),
    AIMessage(content="POSITIVO"),
    HumanMessage(content="Texto a analizar: El servicio fue terrible, muy decepcionante"),
    AIMessage(content="NEGATIVO"),
    HumanMessage(content="Texto a analizar: El clima está nublado hoy"),
    AIMessage(content="NEUTRO")
]
```

Cada ejemplo es un par: lo que diría un humano (`Texto a analizar: X`) y lo que debe responder la IA (una única palabra en mayúsculas). El modelo "ve" la estructura exacta de una conversación, que es un molde que imitar. 📘

**Dos detalles de diseño que se pueden comprobar en el código:**

- Los ejemplos llevan el mismo prefijo `Texto a analizar:` que la plantilla `human`, de modo que la pregunta real tiene **el mismo formato** que los ejemplos.
- Hay un ejemplo de cada categoría: positivo, negativo y neutro.

#### 3.3 Formatear y ver el resultado

```python
mensajes = chat_prompt.format_messages(
    ejemplos=ejemplos_sentimientos,
    texto_usuario="¡Qué día tan maravilloso!"
)

for i, m in enumerate(mensajes):
    print(f"Mensaje {i+1} ({m.__class__.__name__}):")
    print(m.content)
    print("-" * 40)
```

**Claves de `format_messages`:**

| Clave | Debe coincidir con |
|---|---|
| `ejemplos` | El `variable_name` del `MessagesPlaceholder` |
| `texto_usuario` | El hueco `{texto_usuario}` del mensaje `human` |

**Salida real de esta sección** ✅:

```text
Mensaje 1 (SystemMessage):
Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO.
----------------------------------------
Mensaje 2 (HumanMessage):
Texto a analizar: Me encanta este producto, es increíble
----------------------------------------
Mensaje 3 (AIMessage):
POSITIVO
----------------------------------------
Mensaje 4 (HumanMessage):
Texto a analizar: El servicio fue terrible, muy decepcionante
----------------------------------------
Mensaje 5 (AIMessage):
NEGATIVO
----------------------------------------
Mensaje 6 (HumanMessage):
Texto a analizar: El clima está nublado hoy
----------------------------------------
Mensaje 7 (AIMessage):
NEUTRO
----------------------------------------
Mensaje 8 (HumanMessage):
Texto a analizar: ¡Qué día tan maravilloso!
----------------------------------------
```

La lista tiene **8** mensajes: 1 de sistema + 6 de ejemplos + 1 pregunta real. El archivo lo resume así: la lista es "una mezcla de instrucciones, ejemplos y la pregunta real, todo perfectamente estructurado en roles". 📘

**Punto clave.** Un historial y unos ejemplos tienen la **misma forma** (una lista de `HumanMessage` / `AIMessage`); lo que cambia es el **propósito**: recordar una conversación frente a enseñar un patrón.

---

### Lección 4: otros casos de uso (sección 4 del archivo)

El archivo define dos listas de ejemplos más y solo imprime una línea descriptiva de cada una:

```python
ejemplos_extraccion = [
    HumanMessage(content="Texto: Juan Pérez trabaja en Google como ingeniero desde 2020"),
    AIMessage(content="Nombre: Juan Pérez, Empresa: Google, Puesto: ingeniero, Año: 2020"),
    HumanMessage(content="Texto: María Silva es doctora en el Hospital Central"),
    AIMessage(content="Nombre: María Silva, Empresa: Hospital Central, Puesto: doctora, Año: N/A")
]
print("Ejemplo A: Extracción de datos estructurados de texto libre.")

ejemplos_traduccion = [
    HumanMessage(content="Formal en inglés: Good morning, how are you today?"),
    AIMessage(content="Casual en español: ¡Hola! ¿Qué tal?"),
    HumanMessage(content="Formal en inglés: I would like to schedule a meeting"),
    AIMessage(content="Casual en español: ¿Podemos quedar?")
]
print("Ejemplo B: Traducción cambiando el tono (de formal a casual).")
```

| Ejemplo | Tarea | Qué enseñan los pares |
|---|---|---|
| A | Extracción de información (NER) | A convertir un texto libre en `Nombre`, `Empresa`, `Puesto`, `Año` (con `N/A` si no hay dato) |
| B | Traducción con estilo | A pasar de inglés formal a español casual |

> **Importante.** Estas dos listas **no se usan** con ninguna plantilla: solo se definen. Para aprovecharlas habría que montar un `ChatPromptTemplate` con su `MessagesPlaceholder`, como en la Lección 3 ✅.

Salida real de la sección 4 ✅:

```text
Ejemplo A: Extracción de datos estructurados de texto libre.
Ejemplo B: Traducción cambiando el tono (de formal a casual).
```

---

### Lección 5: ventajas y buenas prácticas (secciones 5 y 6 del archivo)

#### Ventajas de usar `MessagesPlaceholder` para los ejemplos 📘 ⚠️ *(no contrastadas)*

| Con `MessagesPlaceholder` | Sin él (texto plano concatenado) |
|---|---|
| Estructura clara: cada ejemplo mantiene su rol | Todo mezclado en un solo bloque de texto |
| Escalable: añadir o quitar ejemplos es editar una lista de Python | Cambiar los ejemplos exige reescribir el texto |
| Reutilizable: cambiar la lista de ejemplos crea una tarea nueva | El modelo puede confundirse y "alucinar" el formato |
| Natural: el formato conversacional es el de los modelos de chat actuales | Se pierde la separación explícita de turnos que espera la API de chat |

#### Mejores prácticas para ejemplos *few-shot* 📘

| # | Práctica | Detalle |
|---|---|---|
| 1 | **Cantidad (2–5)** | Basta para mostrar el patrón sin saturar la ventana de contexto ni disparar el coste en tokens |
| 2 | **Diversidad** | Cubre casos distintos (por ejemplo, un positivo muy claro, un negativo sutil y un neutro) |
| 3 | **Formato consistente** | El mismo patrón exacto en todos los ejemplos (si una salida no lleva punto final, ninguna debería llevarlo) |
| 4 | **Calidad** | Las salidas deben ser perfectas: el modelo tiende a copiar ese estilo y formato |

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo enseño un formato de salida al modelo con ejemplos?

Crea una lista de pares `HumanMessage` / `AIMessage` con la entrada y la salida deseada, y pásala a un `MessagesPlaceholder`:

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto a analizar: {texto_usuario}")
])
```

### ¿Cómo uso el prompt con ejemplos?

```python
mensajes = chat_prompt.format_messages(
    ejemplos=ejemplos_sentimientos,
    texto_usuario="¡Qué día tan maravilloso!"
)
```

### ¿Cómo cambio de tarea sin tocar la plantilla?

Cambia la lista de ejemplos (y la instrucción `system` si hace falta). 📘

### ¿Cómo elijo buenos ejemplos?

Sigue las cuatro prácticas de la Lección 5: 2–5 ejemplos, diversos, con formato consistente y salidas perfectas.

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `KeyError: 'ejemplos'` | No se pasó la clave `ejemplos` al formatear | Pasar `ejemplos=[...]` (la referencia oficial indica `KeyError` si falta la clave y no es `optional`) |
| El formato de la pregunta real difiere del de los ejemplos | Prefijos distintos | Usar el mismo prefijo (`Texto a analizar:`) en los ejemplos y en la plantilla 📘 |
| El modelo no sigue el formato esperado | Ejemplos inconsistentes o de baja calidad | Revisar las prácticas 3 y 4 📘 |
| Las listas A y B "no hacen nada" | Solo se definen | Montar una plantilla con su `MessagesPlaceholder` ✅ |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `from langchain_core.messages import HumanMessage, AIMessage` | `langchain-core` | Pares de ejemplo e historial |
| `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder` | `langchain-core` | Plantilla y hueco |

### Dos usos de `MessagesPlaceholder` en el archivo

| | Historial (sección 2) | Few-shot (sección 3) |
|---|---|---|
| Plantilla | `chat_prompt_historial` | `chat_prompt` |
| `variable_name` | `historial` | `ejemplos` |
| Variable del mensaje `human` | `{pregunta_actual}` | `{texto_usuario}` |
| Contenido inyectado | Conversación pasada | Pares entrada → salida |
| Propósito | Memoria | Enseñar un patrón |

### Mensajes de la plantilla few-shot

| Posición | Tipo | Origen |
|---|---|---|
| 1 | `SystemMessage` | `("system", ...)` |
| 2–7 | `HumanMessage` / `AIMessage` alternados | `ejemplos_sentimientos` |
| 8 | `HumanMessage` | `("human", "Texto a analizar: {texto_usuario}")` |

### Listas de ejemplos del archivo

| Lista | Pares | Etiquetas / formato |
|---|---|---|
| `ejemplos_sentimientos` | 3 | `POSITIVO`, `NEGATIVO`, `NEUTRO` |
| `ejemplos_extraccion` | 2 | `Nombre: ..., Empresa: ..., Puesto: ..., Año: ...` |
| `ejemplos_traduccion` | 2 | `Casual en español: ...` |

### `MessagesPlaceholder`: lo que muestra la referencia oficial

| Parámetro | Descripción |
|---|---|
| `variable_name` | Nombre de la clave que aporta los mensajes |
| `optional` | Si la clave puede omitirse (sin él, omitirla lanza `KeyError`) |
| `n_messages` | Aparece en un ejemplo oficial; sin descripción en prosa |

### Métodos usados

| Método | Devuelve |
|---|---|
| `ChatPromptTemplate.from_messages([...])` | Plantilla de chat |
| `plantilla.format_messages(**kwargs)` | Lista de mensajes (`list[BaseMessage]`) |

### Qué muestra la documentación oficial sobre *few-shot*

| Elemento | Información |
|---|---|
| `FewShotChatMessagePromptTemplate` | Existe en `langchain_core.prompts.few_shot`; la referencia solo trae firma y bases, sin descripción |
| Guía "few-shot en modelos de chat" | La página antigua redirige a la portada de la documentación actual |

---

## Explicación

### ¿Por qué se llama *In-Context Learning*?

Porque el modelo "aprende" la tarea **dentro del propio prompt**, a partir de los ejemplos, sin entrenamiento adicional. 📘 ⚠️ *(no contrastado: las páginas oficiales consultadas no describen el concepto)*

### ¿Por qué `MessagesPlaceholder` sirve para memoria **y** para ejemplos?

Porque en ambos casos se inserta una **lista de mensajes** `Human` / `AI` en el mismo lugar de la plantilla. La diferencia no está en el mecanismo sino en el contenido y la intención. 📘

### ¿Por qué los ejemplos llevan el mismo formato que la pregunta real?

Porque el modelo debe reconocer el patrón: cuando un humano escribe `Texto a analizar: X`, la IA responde con una única palabra en mayúsculas. Cuanto más consistente sea el formato, más fácil es imitarlo. 📘

### ¿Cuántos ejemplos usar?

El archivo recomienda de 2 a 5: suficiente para mostrar el patrón sin saturar la ventana de contexto ni aumentar el coste en tokens. 📘 ⚠️ *(no contrastado)* El ejemplo del archivo usa 3.

### ¿Cuándo usar este enfoque?

| Situación | Enfoque |
|---|---|
| Quieres que el modelo recuerde una conversación | `MessagesPlaceholder` con historial |
| Quieres que imite un formato o criterio | `MessagesPlaceholder` con pares ejemplo |
| Una tarea que el modelo ya hace bien sin ejemplos | Solo `system` + `human` |

### Lo que este documento no cubre

- La **llamada a un modelo**: el archivo solo construye el prompt, así que no hay forma de comprobar con él que el modelo responda como en los ejemplos.
- `FewShotChatMessagePromptTemplate` y `FewShotPromptTemplate`: son clases propias de LangChain para este fin, pero no aparecen en el archivo.
- Cómo se eligen o se seleccionan dinámicamente los ejemplos.

### Observaciones sobre el archivo

- En la sección 2, `mensajes_historial` se calcula pero **no se imprime** ✅.
- Las listas `ejemplos_extraccion` y `ejemplos_traduccion` se definen pero **no se usan** con ninguna plantilla ✅.
- Las afirmaciones de la sección 5 (el modelo "puede alucinar el formato" sin `MessagesPlaceholder`, o que es "menos efectivo") no se han podido verificar con las fuentes permitidas; se presentan como opinión del curso.

---

## Autoevaluación

1. ¿Qué es el *In-Context Learning*?
2. ¿Qué estructura tiene un prompt few-shot?
3. ¿En qué se parecen y en qué se diferencian un historial y unos ejemplos en `MessagesPlaceholder`?
4. ¿Cuántos mensajes tiene la lista formateada de la Lección 3 y cómo se reparten?
5. ¿Con qué debe coincidir la clave `ejemplos` que se pasa a `format_messages`?
6. ¿Por qué los ejemplos usan el mismo prefijo `Texto a analizar:` que la plantilla?
7. ¿Se usan en algún prompt las listas `ejemplos_extraccion` y `ejemplos_traduccion`?
8. Cita dos de las cuatro buenas prácticas para ejemplos few-shot.

### Respuestas

1. Enseñar al modelo una tarea mediante ejemplos incluidos en el propio prompt.
2. Un mensaje `system` con las instrucciones, varios pares `Human`/`AI` de ejemplo y la pregunta real.
3. Ambos son listas de `HumanMessage` / `AIMessage` insertadas con un `MessagesPlaceholder`; el historial aporta memoria de una conversación y los ejemplos enseñan un patrón.
4. Ocho: 1 de sistema, 6 de ejemplos (3 pares) y 1 pregunta real.
5. Con el `variable_name` del `MessagesPlaceholder` (`"ejemplos"`).
6. Para que la pregunta real tenga el mismo formato que los ejemplos y el modelo reconozca el patrón.
7. No: solo se definen; el script imprime una línea descriptiva de cada una.
8. Por ejemplo: usar entre 2 y 5 ejemplos; cubrir casos diversos; mantener un formato consistente; asegurar salidas perfectas.

---

## Resumen

| Concepto | En una línea |
|---|---|
| ICL | Enseñar con ejemplos dentro del prompt |
| Few-shot | De 2 a 5 pares entrada → salida |
| `MessagesPlaceholder` | Aloja historial o ejemplos como lista de mensajes |
| Par `Human`/`AI` | Un ejemplo: entrada y salida esperada |
| Formato consistente | Mismo patrón en ejemplos y en la pregunta real |
| Prompt resultante | `system` + ejemplos + pregunta real (8 mensajes en el ejemplo) |

---

## Fuentes consultadas

- Referencia de `MessagesPlaceholder`: <https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder>
- Referencia de `ChatPromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate>
- Referencia de `FewShotChatMessagePromptTemplate` (solo firma y bases): <https://reference.langchain.com/python/langchain-core/prompts/few_shot/FewShotChatMessagePromptTemplate>
- Guía antigua de few-shot en modelos de chat (redirige a la portada actual): <https://python.langchain.com/docs/how_to/few_shot_examples_chat>
- Material del tema: `2.6_MessagesPlaceholder_y_FewShot_Examples.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
