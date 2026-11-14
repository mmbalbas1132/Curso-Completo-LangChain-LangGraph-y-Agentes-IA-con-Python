# 2.4 Chatbot con `ChatPromptTemplate`, `MessagesPlaceholder`, DeepSeek y Streamlit

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.4_chatbot_sudio.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice. Lo que se refiere a **Streamlit** no es de LangChain y no está en las fuentes permitidas: se presenta como explicación del archivo 📘.

---

## Antes de empezar

### Qué vas a aprender

1. Construir un chatbot web con Streamlit que usa una **cadena** `ChatPromptTemplate | modelo`.
2. Estructurar el prompt como **system + historial + pregunta actual** con `MessagesPlaceholder`.
3. Cambiar la **personalidad** del asistente desde la barra lateral.
4. Mostrar la respuesta en **streaming**.
5. Gestionar el **estado** de la conversación, la **clave de API** y los **errores**.

### Qué necesitas

- Haber visto `ChatPromptTemplate` (documento 2.3), los mensajes `HumanMessage` / `AIMessage` y el operador `|`.
- Conceptos básicos de Streamlit: el script se **re-ejecuta entero** en cada interacción y solo `st.session_state` conserva valores. 📘
- Una clave `DEEPSEEK_API_KEY`.

### Mapa del tema

```text
Antes (chatbot básico)                Ahora (este archivo)
chat_model.invoke(messages)     →     cadena.stream({"historial": ..., "mensaje": ...})
sin system message              →     system message según la personalidad elegida
historial = lista directa       →     historial = MessagesPlaceholder
sin streaming                   →     streaming con st.empty()
sin configuración dinámica      →     barra lateral: modelo, temperatura, personalidad
```

Estructura del prompt:

```text
[ ("system", <personalidad>),
  MessagesPlaceholder("historial"),   ← lista de HumanMessage / AIMessage
  ("human", "{mensaje}") ]            ← pregunta actual
```

### Glosario

| Término | Significado |
|---|---|
| **`MessagesPlaceholder`** | Hueco de una plantilla de chat que se rellena con una **lista de mensajes**. |
| **`variable_name`** | Nombre del hueco; debe coincidir con la clave del diccionario de entrada. |
| **Personalidad** | Texto del mensaje `system` que define cómo responde el asistente. |
| **Streaming** | Recibir la respuesta por fragmentos (`AIMessageChunk`) mientras se genera. |
| **`st.session_state`** | Almacén de Streamlit que persiste entre re-ejecuciones. 📘 |
| **Rerun** | Nueva ejecución del script completo; ocurre en cada interacción. 📘 |
| **`st.stop()`** | Corta la ejecución actual del script sin mostrar *traceback*. 📘 |
| **Idempotente** | Que llamarla varias veces produce el mismo efecto que una sola. |
| **Operador walrus `:=`** | Asigna y comprueba en la misma expresión. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-core langchain-deepseek streamlit python-dotenv
```

Archivo `.env`:

```bash
DEEPSEEK_API_KEY=tu_clave
```

Ejecución. 🔀 *(discrepancia)* El archivo indica `streamlit run chatbot_estudio.py`, pero el archivo se llama `2.4_chatbot_sudio.py`. Usa el nombre real:

```bash
streamlit run 2.4_chatbot_sudio.py
```

El archivo advierte: **no** uses `python archivo.py`; Streamlit necesita su propio *runner*. 📘

---

### Lección 1: importaciones y entorno (secciones 1 y 2 del archivo)

```python
import os
from dotenv import load_dotenv

import streamlit as st

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

load_dotenv()
```

| Importación | Para qué |
|---|---|
| `os` | Comprobar la clave de API |
| `streamlit as st` | Interfaz web |
| `ChatDeepSeek` | Modelo de chat de DeepSeek |
| `ChatPromptTemplate`, `MessagesPlaceholder` | Plantilla del prompt y hueco para el historial |
| `SystemMessage`, `HumanMessage`, `AIMessage` | `SystemMessage` para decidir qué pintar; `HumanMessage` y `AIMessage` para guardar y pintar el historial |

**Rutas de importación.** El archivo indica que `langchain.prompts` es la ruta antigua (anterior a la 0.2) y que el curso usa `langchain_core`. Mapeo legacy → moderno que cita: 📘

| Antigua | Actual |
|---|---|
| `langchain.prompts` | `langchain_core.prompts` |
| `langchain.schema` | `langchain_core.messages` |

**`load_dotenv()` aquí y una sola vez.** Debe ejecutarse al arrancar. Streamlit re-ejecuta el script en cada interacción, pero `load_dotenv()` es idempotente (no sobrescribe lo ya cargado). 📘 ⚠️ *(comportamiento de `python-dotenv`, no de LangChain)*

**Regla de oro de Streamlit.** `st.set_page_config()` debe ser la primera llamada `st.*`; por eso `load_dotenv()` (que no es `st.*`) va antes. 📘

**Otras notas del archivo** 📘:

- `MessagesPlaceholder` es la pieza que diferencia esta versión de un `ChatPromptTemplate` simple: permite insertar una **lista** de mensajes nativos, no un texto con el historial concatenado.
- `langchain_deepseek` es un paquete "satélite", igual que `langchain_openai`: todos exponen una clase `ChatXxx` con la misma interfaz (`invoke`, `stream`…).
- Modelo y plantilla se unen con `|` porque ambos son Runnables. ⚠️ *(no contrastado)*

---

### Lección 2: página y verificación de la clave (secciones 3 y 4 del archivo)

```python
st.set_page_config(
    page_title="Chatbot con DeepSeek",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")

if not os.getenv("DEEPSEEK_API_KEY"):
    st.error(
        "❌ Falta DEEPSEEK_API_KEY. Defínela en tu archivo `.env` "
        "o expórtala en el entorno antes de arrancar Streamlit."
    )
    st.stop()  # Detiene el script aquí (no lanza excepción).
```

| Elemento | Qué hace |
|---|---|
| `page_title`, `page_icon`, `layout="wide"` | Título y emoji de la pestaña; `wide` usa todo el ancho |
| `st.title`, `st.markdown` | Cabecera de la app |
| `if not os.getenv("DEEPSEEK_API_KEY")` | Comprueba la clave **antes** de usarla |
| `st.error(...)` + `st.stop()` | Muestra un mensaje claro y detiene el script sin *traceback* |

**Fail-fast "amable".** En los scripts de consola se usaba `raise RuntimeError(...)`; en una app de usuario se usa `st.error()` + `st.stop()`, porque un *traceback* en el servidor es ruido innecesario. 📘

---

### Lección 3: barra lateral y prompt dinámico (sección 5 del archivo)

**Objetivo.** Permitir elegir temperatura, modelo y personalidad, y construir la cadena con esos valores.

> Todo el bloque se ejecuta **entero en cada interacción**: `chat_model` y `chat_prompt` se **recrean** en cada rerun. Son objetos baratos, pero hay que entenderlo: si el usuario cambia la temperatura, el nuevo valor se aplica en la **siguiente** interacción, no retroactivamente. 📘

#### 3.1 Controles

```python
with st.sidebar:
    st.header("⚙️ Configuración")

    temperature = st.slider(
        "Temperatura", 0.0, 1.0, 0.7, 0.1
    )

    model_name = st.selectbox(
        "Modelo",
        ["deepseek-chat", "deepseek-reasoner"],
    )

    personalidad = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal",
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido",
        ],
    )
```

| Control | Valores | Notas del archivo 📘 |
|---|---|---|
| `st.slider` | mín. `0.0`, máx. `1.0`, defecto `0.7`, paso `0.1` | `0.0` determinista; `0.7` equilibrado; `1.0` creativo |
| `st.selectbox` de modelo | `deepseek-chat`, `deepseek-reasoner` | `chat`: rápido y general; `reasoner`: razonamiento profundo, más lento y caro |
| `st.selectbox` de personalidad | 5 opciones | Elige el mensaje `system` |

#### 3.2 Diccionario de personalidades

```python
    system_messages = {
        "Útil y amigable": (
            "Eres un asistente útil y amigable llamado ChatBot Pro. "
            "Responde de manera clara y concisa."
        ),
        "Profesional y formal": (
            "Eres un asistente profesional y formal. "
            "Proporciona respuestas precisas y bien estructuradas."
        ),
        "Casual y relajado": (
            "Eres un asistente casual y relajado. "
            "Habla de forma natural y amigable, como un buen amigo."
        ),
        "Experto técnico": (
            "Eres un asistente experto técnico. "
            "Proporciona respuestas detalladas con precisión técnica."
        ),
        "Creativo y divertido": (
            "Eres un asistente creativo y divertido. "
            "Usa analogías, ejemplos creativos y mantén un tono alegre."
        ),
    }
```

**Por qué un diccionario y no `if/elif`.** Legibilidad (cada personalidad con su mensaje), extensibilidad (añadir una personalidad = añadir una entrada) y reutilización. 📘

#### 3.3 Modelo, plantilla y cadena

```python
    chat_model = ChatDeepSeek(
        model=model_name,
        temperature=temperature,
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_messages[personalidad]),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{mensaje}"),
    ])

    cadena = chat_prompt | chat_model
```

| Elemento | Qué hace |
|---|---|
| `ChatDeepSeek(model=..., temperature=...)` | Recrea el modelo con los valores actuales de los controles |
| `("system", system_messages[personalidad])` | Mensaje de sistema según la personalidad elegida |
| `MessagesPlaceholder(variable_name="historial")` | Hueco donde se inserta la **lista** del historial |
| `("human", "{mensaje}")` | Pregunta actual |
| `chat_prompt \| chat_model` | Cadena: recibe `{"mensaje": str, "historial": list}`, rellena la plantilla, envía los mensajes al modelo y devuelve un `AIMessage` (o un generador con `.stream()`) |

**Por qué `MessagesPlaceholder` y no un texto con el historial.** El archivo da cuatro razones: es **nativo** (el modelo recibe `HumanMessage` y `AIMessage`), **escalable** (sin límite de "10 mensajes concatenados"; se puede truncar por tokens), **limpio** (el modelo distingue cada turno) y **compatible** con *tool calling*. 📘 ⚠️ *(no contrastado)*

**Contraste oficial.** La documentación oficial de `ChatPromptTemplate` muestra el mismo concepto: `("placeholder", "{conversation}")` inyecta una lista opcional de mensajes en tiempo de ejecución, y equivale a `MessagesPlaceholder(variable_name="conversation", optional=True)`.

🔀 *(matiz)* En el archivo, `MessagesPlaceholder(variable_name="historial")` se crea **sin** `optional=True`. La referencia oficial de `MessagesPlaceholder` muestra que, sin `optional=True`, omitir la clave del historial lanza `KeyError`, y con `optional=True` devuelve una lista vacía; en este chatbot la clave `"historial"` se pasa siempre (aunque la lista esté vacía), por lo que no hay problema.

**Cambiar la personalidad a mitad de conversación.** El `system` cambia para los mensajes siguientes; los que ya están en el historial conservan el efecto de la personalidad anterior, porque cada llamada es independiente. 📘

**`from_messages` acepta** tuplas `(rol, plantilla)` y objetos `MessagesPlaceholder`; mezclarlos es lo normal en un chatbot conversacional. 📘

---

### Lección 4: estado, historial y botón de reinicio (secciones 6, 7 y 8 del archivo)

#### 4.1 Memoria con `st.session_state`

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

- Un diccionario que **persiste entre reruns**; es la única forma de mantener estado en Streamlit. 📘
- Una variable global se reinicializaría en cada ejecución.
- Patrón "setdefault": la primera vez la clave no existe y se crea vacía; después ya existe y no se sobrescribe (si no, se perdería el historial).
- Es **por sesión** (por pestaña). Al recargar (F5) el historial se pierde; para persistencia real haría falta una base de datos. 📘
- Se guardan objetos LangChain (`HumanMessage` / `AIMessage`), que se pasan **directamente** al `MessagesPlaceholder`. 📘

#### 4.2 Pintar el historial

```python
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)
```

Streamlit reconstruye la página en cada rerun; si no se vuelve a pintar el historial, los mensajes antiguos desaparecerían. 📘

Los roles de Streamlit son distintos de los de LangChain. Mapeo según el archivo:

| LangChain | Streamlit |
|---|---|
| `human` | `user` |
| `ai` | `assistant` |
| `system` | no se muestra (configuración interna) |

#### 4.3 Botón "Nueva conversación"

```python
if st.button("🗑️ Nueva conversación"):
    st.session_state.messages = []
    st.rerun()
```

- `st.button` devuelve `True` solo en el rerun en que se pulsa. 📘
- Sin `st.rerun()`, el historial se vaciaría pero la pantalla seguiría mostrando los mensajes antiguos, porque ya se pintaron antes del `if`. 📘
- El archivo menciona que antiguamente era `st.experimental_rerun()` (obsoleto). 📘

---

### Lección 5: entrada, streaming y errores (sección 9 del archivo)

```python
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    with st.chat_message("user"):
        st.markdown(pregunta)

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in cadena.stream({
                "historial": st.session_state.messages,  # lista de BaseMessage
                "mensaje": pregunta,                     # str
            }):
                full_response += chunk.content

                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.messages.append(HumanMessage(content=pregunta))
        st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info(
            "Verifica que tu DEEPSEEK_API_KEY esté configurada correctamente "
            "y que tengas crédito disponible en tu cuenta."
        )
```

**Flujo de un turno:**

```text
1. El usuario envía la pregunta
2. Se pinta su burbuja (sin guardarla aún en session_state)
3. cadena.stream({...}) → fragmentos → se acumulan y se repintan con el cursor ▌
4. Se quita el cursor
5. Solo si no hubo error: se guardan HumanMessage y AIMessage en el historial
```

| Elemento | Qué hace y por qué |
|---|---|
| `if pregunta := st.chat_input(...)` | `st.chat_input` devuelve el texto solo en el rerun en que se envía; el walrus evita repetir la llamada. 📘 |
| Pintar la pregunta **sin** guardarla aún | Si se guardara antes, aparecería **dos veces**: una por este `with` y otra por el bucle del historial en el siguiente rerun. 📘 |
| `st.empty()` | Hueco que se puede actualizar repetidamente sin añadir widgets: técnica estándar para el streaming en Streamlit. 📘 |
| `cadena.stream({...})` | Generador de `AIMessageChunk`; cada uno trae un trozo de texto en `.content`. |
| `"historial"` y `"mensaje"` | Claves que **deben coincidir exactamente** con el nombre del `MessagesPlaceholder` y con la variable `{mensaje}`. Un error tipográfico lanzaría `KeyError` antes de llegar a la API. 📘 |
| `full_response + "▌"` | Cursor visual que simula que el modelo sigue escribiendo; se retira al final. |
| `try/except` | Cualquier fallo (red, cuota, *timeout*) se muestra como mensaje claro, no como *traceback*. |
| `append` al final | Si el stream falla a medias, no se guarda una respuesta incompleta. Si se prefiere guardar lo parcial, el archivo sugiere mover los `append` a un bloque `finally`. 📘 |

**Contraste oficial del streaming.** La documentación oficial indica que `stream()` devuelve un iterador que emite `AIMessageChunk` a medida que se producen y que mejora la capacidad de respuesta percibida en salidas largas; los fragmentos se pueden sumar para formar el mensaje completo. Que esto aplique igual a una **cadena** (`plantilla | modelo`) no aparece descrito en las páginas consultadas. ⚠️

**Nota sobre `chunk.content`.** Aquí se acumula `chunk.content` directamente, sin comprobar si está vacío. La documentación oficial usa `chunk.text` en sus ejemplos de streaming. ⚠️ *(no se contrastó la diferencia entre ambos atributos)*

**Guardar objetos y no textos.** Ventaja: `MessagesPlaceholder` los consume sin serializar. Desventaja: para guardarlos en disco habría que serializarlos (el archivo menciona `.model_dump()`). 📘 ⚠️

---

### Ejercicios propuestos en el archivo

| | Ejercicio |
|---|---|
| A | Haz que `deepseek-reasoner` sea el modelo por defecto. Observa cómo cambia el tiempo de respuesta (el archivo dice que es más lento). |
| B | Añade una personalidad nueva (por ejemplo, "Pirata gracioso") y comprueba que aparece en la barra lateral y que el modelo responde en consecuencia. |
| C | Trunca el historial a los últimos 10 mensajes con `"historial": st.session_state.messages[-10:]`. ¿Cómo afecta a la calidad y al coste de tokens? |
| D | Añade un contador "Mensajes: N" en la barra lateral. Pista: `len(st.session_state.messages)`. |
| E | Haz que el `system` sea una variable del usuario (`st.text_input` en la barra lateral). ¿Qué flexibilidad ganas y qué riesgo introduces? |
| F | (Avanzado) Sustituye el diccionario por un `st.text_area` con el prompt del sistema. ¿Cómo cambian la experiencia de usuario y la robustez del prompt? |
| G | (Avanzado) Añade un botón "Descargar conversación" que genere un `.txt`. Pista: `st.download_button`. |
| H | (Avanzado) Persiste el historial en un JSON local y cárgalo al inicio. Pista: `json.dump` / `json.load` con `message.model_dump()`. |

---

## Guía práctica

### ¿Cómo añado un historial nativo a una plantilla de chat?

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_messages[personalidad]),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{mensaje}"),
])
```

Y pásalo al invocar: `{"historial": lista_de_mensajes, "mensaje": pregunta}`.

### ¿Cómo cambio la personalidad del asistente?

Guarda cada mensaje `system` en un diccionario y elige uno con un `st.selectbox` (Lección 3).

### ¿Cómo muestro la respuesta mientras se genera?

`st.empty()` + bucle sobre `cadena.stream(...)` + `placeholder.markdown(texto + "▌")` (Lección 5).

### ¿Cómo evito que un mensaje aparezca duplicado?

No lo guardes en `session_state` hasta que termine la generación; la pantalla lo pinta ya con el `with st.chat_message("user")`.

### ¿Cómo detengo la app si falta la clave?

```python
if not os.getenv("DEEPSEEK_API_KEY"):
    st.error("❌ Falta DEEPSEEK_API_KEY. ...")
    st.stop()
```

### ¿Cómo limpio la conversación?

Vacía la lista y llama a `st.rerun()` (Lección 4).

### ¿Cómo limito cuánto historial se envía?

Pasa solo los últimos mensajes: `st.session_state.messages[-10:]` (ejercicio C). 📘

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Error al ejecutar con `python archivo.py` | Streamlit necesita su *runner* | `streamlit run 2.4_chatbot_sudio.py` 📘 |
| Error de Streamlit sobre `set_page_config` | No es la primera llamada `st.*` | Colocarla justo después de `load_dotenv()` 📘 |
| Aparece el mensaje "Falta DEEPSEEK_API_KEY" | La clave no está en el entorno | Definirla en `.env` o exportarla |
| `KeyError` al invocar la cadena | Las claves del diccionario no coinciden con la plantilla | Usar exactamente `"historial"` y `"mensaje"` 📘 |
| Los mensajes salen duplicados | Se guardó la pregunta antes del stream | Guardar al final |
| Tras "Nueva conversación" se ven los mensajes antiguos | Falta `st.rerun()` | Llamarlo tras vaciar la lista 📘 |
| El historial desaparece al recargar | `session_state` es por sesión | Persistir en disco o base de datos 📘 |
| Cambias la temperatura y no notas efecto inmediato | Se aplica en la siguiente interacción | Esperar al siguiente mensaje 📘 |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `import os` | Biblioteca estándar | Comprobar la clave |
| `from dotenv import load_dotenv` | `python-dotenv` | Cargar `.env` |
| `import streamlit as st` | `streamlit` | Interfaz |
| `from langchain_deepseek import ChatDeepSeek` | `langchain-deepseek` | Modelo |
| `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder` | `langchain-core` | Prompt |
| `from langchain_core.messages import SystemMessage, HumanMessage, AIMessage` | `langchain-core` | Mensajes |

### Variables de entorno

| Variable | Uso | Confirmada en documentación oficial |
|---|---|---|
| `DEEPSEEK_API_KEY` | Clave de `ChatDeepSeek` | Sí |

### Modelos de DeepSeek

| Modelo | Documentación oficial | Archivo |
|---|---|---|
| `deepseek-chat` | DeepSeek-V3; admite *tool calling* y salida estructurada | Rápido, general |
| `deepseek-reasoner` | DeepSeek-R1; no admite *tool calling* ni salida estructurada | Razonamiento profundo, más lento y caro 📘 |

El archivo pasa `temperature` también a `deepseek-reasoner`. La documentación oficial consultada no indica si ese modelo admite ese parámetro. ⚠️

### Entrada y salida de la cadena

| Elemento | Detalle |
|---|---|
| Entrada | `{"historial": list[BaseMessage], "mensaje": str}` |
| Prompt resultante | `system` + mensajes del historial + `human` con la pregunta |
| Salida de `.stream(...)` | Generador de `AIMessageChunk` |
| Salida de `.invoke(...)` | `AIMessage` |

### Componentes de Streamlit usados

| Elemento | Función 📘 |
|---|---|
| `st.set_page_config(page_title, page_icon, layout)` | Metadatos y diseño; primera llamada `st.*` |
| `st.title`, `st.markdown`, `st.header` | Texto y cabeceras |
| `st.error`, `st.info` | Mensajes de error e información |
| `st.stop()` | Detiene el script sin *traceback* |
| `st.sidebar` (`with`) | Barra lateral |
| `st.slider(label, min, max, valor, paso)` | Control deslizante |
| `st.selectbox(label, opciones)` | Menú desplegable |
| `st.session_state` | Estado persistente |
| `st.chat_message(rol)` | Burbuja de chat (`user` / `assistant`) |
| `st.button`, `st.rerun()` | Botón y re-ejecución forzada |
| `st.chat_input(texto)` | Entrada fija inferior; devuelve el texto solo al enviarse |
| `st.empty()` | Hueco reescribible |

### Roles: LangChain frente a Streamlit

| Clase LangChain | Rol en tuplas | Rol en Streamlit |
|---|---|---|
| `HumanMessage` | `human` | `user` |
| `AIMessage` | `ai` | `assistant` |
| `SystemMessage` | `system` | (no se pinta) |

### Comparativa con el chatbot básico

| Aspecto | Chatbot básico | Este archivo |
|---|---|---|
| Invocación | `chat_model.invoke(messages)` | `cadena.stream({...})` |
| System message | No | Sí, por personalidad |
| Historial | Lista directa | `MessagesPlaceholder` |
| Streaming | No | Sí |
| Configuración | Fija | Barra lateral |
| Clave de API | Sin comprobar | `st.error` + `st.stop()` |

### Patrones reutilizables (resumen del archivo)

| # | Patrón | Idea |
|---|---|---|
| 1 | Cabecera estándar con `dotenv` | Fail-fast "amable" con `st.error` + `st.stop` |
| 2 | `set_page_config` primero | Primera llamada `st.*` |
| 3 | Sidebar recreado por rerun | Propaga la configuración actual |
| 4 | `session_state` setdefault | Crear la clave solo la primera vez |
| 5 | System + placeholder + human | Esqueleto de un chatbot moderno |
| 6 | Personalidad con diccionario | Sin `if/elif` en cadena |
| 7 | Streaming con `st.empty()` | Repintar con cursor `▌` |
| 8 | Rerun explícito | Limpiar la interfaz |
| 9 | Guardar al final | Evitar duplicados |
| 10 | Cadena frente a modelo directo | La cadena es más estructurada y configurable |

---

## Explicación

### ¿Por qué usar una cadena en lugar de `chat_model.invoke(messages)`?

Pasar la lista directamente funciona, pero sin `system`, sin plantilla y sin flexibilidad. Con `ChatPromptTemplate` el modelo recibe una estructura clara con roles y contexto, la personalidad es configurable y el historial es nativo. 📘

### ¿Por qué se recrean modelo y plantilla en cada rerun?

Porque Streamlit ejecuta el script completo en cada interacción; así los valores actuales de los controles llegan al modelo y a la plantilla. Son objetos baratos. 📘

### ¿Por qué el historial se pasa como lista y no como texto?

Porque es lo que el modelo espera: mensajes tipados por rol. El archivo contrasta esta versión con la de `chatbot_mejorado`/Partes anteriores, donde el historial se serializaba a texto. 📘

### ¿Por qué no se guarda la pregunta antes del stream?

Porque saldría duplicada: se pinta una vez en el momento y otra al recorrer el historial en el siguiente rerun. 📘

### ¿Por qué `st.stop()` y no `raise`?

En una app de usuario, un `raise` genera un *traceback* en la consola del servidor; `st.stop()` corta el rerun y deja la interfaz como estaba. 📘

### ¿Qué ocurre con el historial si falla el stream?

No se guarda nada: los `append` están después del bucle, dentro del `try`. 📘

### ¿Cuándo usar este enfoque?

| Situación | Enfoque |
|---|---|
| Chatbot con personalidad y memoria | `ChatPromptTemplate` + `MessagesPlaceholder` |
| Prueba mínima sin `system` | `chat_model.invoke(messages)` |
| Respuestas largas | Streaming |
| App para usuarios finales | `st.error` + `st.stop`, no `raise` |

### Lo que este documento no cubre

- Otros archivos del curso que el original menciona (Partes 1 y 2 de sentimientos, `2.3` y `streamlit_chatbot.py`): no son fuente de este documento.
- Persistencia en base de datos, memoria con resumen, *tool calling* y RAG: el archivo los cita como próximos pasos.
- La documentación oficial de Streamlit: no está entre las fuentes permitidas.
- Una descripción oficial en prosa de `MessagesPlaceholder` y de la composición con `|`: la referencia solo trae firmas y ejemplos (ver documento 2.5 para `MessagesPlaceholder`).

### Observaciones sobre el archivo

- 🔀 El comando de ejecución del archivo usa un nombre (`chatbot_estudio.py`) distinto del real (`2.4_chatbot_sudio.py`).
- El archivo describe como obsoleta la ruta `langchain.prompts`; la documentación consultada no habla de rutas de importación.
- El archivo pasa `temperature` a ambos modelos sin distinción; no se pudo contrastar si `deepseek-reasoner` lo admite.
- Al acumular `chunk.content` no se filtran fragmentos sin contenido.

---

## Autoevaluación

1. ¿Qué tres piezas forman el prompt de este chatbot?
2. ¿Para qué sirve `MessagesPlaceholder`?
3. ¿Qué claves debe llevar el diccionario que se pasa a `cadena.stream(...)` y por qué?
4. ¿Por qué `st.set_page_config()` va después de `load_dotenv()`?
5. ¿Qué hacen `st.error()` y `st.stop()` cuando falta la clave?
6. ¿Por qué no se añade la pregunta a `session_state` antes de generar la respuesta?
7. ¿Qué hace `st.empty()` en el streaming?
8. ¿Por qué hace falta `st.rerun()` en el botón "Nueva conversación"?
9. ¿Qué rol de Streamlit corresponde a `AIMessage`?
10. ¿Cuándo se guardan los mensajes en el historial y por qué?

### Respuestas

1. Un mensaje `system` (personalidad), un `MessagesPlaceholder("historial")` y un mensaje `human` con `{mensaje}`.
2. Para insertar el historial como una lista de mensajes nativos dentro de la plantilla.
3. `"historial"` (nombre del `MessagesPlaceholder`) y `"mensaje"` (variable de la plantilla); si no coinciden, se lanza `KeyError`.
4. Porque `st.set_page_config()` debe ser la primera llamada `st.*`, y `load_dotenv()` no lo es.
5. Muestran un mensaje de error claro y detienen el script sin lanzar excepción.
6. Porque saldría duplicada: se pinta al momento y otra vez al recorrer el historial en el siguiente rerun.
7. Crea un hueco que se actualiza repetidamente sin añadir widgets nuevos.
8. Porque sin él, la pantalla seguiría mostrando los mensajes antiguos aunque la lista estuviera vacía.
9. `assistant`.
10. Al final, solo si la generación terminó sin error, para no guardar respuestas incompletas.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `ChatPromptTemplate` | Estructura el prompt con roles |
| `MessagesPlaceholder` | Hueco para una lista de mensajes (historial) |
| `cadena.stream(...)` | Respuesta por fragmentos |
| `st.session_state` | Memoria entre reruns |
| `st.empty()` | Contenedor para repintar el streaming |
| `st.stop()` | Detiene el script sin traceback |
| `st.rerun()` | Fuerza una nueva ejecución |
| Personalidad | Mensaje `system` elegido de un diccionario |
| Guardar al final | Evita duplicados y respuestas incompletas |

---

## Fuentes consultadas

- Referencia de `ChatPromptTemplate` (forma `placeholder` equivalente a `MessagesPlaceholder`): <https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate>
- Referencia de `MessagesPlaceholder` (`variable_name`, `optional`): <https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder>
- Documentación oficial de modelos (`stream`, `AIMessageChunk`): <https://docs.langchain.com/oss/python/langchain/models>
- Documentación oficial de mensajes: <https://docs.langchain.com/oss/python/langchain/messages>
- Documentación oficial de `ChatDeepSeek`: <https://python.langchain.com/docs/integrations/chat/deepseek/> (redirige a <https://docs.langchain.com/oss/python/integrations/chat/deepseek>)
- Material del tema: `2.4_chatbot_sudio.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
