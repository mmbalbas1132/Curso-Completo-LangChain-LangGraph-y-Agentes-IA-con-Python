# 1.4 Un chatbot con Streamlit y memoria de conversación

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `streamlit_chatbot.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo (el comportamiento de Streamlit no está en las fuentes oficiales de LangChain consultadas); ⚠️ *(no contrastado)*; 🔀 *(discrepancia)* = el archivo se contradice o difiere de la documentación; ✅ *(ejecutado)*. La aplicación necesita una clave de API y un navegador, por lo que **no se ejecutó**; solo se comprobó con un modelo simulado que `invoke` acepta una lista de mensajes y devuelve un `AIMessage`.

---

## Antes de empezar

### Qué vas a aprender

1. Cómo "piensa" Streamlit: reejecuta todo el script en cada interacción.
2. Guardar el historial en `st.session_state`.
3. Usar `HumanMessage` y `AIMessage` y enviar **toda** la lista al modelo.
4. Pintar una conversación con `st.chat_message` y `st.chat_input`.

### Qué necesitas

- Documentos 1.1 a 1.3.
- `pip install streamlit langchain-deepseek python-dotenv` y `DEEPSEEK_API_KEY` en `.env`.

### Mapa del tema

```text
Usuario escribe ──► st.chat_input ──► pregunta
                                        │
 session_state.messages ◄── HumanMessage(pregunta)
        │
        └──► chat_model.invoke(TODA la lista) ──► AIMessage ──► se pinta y se guarda
Streamlit vuelve a ejecutar el script entero en cada interacción
```

### Glosario

| Término | Significado |
|---|---|
| **Streamlit** | Framework para crear webs interactivas en Python. 📘 |
| **Rerun** | Streamlit ejecuta el script de arriba abajo en cada interacción. 📘 |
| **`st.session_state`** | Almacén que **persiste** entre reruns; las variables normales se reinician. 📘 |
| **`HumanMessage` / `AIMessage` / `SystemMessage`** | Mensajes del usuario / del modelo / instrucciones del sistema. |
| **Operador walrus `:=`** | Asigna y comprueba en una sola expresión. |

---

## Tutorial

### Preparación

```bash
streamlit run streamlit_chatbot.py
```

---

### Lección 1: importaciones y constantes (secciones 1 y 2)

```python
import streamlit as st
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

MODEL_NAME = "deepseek-chat"

TEMPERATURE = 0.7
```

Las constantes en mayúsculas agrupan la configuración al principio. 📘

---

### Lección 2: configuración de la app y del modelo (sección 3)

```python
load_dotenv()

st.set_page_config(page_title="Chatbot con DeepSeek", page_icon="🤖", layout="wide")

st.title("🤖🤖🤖 Chatbot con LangChain y DeepSeek 🤖🤖🤖")
st.markdown("Escribe tu pregunta en el cuadro de texto y presiona Enter.")

chat_model = ChatDeepSeek(model=MODEL_NAME, temperature=TEMPERATURE)
```

| Línea | Qué hace |
|---|---|
| `st.set_page_config(...)` | Título e icono de la pestaña; `layout="wide"` usa todo el ancho. 📘 |
| `st.title` / `st.markdown` | Cabecera y texto de ayuda. |
| `ChatDeepSeek(...)` | Crea el modelo. |

🔀 El comentario del archivo dice "Instanciamos el modelo una sola vez", pero a la vez explica que esa línea se ejecuta **en cada rerun**. Ambas cosas no son compatibles: el modelo se crea de nuevo en cada interacción (lo que explica por qué en 1.5 se puede cambiar de modelo con un menú).

---

### Lección 3: la memoria (sección 4)

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Como el script se reejecuta y las variables normales se reinician, la lista de mensajes se crea **solo la primera vez** y se guarda en `session_state`. 📘

---

### Lección 4: pintar el historial (sección 5)

```python
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)
```

| Línea | Qué hace |
|---|---|
| `for message in ...` | Recorre los mensajes guardados. |
| `isinstance(message, SystemMessage): continue` | Omite mensajes del sistema (no se muestran). |
| `role = "assistant" if ... else "user"` | Traduce el tipo de mensaje al rol de Streamlit. |
| `with st.chat_message(role)` | Dibuja la burbuja con avatar. |

Así la conversación "no desaparece" en cada rerun: se vuelve a dibujar desde la memoria. 📘

---

### Lección 5: preguntar y responder (sección 6)

```python
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):

    with st.chat_message("user"):
        st.markdown(pregunta)

    st.session_state.messages.append(HumanMessage(content=pregunta))

    respuesta_ia = chat_model.invoke(st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(respuesta_ia.content)

    st.session_state.messages.append(respuesta_ia)
```

| Paso | Qué hace |
|---|---|
| `st.chat_input(...)` | Caja de texto; devuelve `None` hasta que se envía algo. |
| A) pintar la pregunta | Feedback inmediato. |
| B) `append(HumanMessage(...))` | La pregunta entra en el historial. |
| C) `chat_model.invoke(lista)` | Envía **toda** la lista; el modelo no tiene memoria propia, el contexto se lo damos nosotros. 📘 |
| D) pintar la respuesta | Burbuja del asistente. |
| E) `append(respuesta_ia)` | Ya es un `AIMessage`; se guarda tal cual. |

**Comprobado** ✅ (modelo simulado): `invoke` acepta una lista de mensajes y devuelve un `AIMessage`. La documentación oficial indica que `invoke` admite una lista de mensajes para conversaciones.

**Puntos clave.**

- La memoria es una lista que tú mantienes y reenvías entera.
- `session_state` hace que la lista sobreviva a los reruns.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo hago que el bot tenga una personalidad?

Añade un `SystemMessage` al principio de la lista (el archivo no lo hace; ver observaciones).

### ¿Cómo borro la conversación?

Este archivo no lo ofrece; el documento 1.5 añade un botón "Nueva conversación".

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| El bot "olvida" lo anterior | No se envía la lista completa | Pasar `st.session_state.messages` a `invoke` |
| Aparece un error de autenticación | Falta `DEEPSEEK_API_KEY` o `load_dotenv()` va después | Revisar `.env` |

---

## Referencia

| Elemento | Función |
|---|---|
| `st.set_page_config` | Configura pestaña y diseño |
| `st.session_state` | Memoria entre reruns |
| `st.chat_message(rol)` | Burbuja de chat |
| `st.chat_input(texto)` | Caja de entrada de chat |
| `HumanMessage(content=...)` | Mensaje del usuario |
| `AIMessage` | Respuesta del modelo (lo devuelve `invoke`) |
| `chat_model.invoke(lista)` | Envía el historial completo |

---

## Explicación

### ¿Por qué hay que reenviar todo el historial?

Cada llamada al modelo es independiente; si no se le manda lo anterior, no lo conoce. 📘

### ¿Por qué `session_state`?

Porque las variables normales se reinician con cada rerun. 📘

### Lo que este documento no cubre

- Streaming y manejo de errores (documento 1.5).
- Mensajes de sistema reales.

### Observaciones sobre el archivo

- 🔀 El comentario "una sola vez" contradice a su propia explicación (ver Lección 2).
- Nunca se añade un `SystemMessage`, así que el filtro `isinstance(message, SystemMessage)` no llega a actuar; queda preparado para el futuro.
- La pregunta se guarda en el historial **antes** de llamar al modelo: si `invoke` falla, el historial queda con una pregunta sin respuesta. El archivo no tiene `try/except` ni comprobación de clave (lo resuelve 1.5).
- El `import` de `SystemMessage` solo se usa en ese filtro.

---

## Autoevaluación

1. ¿Qué ocurre con el script cada vez que el usuario interactúa?
2. ¿Por qué se usa `st.session_state` para los mensajes?
3. ¿Qué se envía a `invoke` y por qué?
4. ¿Qué hace el operador `:=` en `if pregunta := st.chat_input(...)`?
5. ¿Qué es `respuesta_ia` y por qué no hay que envolverla?
6. ¿Qué contradicción tiene el comentario "una sola vez"?
7. ¿Qué pasa si `invoke` falla en este script?

### Respuestas

1. Se vuelve a ejecutar entero de arriba abajo.
2. Porque es lo único que persiste entre reruns.
3. La lista completa de mensajes, porque el modelo no tiene memoria propia.
4. Asigna el valor y comprueba si hay texto en la misma expresión.
5. Un `AIMessage` devuelto por `ChatDeepSeek`; ya tiene el tipo correcto.
6. Dice que el modelo se crea una vez, pero esa línea se ejecuta en cada rerun.
7. El script se detiene con error y el historial queda con la pregunta sin respuesta.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Rerun | Streamlit ejecuta todo el script en cada interacción |
| `session_state` | Memoria persistente de la lista de mensajes |
| `invoke(lista)` | Envía todo el historial |
| `chat_message` / `chat_input` | Burbujas y caja de entrada |

---

## Fuentes consultadas

- Documentación oficial de modelos: <https://docs.langchain.com/oss/python/langchain/models>
- Documentación oficial de `ChatDeepSeek`: <https://docs.langchain.com/oss/python/integrations/chat/deepseek>
- Material del tema: `streamlit_chatbot.py`. (Streamlit: sin fuente oficial consultada; todo lo relativo a Streamlit procede del archivo.) 📘

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
