# 1.5 Interfaz de chat para un asistente legal RAG con Streamlit (`app.py`)

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `app.py` (código original, sin modificar) y sus dos versiones `1.5_app.py` (comentada) y `1.5_app_limpio.py` (limpia), cuyo código es idéntico (✅ comprobado con el árbol sintáctico). Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar con este archivo ni con las fuentes permitidas; 🔀 *(discrepancia)*; ✅ *(ejecutado)* = observado al ejecutar la app con **Streamlit 1.65.0** y un módulo `rag_system` **simulado** (`streamlit.testing.v1.AppTest`). Este archivo **no usa ninguna clase de LangChain directamente**: toda la lógica RAG está en `rag_system.py` (documento 1.4). *Actualización:* este documento se escribió antes de leer `rag_system.py`, `config.py` y `prompts.py` (documentos 1.1, 1.2 y 1.4); tras leerlos se corrigieron las afirmaciones que dependían de ellos (los ⚠️ de "deducido del uso" quedan **confirmados** ✅ salvo lo indicado, y se corrigió lo relativo a los errores). La documentación de Streamlit no es una fuente permitida; lo que se afirma de Streamlit es 📘 o ✅.

---

## Antes de empezar

### Qué vas a aprender

1. Cómo se construye una interfaz de chat con Streamlit.
2. Cómo funciona el modelo de ejecución de Streamlit (el script se reejecuta entero) y por qué el historial se guarda en `st.session_state`.
3. Cómo se separa la **interfaz** (`app.py`) de la **lógica RAG** (`rag_system.py`).
4. Cómo se muestran en una segunda columna los fragmentos de documentos que respaldan la respuesta.

### Qué necesitas

- Documentos 1.4 y 1.5 de Tema_1 (Streamlit, `session_state`, chat).
- Saber qué es un RAG a grandes rasgos (este archivo no lo explica; ver "Lo que este documento no cubre").
- Para ejecutarla de verdad: `streamlit run app.py` con `rag_system.py` funcionando (necesita claves y modelos que este archivo no define).

### Mapa del tema

```text
Usuario escribe ──► st.chat_input ──► prompt
                                        │
              messages += {"role":"user", ...}
                                        │
                  query_rag(prompt)  ──►  (respuesta, docs)      ← lógica en rag_system.py
                                        │
     messages += {"role":"assistant", "content": respuesta, "docs": docs}
                                        │
                                   st.rerun()
                                        │
   col1: historial de burbujas        col2: fragmentos de la última respuesta (expanders)
```

### Glosario

| Término | Significado |
|---|---|
| **RAG** | Técnica en la que el modelo responde apoyándose en fragmentos recuperados de documentos. Aquí solo aparece en nombres y textos; su implementación está en otro archivo. |
| **Retriever** | Componente que busca los fragmentos relevantes. Este archivo solo muestra su "tipo". |
| **Rerun** | Reejecución completa del script que hace Streamlit en cada interacción. 📘 ✅ |
| **`st.session_state`** | Almacén que persiste entre reruns durante la sesión. 📘 |
| **Fragmento (`docs`)** | Trozo de documento usado para responder; aquí un diccionario con `fragmento`, `fuente`, `pagina` y `contenido`. |
| **Sidebar** | Barra lateral de la app. |

---

## Tutorial

### Preparación

```bash
pip install streamlit
streamlit run app.py
```

(`rag_system.py`, `config.py` y `prompts.py` están en la misma carpeta; este documento no los analiza.)

---

### Lección 1: importaciones y contrato con `rag_system` (líneas 1 y 2)

```python
import streamlit as st
from rag_system import query_rag, get_retriever_info
```

| Línea | Qué hace |
|---|---|
| `import streamlit as st` | Biblioteca de la interfaz web. |
| `from rag_system import ...` | Importa dos funciones del propio proyecto. |

Por cómo se usan más abajo, y ✅ confirmado al leer `rag_system.py` (documento 1.4):

- `query_rag(prompt)` devuelve **dos valores**: el texto de la respuesta y la lista de fragmentos (`docs_info`).
- `get_retriever_info()` devuelve un diccionario de 5 claves; aquí solo se lee `'tipo'` (con la configuración actual: `MMR + MultiQuery + Hybrid`).

---

### Lección 2: configuración de la página, título y estado (líneas 4 a 17)

```python
st.set_page_config(
    page_title="Sistema RAG - Asistente Legal",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
```

| Línea | Qué hace y por qué |
|---|---|
| `st.set_page_config(...)` | Título e icono de la pestaña; `layout="wide"` usa todo el ancho. 📘 |
| `st.title` / `st.divider` | Título grande y línea de separación. |
| `if "messages" not in st.session_state` | Crea la lista de mensajes **solo la primera vez**; después se conserva. |

**Estructura de cada mensaje** ✅ (comprobada ejecutando la app simulada): un diccionario `{"role": "user" | "assistant", "content": texto}` y, solo en los del asistente, una clave `"docs"` con los fragmentos.

**Punto clave.** Streamlit reejecuta el script entero en cada interacción; lo único que sobrevive es `st.session_state`. 📘

---

### Lección 3: barra lateral (líneas 19 a 36)

```python
with st.sidebar:
    st.header("📋 Información del Sistema")
    
    retriever_info = get_retriever_info()
    
    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {retriever_info['tipo']}")
    
    st.markdown("**🤖 Modelos:**")
    st.info("Consultas: GPT-4o-mini\nRespuestas: GPT-4o")
    
    st.divider()
    
    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
```

| Parte | Qué hace |
|---|---|
| `with st.sidebar:` | Todo lo de dentro va a la barra lateral. |
| `get_retriever_info()` | Se llama en **cada** reejecución. |
| `st.info(f"Tipo: ...")` | Muestra el tipo de retriever. ✅ (con el módulo simulado se mostró `Tipo: MMR (simulado)`). |
| `st.info("Consultas: GPT-4o-mini\nRespuestas: GPT-4o")` | Texto fijo escrito a mano. ✅ Coincide con `config.py` (`QUERY_MODEL = gpt-4o-mini`, `GENERATION_MODEL = gpt-4o`), pero no se actualiza si cambian allí. |
| `st.button(...)` | Devuelve `True` solo en la pasada en que se pulsa. |
| `st.session_state.messages = []` + `st.rerun()` | Vacía el historial y reinicia el script. |

**Comprobado** ✅ (Streamlit 1.65.0): tras enviar una pregunta y pulsar el botón, `messages` queda vacío y no hay burbujas ni fragmentos en pantalla.

**Observación** ✅: la barra lateral se ejecuta **antes** de pintar el historial, así que, probado **sin** `st.rerun()`, la pantalla también queda vacía. Aquí el `st.rerun()` del botón no es imprescindible (no hace daño).

---

### Lección 4: dos columnas e historial (líneas 38 a 47)

```python
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
```

| Parte | Qué hace |
|---|---|
| `st.columns([2, 1])` | Dos columnas con anchos relativos 2:1. |
| `for message in ...messages` | Recorre el historial guardado. |
| `st.chat_message(message["role"])` | Burbuja con el estilo del rol (`"user"` o `"assistant"`). |
| `st.markdown(message["content"])` | Pinta el texto del mensaje. |

Como el script se reejecuta, la conversación se **vuelve a pintar desde la memoria** en cada pasada.

---

### Lección 5: fragmentos de la última respuesta (líneas 49 a 64)

```python
with col2:
    st.markdown("### 📄 Documentos Relevantes")
    
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant" and "docs" in last_message:
            docs = last_message["docs"]
            
            if docs:
                for doc in docs:
                    with st.expander(f"📄 Fragmento {doc['fragmento']}", expanded=False):
                        st.markdown(f"**Fuente:** {doc['fuente']}")
                        st.markdown(f"**Página:** {doc['pagina']}")
                        st.markdown("**Contenido:**")
                        st.text(doc['contenido'])
```

| Condición o línea | Qué hace |
|---|---|
| `if st.session_state.messages:` | Evita fallar con el historial vacío. |
| `last_message = ...[-1]` | Toma el último mensaje. |
| `role == "assistant" and "docs" in last_message` | Solo muestra fragmentos si el último mensaje es del asistente y trae `docs`. |
| `if docs:` | Evita pintar nada si la lista está vacía. |
| `st.expander(..., expanded=False)` | Panel plegable, cerrado por defecto. |
| `st.text(doc['contenido'])` | Muestra el contenido tal cual, sin interpretar Markdown. |

**Comprobado** ✅: tras una pregunta con un fragmento simulado apareció un expander con la etiqueta `📄 Fragmento 1`.

✅ Las claves `fragmento`, `fuente`, `pagina` y `contenido` son las que genera `query_rag` en `rag_system.py` (documento 1.4): es un **contrato implícito**; si cambian allí, esta parte falla.

---

### Lección 6: entrada del usuario y respuesta (líneas 66 a 77)

```python
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("🔍 Analizando..."):
        response, docs = query_rag(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})
    
    st.rerun()
```

| Línea | Qué hace |
|---|---|
| `if prompt := st.chat_input(...)` | `st.chat_input` devuelve el texto enviado (o `None`); `:=` asigna y comprueba a la vez. |
| `append({"role": "user", ...})` | Guarda la pregunta **antes** de llamar al RAG. |
| `with st.spinner(...)` | Muestra "Analizando..." mientras se ejecuta el bloque. |
| `response, docs = query_rag(prompt)` | Obtiene respuesta y fragmentos. Solo recibe la pregunta actual. |
| `append({"role": "assistant", ..., "docs": docs})` | Guarda la respuesta con sus fragmentos. |
| `st.rerun()` | Reejecuta el script para pintar los mensajes nuevos. |

**¿Por qué el `st.rerun()` aquí sí es necesario?** El historial y el panel de fragmentos se pintan **antes** (columnas 1 y 2), con la lista antigua. **Comprobado** ✅: sin ese `st.rerun()`, en esa pasada no se pinta ninguna burbuja ni fragmento; con él aparecen la pregunta, la respuesta y el fragmento.

**Corrección sobre los errores.** `app.py` no tiene `try/except`, y ✅ con un `rag_system` simulado que lanzaba una excepción la ejecución se interrumpía dejando la pregunta sin respuesta. **Pero el `query_rag` real captura todas las excepciones** y devuelve `(texto_de_error, [])` (documento 1.4): en la app real el error se guarda y se muestra **como si fuera la respuesta del asistente**, sin fragmentos.

---

### Lección 7: pie de página (líneas 79 a 84)

```python
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con MMR Retriever</div>", 
    unsafe_allow_html=True
)
```

`unsafe_allow_html=True` permite que `st.markdown` interprete HTML. Aquí el HTML es un texto fijo del propio archivo, no datos del usuario. El texto nombra **MMR** (*Maximal Marginal Relevance*, un tipo de búsqueda de fragmentos); cómo se configura está en `config.py` y `rag_system.py` (documentos 1.1 y 1.4).

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo ejecuto la app?

`streamlit run app.py` desde la carpeta del proyecto, con `rag_system.py` funcionando.

### ¿Cómo borro la conversación?

Botón **Limpiar Chat** de la barra lateral. ✅

### ¿Cómo pruebo la interfaz sin claves?

Sustituye temporalmente `rag_system` por un módulo con las mismas dos funciones que devuelva datos de ejemplo (`(respuesta, [ {fragmento, fuente, pagina, contenido} ])`). Así se hizo la comprobación ✅ de este documento.

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| La "respuesta" es un texto `Error al procesar la cosulta: ...` | `query_rag` captura la excepción y la devuelve como texto ✅ | Leer el mensaje; ver 1.4 (con LangChain 1.x falla por imports y `get_relevant_documents`) |
| La app se detiene y la pregunta queda sin respuesta | Solo ocurriría si `query_rag` lanzara una excepción (no pasa con el real) ✅ | — |
| No aparecen fragmentos | La respuesta no trae `docs` o la lista está vacía | Revisar lo que devuelve `query_rag` |
| `KeyError` en el panel de fragmentos | Faltan claves en un fragmento | Comprobar `fragmento`, `fuente`, `pagina`, `contenido` |

---

## Referencia

### Elementos de Streamlit usados

| Elemento | Uso en el archivo |
|---|---|
| `st.set_page_config` | Título, icono y ancho de la página |
| `st.title`, `st.header`, `st.markdown` | Textos |
| `st.divider` | Separador |
| `st.session_state` | Historial persistente |
| `st.sidebar` | Barra lateral |
| `st.info` | Cuadros informativos |
| `st.button` | Botón "Limpiar Chat" |
| `st.columns` | Distribución en dos columnas |
| `st.chat_message` / `st.chat_input` | Burbujas y caja de entrada |
| `st.expander` / `st.text` | Fragmentos plegables |
| `st.spinner` | Indicador de carga |
| `st.rerun` | Reejecución del script |

### Estructura de un mensaje ✅

| Clave | Presente en | Contenido |
|---|---|---|
| `role` | todos | `"user"` o `"assistant"` |
| `content` | todos | Texto |
| `docs` | solo asistente | Lista de fragmentos |

### Estructura de un fragmento ✅ (confirmada en `rag_system.py`)

| Clave | Uso |
|---|---|
| `fragmento` | Etiqueta del expander |
| `fuente` | Se muestra como "Fuente" |
| `pagina` | Se muestra como "Página" |
| `contenido` | Texto del fragmento |

### Funciones importadas ✅ (confirmadas en `rag_system.py`)

| Función | Entrada | Salida |
|---|---|---|
| `query_rag` | pregunta (`str`) | `(respuesta, docs)` |
| `get_retriever_info` | — | diccionario con 5 claves (aquí se usa `'tipo'`) |

---

## Explicación

### ¿Por qué separar `app.py` y `rag_system.py`?

Porque la interfaz solo necesita dos funciones; cambiar el retriever o el modelo no obliga a tocar la pantalla. 📘

### ¿Por qué guardar los fragmentos dentro del mensaje del asistente?

Así cada respuesta lleva consigo las fuentes que la respaldan y el panel derecho siempre puede mostrar las de la **última** respuesta.

### ¿Por qué el historial no se envía al RAG?

A `query_rag` solo se le pasa la pregunta actual y ✅ ni `query_rag` ni la cadena de `rag_system.py` guardan memoria (documento 1.4): cada pregunta se responde de forma independiente.

### Lo que este documento no cubre

- `rag_system.py`, `config.py` y `prompts.py`: se documentan aparte (1.1, 1.2 y 1.4).
- La ejecución real contra modelos (requiere claves).
- La documentación oficial de Streamlit (no es fuente permitida); lo dicho de Streamlit es 📘 o ✅.

### Observaciones sobre el archivo

- `app.py` no tiene `try/except`, pero `query_rag` ya captura todo: los fallos se muestran como respuesta del asistente. ✅
- El texto de los modelos (`GPT-4o-mini` / `GPT-4o`) está escrito a mano (coincide con `config.py`). ✅
- Con LangChain 1.x, `rag_system.py` falla (1.4) y la interfaz solo mostraría el texto de error. ✅
- El `st.rerun()` del botón no es imprescindible; el del final sí. ✅
- `get_retriever_info()` se llama en cada reejecución.
- El archivo no lleva prefijo numérico (`app.py` dentro de la carpeta `Asistente_Legal_RAG`); se le asignó el número 1.5.

---

## Autoevaluación

1. ¿Qué ocurre con el script cada vez que el usuario interactúa?
2. ¿Por qué el historial se guarda en `st.session_state`?
3. ¿Qué dos valores devuelve `query_rag` según su uso?
4. ¿Qué contiene cada mensaje del asistente que no tienen los del usuario?
5. ¿Por qué hace falta el `st.rerun()` tras guardar la respuesta?
6. ¿Qué ve el usuario si `query_rag` falla (con el `rag_system.py` real)?
7. ¿Por qué el `st.rerun()` del botón "Limpiar Chat" no es imprescindible?
8. ¿Qué parte de este archivo no contiene lógica RAG y dónde está esa lógica?

### Respuestas

1. Se reejecuta entero de arriba abajo.
2. Porque es lo único que persiste entre reejecuciones.
3. El texto de la respuesta y la lista de fragmentos.
4. La clave `docs` con los fragmentos usados ✅.
5. Porque el historial y el panel ya se pintaron antes con datos antiguos; sin él no aparecería nada nuevo en esa pasada ✅.
6. Un mensaje de error como si fuera la respuesta del asistente, sin fragmentos: `query_rag` captura la excepción ✅.
7. Porque la barra lateral se ejecuta antes de pintar el historial, que ya se ve vacío ✅.
8. Todo `app.py`; la lógica está en `rag_system.py`.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Rerun | Streamlit reejecuta todo el script en cada interacción |
| `session_state` | Memoria del historial |
| Mensaje | `{role, content}` (+ `docs` en el asistente) |
| `query_rag` | Devuelve `(respuesta, docs)` ✅ |
| `st.rerun()` final | Necesario para pintar lo nuevo ✅ |
| Errores | `query_rag` los devuelve como texto de respuesta ✅ |

---

## Fuentes consultadas

- Material del tema: `app.py` (original), `1.5_app.py` (comentada) y `1.5_app_limpio.py` (limpia), en `Tema_3/Asistente_Legal_RAG/`.
- Documentación oficial de LangChain: no se citan páginas porque este archivo no usa clases de LangChain directamente.
- Comprobaciones ✅: Streamlit 1.65.0 con `streamlit.testing.v1.AppTest` y un `rag_system` simulado.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
