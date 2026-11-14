# 2.7 Plantillas por rol: `SystemMessagePromptTemplate` y `HumanMessagePromptTemplate`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `2.7_rol_prompt_templates.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice. ✅ *(ejecutado)* = comportamiento observado al ejecutar el código con `langchain-core` 1.7.7; este archivo no llama a ningún modelo, así que su salida es reproducible.

---

## Antes de empezar

### Qué vas a aprender

1. Crear **sub-plantillas** especializadas por rol: una para el mensaje de sistema y otra para el del usuario.
2. Ensamblarlas con `ChatPromptTemplate.from_messages`.
3. Rellenar **todos** los huecos de ambas con una sola llamada a `format_messages`.

### Qué necesitas

- Haber visto `ChatPromptTemplate` y `format_messages` (documento 2.3).
- No necesitas clave de API: el archivo **no llama a ningún modelo**.

### Mapa del tema

```text
SystemMessagePromptTemplate  ─┐
  "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
                              ├─►  ChatPromptTemplate.from_messages([...])  ─►  format_messages(5 valores)
HumanMessagePromptTemplate   ─┘                                                   │
  "Mi pregunta sobre {tema} es: {pregunta}"                                       ▼
                                                              [SystemMessage, HumanMessage]
```

### Glosario

| Término | Significado |
|---|---|
| **`SystemMessagePromptTemplate`** | Molde que, al formatearse, se convierte en un `SystemMessage`; ideal para inyectar la personalidad del bot de forma dinámica. 📘 |
| **`HumanMessagePromptTemplate`** | Molde que, al formatearse, se convierte en un `HumanMessage`, el turno del usuario. 📘 |
| **`.from_template(texto)`** | Crea cada sub-plantilla a partir de un texto con huecos `{variable}`. |
| **Sub-plantilla** | Plantilla de un único mensaje que se usa como pieza de un `ChatPromptTemplate`. El archivo la llama el "Lego" del prompt. 📘 |
| **`format_messages(**kwargs)`** | Rellena todos los huecos y devuelve la lista de mensajes. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-core
```

Ejecución (no necesita clave de API):

```bash
python 2.7_rol_prompt_templates.py
```

---

### Lección 1: importaciones (sección 1 del archivo)

```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
```

Además del contenedor principal (`ChatPromptTemplate`), se importan las dos sub-plantillas especializadas.

---

### Lección 2: las sub-plantillas (sección 2 del archivo)

```python
plantilla_sistema = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
)

plantilla_humano = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)
```

| Sub-plantilla | Variables | Se convertirá en |
|---|---|---|
| `plantilla_sistema` | `{rol}`, `{especialidad}`, `{tono}` | `SystemMessage` |
| `plantilla_humano` | `{tema}`, `{pregunta}` | `HumanMessage` |

**Punto clave.** Cada sub-plantilla se crea con `.from_template(...)` a partir de un texto con huecos y representa **un solo mensaje** de un rol concreto.

---

### Lección 3: ensamblar el prompt (sección 3 del archivo)

```python
chat_prompt = ChatPromptTemplate.from_messages([
    plantilla_sistema,
    plantilla_humano
])
```

`from_messages` une las piezas **en el orden exacto** en que el modelo las recibirá: primero el contexto (sistema) y después la acción (humano). 📘

Aquí se pasan **objetos sub-plantilla** en lugar de tuplas `(rol, texto)`. El archivo añade que las plantillas podrían estar en archivos separados y cargarse dinámicamente, pero aquí se definen en el propio script. 📘

---

### Lección 4: rellenar y ver el resultado (secciones 4 y 5 del archivo)

```python
mensajes = chat_prompt.format_messages(
    rol="nutricionista",
    especialidad="dietas veganas",
    tono="profesional pero accesible",
    tema="proteínas vegetales",
    pregunta="¿Cuáles son las mejores fuentes de proteína vegana para un atleta profesional?"
)

for m in mensajes:
    print(m.content)
```

**Qué ocurre:** `format_messages` recorre ambas plantillas, detecta los **5 huecos** (`rol`, `especialidad`, `tono`, `tema`, `pregunta`) y los sustituye por los valores del diccionario. El resultado es una lista de objetos `BaseMessage` lista para el modelo. 📘

**Salida real del script** ✅:

```text
Eres un nutricionista especializado en dietas veganas. Responde de manera profesional pero accesible
Mi pregunta sobre proteínas vegetales es: ¿Cuáles son las mejores fuentes de proteína vegana para un atleta profesional?
```

El archivo indica que la lista tiene **dos** mensajes: un `SystemMessage` (el primero) y un `HumanMessage` (el segundo). 📘 Dos líneas impresas es coherente con ello; el script no imprime los tipos.

**Puntos clave.**

- Los cinco valores se pasan juntos en **una sola** llamada.
- Los huecos de ambas sub-plantillas comparten el mismo conjunto de valores.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo creo una plantilla de mensaje de sistema con variables?

```python
plantilla_sistema = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
)
```

### ¿Cómo creo la plantilla del turno del usuario?

```python
plantilla_humano = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)
```

### ¿Cómo las combino?

```python
chat_prompt = ChatPromptTemplate.from_messages([plantilla_sistema, plantilla_humano])
```

### ¿Cómo cambio el rol o el tono sin tocar la plantilla?

Cambia los valores que pasas a `format_messages`: por ejemplo, otro `rol` o `tono`.

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Error por variable faltante al formatear | No se pasó alguno de los 5 valores | Pasar `rol`, `especialidad`, `tono`, `tema` y `pregunta` ⚠️ *(comportamiento análogo al de `PromptTemplate`; no se ejecutó este caso)* |
| El orden de los mensajes no es el esperado | Orden de la lista de `from_messages` | Colocar primero la plantilla de sistema y después la del usuario 📘 |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate` | `langchain-core` | Contenedor y sub-plantillas |

### Sub-plantillas del archivo

| Nombre | Clase | Texto | Variables |
|---|---|---|---|
| `plantilla_sistema` | `SystemMessagePromptTemplate` | `Eres un {rol} especializado en {especialidad}. Responde de manera {tono}` | `rol`, `especialidad`, `tono` |
| `plantilla_humano` | `HumanMessagePromptTemplate` | `Mi pregunta sobre {tema} es: {pregunta}` | `tema`, `pregunta` |

### Valores usados

| Variable | Valor |
|---|---|
| `rol` | `nutricionista` |
| `especialidad` | `dietas veganas` |
| `tono` | `profesional pero accesible` |
| `tema` | `proteínas vegetales` |
| `pregunta` | `¿Cuáles son las mejores fuentes de proteína vegana para un atleta profesional?` |

### Qué muestra la referencia oficial

| Clase | Información de la página |
|---|---|
| `SystemMessagePromptTemplate` | Solo la firma; base `_StringImageMessagePromptTemplate`; sin descripción |
| `HumanMessagePromptTemplate` | Solo la firma; base `_StringImageMessagePromptTemplate`; sin descripción |
| `ChatPromptTemplate.from_messages` | Los ejemplos oficiales usan tuplas `(rol, contenido)`; la página no describe el método |

⚠️ La documentación consultada no describe `.from_template(...)` de estas sub-plantillas ni confirma que `from_messages` acepte objetos sub-plantilla; el código del archivo sí funciona al ejecutarlo ✅.

### Métodos usados

| Método | Devuelve |
|---|---|
| `SystemMessagePromptTemplate.from_template(texto)` | Sub-plantilla de mensaje de sistema |
| `HumanMessagePromptTemplate.from_template(texto)` | Sub-plantilla de mensaje de usuario |
| `ChatPromptTemplate.from_messages([...])` | Plantilla de chat |
| `chat_prompt.format_messages(**kwargs)` | Lista de mensajes (`list[BaseMessage]`) |

---

## Explicación

### ¿Por qué usar sub-plantillas por rol?

Porque cada pieza representa **un mensaje con un rol concreto** y se puede definir, reutilizar y combinar por separado, como piezas de Lego. El archivo destaca que `SystemMessagePromptTemplate` es ideal para inyectar la personalidad del bot de forma dinámica. 📘

### ¿Por qué se pasan todos los valores en una sola llamada?

Porque `format_messages` recorre todas las plantillas del `ChatPromptTemplate` y rellena todos los huecos con el mismo diccionario de valores. 📘

### ¿Cuándo usar este enfoque?

| Situación | Enfoque |
|---|---|
| Prompt corto y sencillo | Tuplas `(rol, texto)` en `from_messages` (documento 2.3) |
| Quieres definir y reutilizar cada mensaje por separado | Sub-plantillas por rol |
| Personalidad configurable | `SystemMessagePromptTemplate` con variables |

### Lo que este documento no cubre

- Cargar plantillas desde archivos externos (el archivo solo lo menciona).
- La llamada a un modelo: el archivo no la hace.
- `AIMessagePromptTemplate` u otras sub-plantillas: no aparecen en el archivo.
- Una descripción oficial de estas clases (la referencia solo trae firmas).

### Observaciones sobre el archivo

- La plantilla de sistema no termina con punto tras `{tono}`: es el texto tal cual del archivo.
- El archivo dice "detecta los 5 huecos": coincide con las cinco variables (`rol`, `especialidad`, `tono`, `tema`, `pregunta`).

---

## Autoevaluación

1. ¿Qué hace `SystemMessagePromptTemplate.from_template(...)`?
2. ¿En qué tipo de mensaje se convierte `HumanMessagePromptTemplate` al formatearse?
3. ¿Qué se pasa a `from_messages` en este archivo en lugar de tuplas?
4. ¿Cuántos huecos tiene el prompt completo y cuáles son?
5. ¿Cuántas llamadas hacen falta para rellenar ambas sub-plantillas?
6. ¿Cuántos mensajes devuelve `format_messages` y en qué orden?
7. ¿Hace falta una clave de API para ejecutar este archivo?

### Respuestas

1. Crea una sub-plantilla de mensaje de sistema a partir de un texto con huecos `{variable}`.
2. En un `HumanMessage`.
3. Objetos sub-plantilla: `plantilla_sistema` y `plantilla_humano`.
4. Cinco: `rol`, `especialidad`, `tono`, `tema` y `pregunta`.
5. Una: `format_messages` rellena todos los huecos de una vez.
6. Dos: primero el de sistema y después el del usuario.
7. No: no llama a ningún modelo.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `SystemMessagePromptTemplate` | Plantilla de un mensaje de sistema |
| `HumanMessagePromptTemplate` | Plantilla de un mensaje de usuario |
| `.from_template(texto)` | Crea cada sub-plantilla |
| `from_messages([...])` | Las ensambla en orden |
| `format_messages(...)` | Rellena todos los huecos y devuelve los mensajes |

---

## Fuentes consultadas

- Referencia de `SystemMessagePromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/chat/SystemMessagePromptTemplate>
- Referencia de `HumanMessagePromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/chat/HumanMessagePromptTemplate>
- Referencia de `ChatPromptTemplate`: <https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate>
- Material del tema: `2.7_rol_prompt_templates.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
