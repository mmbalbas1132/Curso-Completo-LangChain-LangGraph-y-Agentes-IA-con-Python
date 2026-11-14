# 1.1 Primera llamada a un modelo de chat con `ChatOpenAI`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `hello_world.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice; ✅ *(ejecutado)* = comportamiento observado al ejecutar código con `langchain-core` 1.7.7. Los scripts de este tema necesitan claves de API, por lo que **no se ejecutaron con modelos reales**: el mecanismo (plantillas, cadenas, `AIMessage`) se comprobó con un modelo simulado (`FakeListChatModel`) y la respuesta de un modelo real nunca se reproduce.

---

## Antes de empezar

### Qué vas a aprender

1. Guardar una clave de API en un archivo `.env` y cargarla.
2. Crear un modelo de chat de OpenAI con `ChatOpenAI`.
3. Enviarle una pregunta con `.invoke()`.
4. Entender qué devuelve (`AIMessage`) y cómo leer su texto (`.content`).

### Qué necesitas

- Python básico (variables, `import`, `print`).
- Una clave de API de OpenAI.
- Saber ejecutar un script desde la terminal.

### Mapa del tema

```text
.env ──load_dotenv()──► variable de entorno OPENAI_API_KEY
                               │
ChatOpenAI(model_name, temperature) ──► llm
                               │
llm.invoke("pregunta") ──► AIMessage ──► .content ──► texto
```

### Glosario

| Término | Significado |
|---|---|
| **Modelo de chat** | Objeto de LangChain que envía mensajes a una IA y recibe su respuesta. Aquí, `ChatOpenAI`. |
| **Clave de API** | Credencial de tu cuenta ante el proveedor; no debe escribirse en el código. |
| **Variable de entorno** | Valor del sistema accesible por nombre (aquí `OPENAI_API_KEY`). |
| **`.env` / `load_dotenv()`** | Archivo con pares `NOMBRE=valor` y la función que los carga en el entorno. |
| **`temperature`** | Parámetro de aleatoriedad. Según el archivo, va de `0.0` (muy lógico y determinista) a `1.0` (muy creativo). 📘 |
| **`.invoke()`** | Envía la entrada y espera la respuesta completa. |
| **`AIMessage`** | Objeto que devuelve el modelo; el texto está en `.content`. |

---

## Tutorial

### Preparación

```bash
pip install -U langchain-openai python-dotenv
```

Archivo `.env` en la carpeta del script:

```bash
OPENAI_API_KEY=tu_clave
```

Ejecución:

```bash
python hello_world.py
```

---

### Lección 1: importaciones y entorno (secciones 1 y 2 del archivo)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
```

| Línea | Qué hace y por qué |
|---|---|
| `from dotenv import load_dotenv` | Importa la función que lee el `.env`. |
| `from langchain_openai import ChatOpenAI` | Importa la clase del modelo de chat de OpenAI. |
| `load_dotenv()` | Lee el `.env` de la carpeta actual y "mete" las claves en las variables de entorno. **Siempre antes de inicializar el modelo.** 📘 |

`ChatOpenAI` busca automáticamente `OPENAI_API_KEY`; la documentación oficial confirma esa variable. El archivo indica que se quitó `import os` porque no se usa. 📘

**Punto clave.** Las importaciones van al principio del archivo y `load_dotenv()` antes de crear el modelo.

---

### Lección 2: crear el modelo (sección 3 del archivo)

```python
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
```

| Parámetro | Valor | Significado según el archivo 📘 |
|---|---|---|
| `model_name` | `"gpt-4o-mini"` | Modelo "rápido, barato y potente" |
| `temperature` | `0.7` | Más natural; para datos exactos (como un año) se usaría `0.0` |

🔀 *(matiz)* La documentación oficial de `ChatOpenAI` usa el parámetro `model` en sus ejemplos; `model_name` solo aparece en su ejemplo de *fine-tuning*. El script usa `model_name`. Para código nuevo, sigue la forma oficial: `ChatOpenAI(model="gpt-4o-mini", temperature=0.7)`.

---

### Lección 3: preguntar y leer la respuesta (sección 4 del archivo)

```python
pregunta = "¿En qué año llegó el ser humano a la luna por primera vez?"
print("→ Pregunta:", pregunta)

respuesta = llm.invoke(pregunta)

print("→ Respuesta del modelo:", respuesta.content)
```

| Paso | Qué hace |
|---|---|
| `pregunta = "..."` | Texto que se enviará al modelo |
| `llm.invoke(pregunta)` | Método estándar de LangChain para enviar algo al modelo y **esperar** a que termine (llamada síncrona y bloqueante). 📘 |
| `respuesta.content` | `invoke` devuelve un `AIMessage`, **no** un texto plano; `.content` contiene el texto |

**Comprobado** ✅: con un modelo simulado, `invoke("texto suelto")` devuelve un `AIMessage`.

**La respuesta no se reproduce aquí:** la genera el modelo y varía.

**Contraste oficial.** La documentación indica que `invoke()` es "la forma más directa de llamar a un modelo", acepta un mensaje o una lista de mensajes y devuelve un único `AIMessage` al terminar de generar. Sus ejemplos leen el texto con `.text`; este script usa `.content`.

**Puntos clave.**

- `invoke` devuelve un objeto, no un `str`.
- El texto se lee con `.content`.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo evito escribir la clave en el código?

Guárdala en `.env` y llama a `load_dotenv()` antes de crear el modelo.

### ¿Cómo hago que el modelo sea más o menos creativo?

Cambia `temperature`: valores bajos para respuestas deterministas, altos para más creatividad. 📘

### ¿Cómo obtengo solo el texto de la respuesta?

```python
respuesta.content
```

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Error de autenticación | `load_dotenv()` se ejecutó después de crear el modelo, o falta `OPENAI_API_KEY` | Cargar el entorno antes y comprobar el nombre de la variable |
| Se imprime un objeto en lugar de texto | Se imprimió `respuesta` en vez de `respuesta.content` | Usar `.content` |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `from dotenv import load_dotenv` | `python-dotenv` | Cargar `.env` |
| `from langchain_openai import ChatOpenAI` | `langchain-openai` | Modelo de chat |

### Variables de entorno

| Variable | Clase | Confirmada en documentación oficial |
|---|---|---|
| `OPENAI_API_KEY` | `ChatOpenAI` | Sí |

### Parámetros de `ChatOpenAI` usados

| Parámetro | Valor | Notas |
|---|---|---|
| `model_name` | `"gpt-4o-mini"` | La documentación oficial usa `model` 🔀 |
| `temperature` | `0.7` | La documentación oficial lo lista sin describir su comportamiento |

### Métodos y atributos

| Elemento | Devuelve | Confirmado en documentación oficial |
|---|---|---|
| `.invoke(entrada)` | `AIMessage` | Sí |
| `.content` | Texto de la respuesta | Se usa en el archivo; la documentación oficial emplea también `.text` |

---

## Explicación

### ¿Por qué el modelo devuelve un `AIMessage`?

Los modelos de chat trabajan con mensajes tipados por rol. La respuesta es un `AIMessage`, que además del texto puede llevar otra información; por eso se accede al texto con `.content`.

### ¿Por qué `load_dotenv()` va antes del modelo?

`ChatOpenAI` busca la clave en el entorno; si aún no está cargada, no la encuentra. 📘

### ¿Cuándo usar `invoke`?

Cuando quieres la respuesta completa de una vez y puedes esperar. Para ir recibiendo texto mientras se genera, existe `.stream()` (documento 1.5).

### Lo que este documento no cubre

- Plantillas de prompt y cadenas (documento 1.3).
- Otros proveedores (documento 1.2).
- La respuesta real del modelo: necesita una clave de API.

### Observaciones sobre el archivo

- Los comentarios dicen "Aquí pusiste 0.7": se dirigen a quien escribió el código.
- El comentario sobre `import os` indica que se eliminó por no usarse.

---

## Autoevaluación

1. ¿Para qué sirve `load_dotenv()` y por qué va antes de crear el modelo?
2. ¿Qué variable de entorno busca `ChatOpenAI`?
3. ¿Qué controla `temperature`?
4. ¿Qué devuelve `llm.invoke(...)`?
5. ¿Cómo se obtiene el texto de la respuesta?
6. ¿Qué significa que `.invoke()` sea "bloqueante"?

### Respuestas

1. Carga las claves del `.env` en el entorno; el modelo las lee al crearse.
2. `OPENAI_API_KEY`.
3. La aleatoriedad o creatividad de la respuesta.
4. Un `AIMessage`.
5. Con `.content`.
6. Que el programa espera a que el modelo termine antes de continuar.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `.env` + `load_dotenv()` | Guardan y cargan la clave |
| `ChatOpenAI` | Modelo de chat de OpenAI |
| `temperature` | Creatividad: baja = determinista |
| `.invoke()` | Envía y espera la respuesta completa |
| `AIMessage` / `.content` | Objeto de respuesta y su texto |

---

## Fuentes consultadas

- Documentación oficial de `ChatOpenAI`: <https://python.langchain.com/docs/integrations/chat/openai/> (redirige a <https://docs.langchain.com/oss/python/integrations/chat/openai>)
- Documentación oficial de modelos (`invoke`): <https://docs.langchain.com/oss/python/langchain/models>
- Material del tema: `hello_world.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
