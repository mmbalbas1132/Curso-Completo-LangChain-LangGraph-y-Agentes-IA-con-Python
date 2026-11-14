# 1.2 Cambiar de proveedor: `ChatDeepSeek`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `Usando_DeepSeek.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación oficial difiere del archivo o el archivo se contradice; ✅ *(ejecutado)* = comportamiento observado al ejecutar código con `langchain-core` 1.7.7. Los scripts de este tema necesitan claves de API, por lo que **no se ejecutaron con modelos reales**: el mecanismo (plantillas, cadenas, `AIMessage`) se comprobó con un modelo simulado (`FakeListChatModel`) y la respuesta de un modelo real nunca se reproduce.

---

## Antes de empezar

### Qué vas a aprender

1. Usar `ChatDeepSeek` en lugar de `ChatOpenAI`.
2. Comprobar qué cambia (y qué no) al cambiar de proveedor.
3. Distinguir los dos modelos de DeepSeek que cita el archivo.

### Qué necesitas

- Haber visto el documento 1.1 (`invoke`, `AIMessage`, `.env`).
- Una clave de API de DeepSeek.

### Mapa del tema

```text
1.1  ChatOpenAI(model_name=...)   ──┐
                                    ├─►  misma interfaz: .invoke() → AIMessage → .content
1.2  ChatDeepSeek(model=...)      ──┘
```

### Glosario

| Término | Significado |
|---|---|
| **`ChatDeepSeek`** | Clase de LangChain para el modelo de chat de DeepSeek (paquete `langchain_deepseek`). |
| **`deepseek-chat`** | Modelo general, "rápido y versátil". 📘 |
| **`deepseek-reasoner`** | Modelo para tareas de razonamiento complejo. 📘 |
| **`DEEPSEEK_API_KEY`** | Variable de entorno con la clave. |

---

## Tutorial

### Preparación

```bash
pip install -qU langchain-deepseek python-dotenv
```

Archivo `.env`:

```bash
DEEPSEEK_API_KEY=tu_clave
```

Ejecución:

```bash
python Usando_DeepSeek.py
```

---

### Lección 1: el script completo (secciones 1 a 4 del archivo)

```python
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

pregunta = "¿En qué año se enunció la teoría de la relatividad?"
print("→ Pregunta:", pregunta)

respuesta = llm.invoke(pregunta)

print("→ Respuesta del modelo (DeepSeek):", respuesta.content)
```

| Línea | Qué hace y por qué |
|---|---|
| `from langchain_deepseek import ChatDeepSeek` | Importa la clase del modelo de chat de DeepSeek. |
| `load_dotenv()` | Carga el `.env`; asegúrate de tener `DEEPSEEK_API_KEY`. |
| `ChatDeepSeek(model="deepseek-chat", temperature=0.7)` | Crea el modelo; `temperature` "igual que antes" controla la creatividad (`0.0` = determinista). 📘 |
| `llm.invoke(pregunta)` | Envía la pregunta y espera la respuesta completa. |
| `respuesta.content` | El objeto devuelto es un `AIMessage`, "igual que con OpenAI"; `.content` extrae el texto. 📘 |

La respuesta no se reproduce: la genera el modelo y varía.

### Lección 2: qué cambia respecto al documento 1.1

| Elemento | 1.1 (OpenAI) | 1.2 (DeepSeek) |
|---|---|---|
| Paquete | `langchain_openai` | `langchain_deepseek` |
| Clase | `ChatOpenAI` | `ChatDeepSeek` |
| Variable de entorno | `OPENAI_API_KEY` | `DEEPSEEK_API_KEY` |
| Parámetro del modelo | `model_name="gpt-4o-mini"` | `model="deepseek-chat"` |
| `.invoke()` y `.content` | igual | igual |

**Punto clave.** La interfaz es la misma; solo cambian la clase, el modelo y la variable de entorno.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo cambio de OpenAI a DeepSeek?

Sustituye la importación y la clase, define `DEEPSEEK_API_KEY` y usa `model="deepseek-chat"`.

### ¿Cómo elijo entre los dos modelos de DeepSeek?

`deepseek-chat` para uso general; `deepseek-reasoner` para razonamiento complejo. 📘

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Falla la autenticación | Falta `DEEPSEEK_API_KEY` o `load_dotenv()` va después del modelo | Definir la variable y cargar el entorno antes |
| `ModuleNotFoundError` con `langchain_deepseek` | No está instalado el paquete | `pip install -qU langchain-deepseek` (comando de la documentación oficial) |

---

## Referencia

### Importaciones

| Importación | Paquete | Uso |
|---|---|---|
| `from dotenv import load_dotenv` | `python-dotenv` | Cargar `.env` |
| `from langchain_deepseek import ChatDeepSeek` | `langchain-deepseek` | Modelo de chat |

### `ChatDeepSeek`: parámetros

| Parámetro | Valor en el archivo | Documentación oficial |
|---|---|---|
| `model` | `"deepseek-chat"` | Sí |
| `temperature` | `0.7` | Aparece (`0` en el ejemplo oficial) |
| `max_tokens`, `timeout`, `max_retries` | No se usan | Aparecen en el ejemplo oficial |

### Modelos de DeepSeek

| Modelo | Documentación oficial | Archivo |
|---|---|---|
| `deepseek-chat` | DeepSeek-V3; admite *tool calling* y salida estructurada | Rápido y versátil |
| `deepseek-reasoner` | DeepSeek-R1; no admite *tool calling* ni salida estructurada | Razonamiento complejo |

### Variable de entorno

| Variable | Confirmada en documentación oficial |
|---|---|
| `DEEPSEEK_API_KEY` | Sí |

---

## Explicación

### ¿Por qué cambiar de proveedor es tan sencillo?

Porque `ChatDeepSeek` expone la misma interfaz que `ChatOpenAI` (`invoke`, `AIMessage`, `.content`). La documentación oficial de `ChatDeepSeek` muestra el mismo patrón de `invoke` con mensajes.

### ¿Cuándo usar cada modelo?

| Situación | Modelo |
|---|---|
| Consulta general rápida | `deepseek-chat` |
| Razonamiento complejo | `deepseek-reasoner` |

### Lo que este documento no cubre

- Si `deepseek-reasoner` admite `temperature`: el archivo no lo dice y la documentación oficial consultada tampoco lo indica. ⚠️
- La respuesta real del modelo.

### Observaciones sobre el archivo

- El comentario sobre `deepseek-reasoner` es solo una mención; el script usa `deepseek-chat`.

---

## Autoevaluación

1. ¿Qué clase y paquete se usan para DeepSeek?
2. ¿Qué variable de entorno hace falta?
3. ¿Qué cambia respecto a usar OpenAI y qué se mantiene?
4. ¿Qué devuelve `.invoke()` en DeepSeek?
5. ¿Para qué se usaría `deepseek-reasoner` según el archivo?

### Respuestas

1. `ChatDeepSeek`, del paquete `langchain_deepseek`.
2. `DEEPSEEK_API_KEY`.
3. Cambian paquete, clase, variable de entorno y parámetro del modelo; se mantienen `.invoke()` y `.content`.
4. Un `AIMessage`.
5. Para tareas de razonamiento complejo.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `ChatDeepSeek` | Modelo de chat de DeepSeek |
| `DEEPSEEK_API_KEY` | Variable con la clave |
| `deepseek-chat` / `deepseek-reasoner` | General y rápido / razonamiento complejo |
| Misma interfaz | `invoke` → `AIMessage` → `.content` |

---

## Fuentes consultadas

- Documentación oficial de `ChatDeepSeek`: <https://python.langchain.com/docs/integrations/chat/deepseek/> (redirige a <https://docs.langchain.com/oss/python/integrations/chat/deepseek>)
- Material del tema: `Usando_DeepSeek.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
