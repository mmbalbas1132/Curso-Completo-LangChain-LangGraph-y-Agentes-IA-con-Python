# 3.4 Plantillas de prompt del asistente legal (`prompts.py`)

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `prompts.py` (original, sin modificar) y sus versiones `3.4_prompts.py` (comentada) y `3.4_prompts_limpio.py` (limpia), con código idéntico (✅ AST). Marcas: 📘 *(material del curso)*; ⚠️ *(no contrastado)*; 🔀 *(discrepancia)*; ✅ *(comprobado)* con `langchain-core` 1.8.4: se cargó cada plantilla con `PromptTemplate.from_template` y se leyeron sus variables. El archivo solo contiene texto; **no se envió ningún prompt a un modelo**, así que no se describe la calidad de las respuestas.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es una plantilla de prompt con variables `{...}`.
2. Cuáles son las cuatro plantillas del archivo y cuáles se usan realmente.
3. Cómo encajan `{context}` y `{question}` con la cadena RAG.

### Qué necesitas

- Documento 1.3 (`PromptTemplate`) y 3.3/3.6 como contexto.

### Mapa del tema

```text
RAG_TEMPLATE ({context}, {question}) ──► cadena RAG ──► respuesta final        (usada)
MULTI_QUERY_PROMPT ({question}) ──────► MultiQueryRetriever ──► 3 variantes    (usada)
RELEVANCE_PROMPT ({document}, {question})                                      (sin usar)
ENTITY_EXTRACTION_PROMPT ({text})                                              (sin usar)
```

### Glosario

| Término | Significado |
|---|---|
| **Plantilla (`PromptTemplate`)** | Texto con huecos `{variable}` que se rellenan al ejecutarse. |
| **Variable** | Nombre entre llaves que debe recibir un valor. |
| **Contexto (`{context}`)** | Fragmentos recuperados de los contratos. |
| **Prompt de sistema/rol** | Primera frase que define quién es el asistente. |

---

## Tutorial

### Lección 1: `RAG_TEMPLATE` (líneas 1 a 18)

Plantilla principal. Estructura (resumen; texto completo en `prompts.py`):

```text
Eres un asistente legal especializado en contratos de arrendamiento.
Basándote ÚNICAMENTE en los siguientes fragmentos... responde a la pregunta
FRAGMENTOS DE CONTRATOS: {context}
PREGUNTA: {question}
INSTRUCCIONES: (6 viñetas)
RESPUESTA:
```

| Parte | Función |
|---|---|
| Primera frase | Define el rol |
| "ÚNICAMENTE en los siguientes fragmentos" | Limita al contexto recuperado (no garantiza por sí sola que el modelo lo cumpla) |
| `{context}` y `{question}` | Variables ✅ (comprobado con `PromptTemplate.from_template`) |
| Instrucciones | Citar textualmente, incluir nombres/importes/fechas, avisar si falta información, aclarar a qué contrato o persona se refiere |
| `RESPUESTA:` | Pista para que el modelo continúe |

Se usa en `rag_system.py`: `PromptTemplate.from_template(RAG_TEMPLATE)`. ✅

### Lección 2: `MULTI_QUERY_PROMPT` (líneas 20 a 32)

Pide al modelo **exactamente 3 versiones alternativas** de la consulta, una por línea, sin numeración. Tiene una variable: `{question}` ✅. Se pasa como `prompt=` a `MultiQueryRetriever.from_llm(...)`. La firma oficial permite sustituir el prompt por defecto, que también pide variantes "separadas por saltos de línea" ✅.

Sus criterios (variantes de nombres, sinónimos legales, formulaciones, ubicaciones) son decisiones del autor del curso. 📘

### Lección 3: `RELEVANCE_PROMPT` (líneas 34 a 42)

Pide decidir si un fragmento es relevante para una consulta, respondiendo "SÍ" o "NO" con una breve justificación. Variables: `{document}` y `{question}` ✅.

⚠️ **No se usa en este proyecto** (✅ ninguna referencia en `app.py` ni `rag_system.py`).

### Lección 4: `ENTITY_EXTRACTION_PROMPT` (líneas 44 a 64)

Pide extraer personas, direcciones, importes, fechas, duración y tipo de propiedad, con un formato de respuesta fijo (`PERSONAS:`, `DIRECCIONES:`...). Variable: `{text}` ✅. **Tampoco se usa en este proyecto.** ⚠️

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo uso una plantilla?

```python
prompt = PromptTemplate.from_template(RAG_TEMPLATE)
```

(así lo hace `rag_system.py`).

### ¿Cómo sé qué variables pide cada plantilla?

`PromptTemplate.from_template(texto).input_variables` ✅:

| Plantilla | Variables |
|---|---|
| `RAG_TEMPLATE` | `context`, `question` |
| `MULTI_QUERY_PROMPT` | `question` |
| `RELEVANCE_PROMPT` | `document`, `question` |
| `ENTITY_EXTRACTION_PROMPT` | `text` |

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Error por variable que falta | La cadena no entrega todas las variables | Que los nombres coincidan con los de la plantilla (`context`, `question`) |
| Aparece una llave `{` literal | Texto con llaves en la plantilla | Duplicarlas (`{{ }}`) ⚠️ no contrastado |

---

## Referencia

| Constante | Variables | Usada en `rag_system.py` |
|---|---|---|
| `RAG_TEMPLATE` | `context`, `question` | Sí |
| `MULTI_QUERY_PROMPT` | `question` | Sí |
| `RELEVANCE_PROMPT` | `document`, `question` | No |
| `ENTITY_EXTRACTION_PROMPT` | `text` | No |

---

## Explicación

### ¿Por qué prompts en un archivo aparte?

Permite revisar y cambiar el texto de los prompts sin tocar la lógica. 📘

### ¿Por qué pedir "ÚNICAMENTE" los fragmentos?

Para reducir respuestas que no se apoyen en los contratos. 📘 No se midió su efecto.

### Lo que este documento no cubre

- La calidad de las respuestas con estos prompts (no se ejecutó ningún modelo).
- Cómo se usarían `RELEVANCE_PROMPT` y `ENTITY_EXTRACTION_PROMPT`.

### Observaciones sobre el archivo

- Dos de las cuatro plantillas no se usan en el proyecto. ✅
- Todo el texto está en español y especializado en arrendamientos.

---

## Autoevaluación

1. ¿Qué variables tiene `RAG_TEMPLATE`?
2. ¿Qué pide `MULTI_QUERY_PROMPT` y cuántas versiones?
3. ¿Qué plantillas se usan en `rag_system.py`?
4. ¿Cómo averiguas las variables de una plantilla?
5. ¿Qué formato de respuesta fija `ENTITY_EXTRACTION_PROMPT`?
6. ¿Por qué los nombres `context` y `question` importan?

### Respuestas

1. `context` y `question` ✅.
2. 3 versiones alternativas de la consulta, una por línea, sin numeración.
3. `RAG_TEMPLATE` y `MULTI_QUERY_PROMPT` ✅.
4. Con `PromptTemplate.from_template(texto).input_variables` ✅.
5. Líneas `PERSONAS:`, `DIRECCIONES:`, `IMPORTES:`, `FECHAS:`, `DURACIÓN:`, `TIPO:`.
6. Porque la cadena debe entregar esas claves exactas.

---

## Resumen

| Concepto | En una línea |
|---|---|
| `RAG_TEMPLATE` | Respuesta basada solo en fragmentos |
| `MULTI_QUERY_PROMPT` | 3 variantes de la consulta |
| Sin usar | `RELEVANCE_PROMPT` y `ENTITY_EXTRACTION_PROMPT` |

---

## Fuentes consultadas

- Referencia de `MultiQueryRetriever.from_llm`: <https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever/from_llm>
- Material del tema: `prompts.py`, `3.4_prompts.py`, `3.4_prompts_limpio.py` (en `Tema_3/asistente_legal_RAG/`) y `rag_system.py` (solo para ver cuáles se usan).

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
