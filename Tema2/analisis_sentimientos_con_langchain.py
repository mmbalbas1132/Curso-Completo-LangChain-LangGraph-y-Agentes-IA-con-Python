"""
================================================================================
MINI-PROYECTO: ANÁLISIS DE SENTIMIENTOS CON LANGCHAIN (LCEL) — PARTE 2
================================================================================
Material de estudio — 
    • Encabezado conceptual ("¿por qué?").
    • Código comentado ("¿cómo?").
    • Notas "🔎 Para profundizar" con matices y errores comunes.

DIFERENCIA CLAVE RESPECTO A LA PARTE 1
--------------------------------------
    Parte 1: `chain.invoke(texto)`       → 1 entrada, 1 salida.
    Parte 2: `chain.batch([t1, t2, t3])` → N entradas, N salidas CONCURRENTES.

El cambio es mínimo en el código, pero ENORME en implicaciones:

    ❌ Bucle con .invoke()          ✅ .batch()
    ─────────────────────────       ──────────────────────────────────
    Secuencial (una tras otra)      Concurrente (paraleliza automáticamente)
    3 llamadas × latencia           Latencia ≈ la más lenta (con límites)
    Sin control de concurrencia     `max_concurrency` configurable
    Errores rompen el bucle         `return_exceptions` los encapsula
    Código imperativo (for)         Declarativo (una línea)

FILOSOFÍA
---------
LCEL no solo compone: también expone los mismos métodos sobre CUALQUIER
cadena. Aprendes `.invoke` una vez → `.batch`, `.stream`, `.ainvoke`,
`.abatch`, `.astream` funcionan igual en todas las cadenas. Ese es el
"contrato Runnable" y es lo que hace LangChain tan componible.

ARQUITECTURA (idéntica a la Parte 1)
------------------------------------
    texto_entrada
        │
        ▼
    [preprocessor]              ← limpieza / truncado
        │
        ▼
    [RunnableParallel]          ← resumen + sentimiento, CONCURRENTES
        ├── resumen
        └── sentimiento_data
        │
        ▼
    [merger]                    ← aplana el resultado
        │
        ▼
    dict final {resumen, sentimiento, razon}

    ⚡ Y AHORA, además: `chain.batch([...])` ejecuta N textos en paralelo.

NOVEDAD RESPECTO A LA PARTE 1 (además de .batch)
------------------------------------------------
    🆕 Carga de secretos vía `python-dotenv` (load_dotenv).
    🆕 Verificación "fail-fast" de OPENAI_API_KEY al inicio.
================================================================================
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
# ¿POR QUÉ LAS MISMAS QUE LA PARTE 1 + dotenv/os?
#   Porque `batch()` no necesita imports nuevos (es un MÉTODO del Runnable).
#   Pero para leer la API key desde .env necesitamos `os` y `load_dotenv`.
#
# ORDEN: stdlib → terceros → locales (convención PEP 8).
# ------------------------------------------------------------------------------
import os                          # 🆕 Acceso a variables de entorno
import json                        # Parsear la respuesta JSON del LLM

from dotenv import load_dotenv     # 🆕 Cargar .env → os.environ

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

# 🔎 Para profundizar
#   • `batch()` internamente es async (usa `asyncio`). No requiere que tú
#     escribas `async def`: la librería lo gestiona. Por eso funciona en
#     scripts síncronos sin `asyncio.run()`.
#   • Si quieres control TOTAL (concurrencia, timeouts por item, reintentos
#     distintos por item), puedes usar `.abatch()` en tu propio `async def`.
#   • Existe también `.batch_as_completed()` (síncrono) que devuelve los
#     resultados SEGÚN VAN TERMINANDO, no en el orden original. Útil para
#     mostrar progreso en UIs.
#   • `python-dotenv` NO es de LangChain; es el estándar de facto en Python
#     para gestionar secretos en desarrollo.


# ==============================================================================
# 2. 🆕 CARGA DE VARIABLES DE ENTORNO
# ==============================================================================
# ¿POR QUÉ .env Y NO hardcodear la clave?
#   🔒 Seguridad:     el archivo .env va al .gitignore → nunca se filtra al repo.
#   🔁 Portabilidad:  cada entorno (dev, staging, prod) usa su propia clave.
#   🚀 Convención:    Docker, Kubernetes, Heroku, etc. inyectan secretos así.
#
# ¿QUÉ HACE load_dotenv() EXACTAMENTE?
#   1. Busca un archivo llamado ".env" en el cwd y en directorios padres.
#   2. Lee líneas tipo `CLAVE=valor` (ignora comentarios con `#`).
#   3. Escribe esas claves en `os.environ` SOLO si no existen ya.
#
# ¿POR QUÉ "SOLO SI NO EXISTEN YA" ES IMPORTANTE?
#   En producción, los secretos suelen venir inyectados por la plataforma
#   (K8s Secrets, AWS Parameter Store…). Si `load_dotenv()` sobreescribiera,
#   un .env obsoleto en el contenedor podría pisar la clave correcta.
#   Para forzar sobreescritura: load_dotenv(override=True)  ← rara vez.
# ------------------------------------------------------------------------------
load_dotenv()

# ¿POR QUÉ "FAIL-FAST"?
#   Si la clave no está, la alternativa es fallar 3 capas más abajo con un
#   error críptico ("AuthenticationError: Incorrect API key") que no dice
#   DÓNDE está el problema. Comprobarlo aquí convierte un bug sutil en un
#   mensaje accionable. Es el principio de "fail fast, fail loud".
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ Falta OPENAI_API_KEY. Defínela en tu .env o expórtala en el entorno."
    )

# 🔎 Para profundizar
#   • `os.getenv("X")` devuelve None si no existe; `os.environ["X"]` lanza
#     KeyError. Preferimos getenv para poder lanzar NUESTRO mensaje.
#   • Nunca imprimas la clave en logs, ni siquiera truncada: los logs pueden
#     persistir en sistemas externos (CloudWatch, Datadog…).
#   • Añade `.env` a `.gitignore` ANTES del primer commit. Si ya lo subiste,
#     rota la clave: el historial de Git conserva lo borrado.


# ==============================================================================
# 3. CONFIGURACIÓN DEL MODELO
# ==============================================================================
# temperature=0 → determinismo en clasificación (ver Parte 1 para detalle).
# ChatOpenAI lee OPENAI_API_KEY de os.environ automáticamente (vía load_dotenv),
# así que NO hace falta pasar api_key=... explícitamente.
# ------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 🔎 Para profundizar
#   • En un contexto de `.batch()`, cada llamada al LLM es INDEPENDIENTE.
#     `temperature=0` no "comparte" estado entre llamadas: solo garantiza
#     que la MISMA entrada produce la MISMA salida en cualquier momento.
#   • Si la API devuelve 429 (rate limit), `.batch()` lo gestiona con
#     reintentos controlados por `max_retries` del LLM. Puedes subirlo en
#     cargas grandes: `ChatOpenAI(..., max_retries=3)`.
#   • Pasar `api_key=os.getenv("OPENAI_API_KEY")` es EQUIVALENTE pero más
#     verboso. Preferimos el default (lee del entorno) para mantener el
#     código "agnóstico" al origen del secreto.


# ==============================================================================
# 4. PREPROCESADOR
# ==============================================================================
# ¿POR QUÉ NO LO CAMBIAMOS?
#   Porque en LCEL cada eslabón es independiente. `batch()` ejecuta la
#   cadena COMPLETA por cada elemento, así que el preprocesador sigue
#   siendo el primer paso para cada texto del lote. No hay que tocarlo.
#
# ¿QUÉ PASA CON EL TRUNCADO EN MODO BATCH?
#   Cada elemento se trunca por separado a 500 chars. No hay "batch-wide"
#   logic aquí. Si quisieras truncar el LOTE (p. ej. limitar el total de
#   tokens enviados), sería otro componente distinto ANTES de la cadena.
# ------------------------------------------------------------------------------
def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud."""
    return text.strip()[:500]

preprocessor = RunnableLambda(preprocess_text)


# ==============================================================================
# 5. GENERACIÓN DE RESUMEN
# ==============================================================================
# Prompt R-T-C: Rol + Tarea + Restricción. Ver Parte 1 para el "por qué".
# En modo batch, esta función se ejecuta UNA VEZ POR CADA texto del lote,
# concurrentemente con las demás (hasta el límite de concurrencia).
# ------------------------------------------------------------------------------
def generate_summary(text):
    """Genera un resumen conciso del texto."""
    prompt = f"Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content

summary_branch = RunnableLambda(generate_summary)

# 🔎 Para profundizar
#   • IMPORTANTE: dentro de `generate_summary` usamos `llm.invoke` (SÍNCRONO).
#     ¿Cómo es posible que `.batch()` sea concurrente entonces?
#     → RunnableLambda detecta si la función es sync o async. Si es sync,
#       la ejecuta en un THREAD POOL (por debajo). Así no bloqueas el event
#       loop principal, pero cada llamada SÍ bloquea SU hilo.
#     → Si quieres máxima eficiencia, usa `async def` + `await llm.ainvoke()`
#       y llama a `chain.abatch(...)`. Evitas el overhead de threads.
#   • En tu archivo original, `summary_branch` estaba escrito como
#     `summary_brach` (typo). Lo he corregido aquí; si quieres mantener el
#     nombre original por coherencia, renómbralo en las dos ocurrencias
#     (definición y uso dentro de RunnableParallel).


# ==============================================================================
# 6. ANÁLISIS DE SENTIMIENTO
# ==============================================================================
# Envolvemos el parseo en try/except para que UN texto problemático NO
# tumbe el lote entero. Esto es CRÍTICO en modo batch: si una excepción
# se propagara, `.batch()` abortaría todo el lote (a menos que uses
# `return_exceptions=True`, que veremos en la sección 8).
# ------------------------------------------------------------------------------
def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado."""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}

    Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback DETERMINISTA: el contrato se mantiene aunque el JSON falle.
        # Sin esto, un solo texto "raro" en el lote devolvería una excepción
        # que rompería la estructura del resultado batch (ver sección 8).
        return {"sentimiento": "neutro", "razon": "Error en análisis"}

sentiment_branch = RunnableLambda(analyze_sentiment)

# 🔎 Para profundizar
#   • Mismo patrón que la Parte 1: función PURA + RunnableLambda + fallback.
#   • ¿Qué pasaría si NO hubiera try/except?
#       `json.loads` lanzaría JSONDecodeError → se propagaría por
#       RunnableLambda → RunnableParallel → chain.batch.
#       Resultado: `chain.batch(...)` lanzaría la excepción y perderías
#       TODOS los resultados del lote (no solo el problemático).
#       Es el "todo o nada" por defecto de batch.
#   • Alternativa: `chain.batch(items, return_exceptions=True)` devuelve
#       la excepción EN LUGAR del resultado para los items que fallen.
#       Así puedes inspeccionar qué falló sin perder el resto.


# ==============================================================================
# 7. COMBINACIÓN DE RESULTADOS
# ==============================================================================
# Aplana la estructura anidada del RunnableParallel en un dict plano.
# Cada ITEM del lote pasa por este merge de forma independiente.
# ------------------------------------------------------------------------------
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado."""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"],
    }

merger = RunnableLambda(merge_results)

# 🔎 Para profundizar
#   • En batch, este merge se ejecuta N veces (una por item). Es una
#     función SIN llamadas al LLM → coste despreciable.
#   • Si quisieras añadir un campo común a TODO el lote (p. ej. "job_id"
#     o "timestamp" compartido), NO lo hagas aquí (se recalcularía por
#     item). Hazlo DESPUÉS del batch: `[{**r, "job_id": jid} for r in res]`.


# ==============================================================================
# 8. EJECUCIÓN PARALELA + CADENA FINAL
# ==============================================================================
# Sin cambios respecto a la Parte 1. La cadena es la MISMA.
# La diferencia está en CÓMO la invocamos: .invoke() vs .batch().
#
# ⚡ IMPORTANTE: la misma `chain` soporta TODOS los métodos:
#       chain.invoke(x)              → 1 item
#       chain.batch([x, y, z])       → N items, concurrente
#       chain.stream(x)              → 1 item, token a token
#       await chain.ainvoke(x)       → 1 item, async
#       await chain.abatch([...])    → N items, async + concurrente
#       await chain.astream(x)       → 1 item, async + streaming
# ------------------------------------------------------------------------------
parallel_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch,
})

chain = preprocessor | parallel_analysis | merger

# 🔎 Para profundizar
#   • Cambiar `.invoke` → `.batch` NO requiere tocar la definición de la
#     cadena. Esto es el "polimorfismo de Runnable": todas las cadenas
#     heredan el mismo contrato de métodos.
#   • Puedes pasar la MISMA cadena a `.invoke`, `.batch` o `.stream` según
#     el caso de uso. No hay versiones distintas.


# ==============================================================================
# 9. LOTE DE PRUEBA Y EJECUCIÓN CON .batch()
# ==============================================================================
reviews_batch = [
    "Excelente producto, muy satisfecho con la compra",
    "Terrible calidad, no lo recomiendo para nada",
    "Está bien, cumple su función básica pero nada especial",
]

resultado_batch = chain.batch(reviews_batch)

print(resultado_batch)

# 🔎 Para profundizar — LO MÁS IMPORTANTE DE ESTA PARTE 2
# ------------------------------------------------------------------------------
# ¿QUÉ DEVUELVE .batch()?
#   Una LISTA del MISMO tamaño que la entrada, en el MISMO ORDEN.
#   Si `reviews_batch` tiene 3 elementos, `resultado_batch` tiene 3 dicts.
#   El orden se PRESERVA aunque internamente terminen desordenados.
#
# ¿CÓMO SE COMPORTA ANTE ERRORES?
#   Por defecto, `return_exceptions=False`:
#     · Si CUALQUIER item lanza excepción → batch lanza la excepción.
#     · Pierdes TODOS los resultados (los que habían terminado también).
#     · En nuestro caso, los fallbacks internos lo evitan casi siempre.
#
#   Con `return_exceptions=True`:
#     · Los items exitosos devuelven su resultado normal.
#     · Los items fallidos devuelven la EXCEPCIÓN como "valor".
#     · Tú decides qué hacer con cada excepción.
#     · Ejemplo:
#         res = chain.batch(items, return_exceptions=True)
#         for r in res:
#             if isinstance(r, Exception):
#                 print("Falló:", r)
#             else:
#                 print(r["sentimiento"])
#
# ¿QUÉ ES `max_concurrency`?
#   El número máximo de items procesados SIMULTÁNEAMENTE.
#   · Por defecto: sin límite (todos a la vez) → riesgo de 429.
#   · Recomendado: 5–10 para OpenAI (respeta rate limits).
#   · Ejemplo: `chain.batch(items, max_concurrency=5)`.
#   · Con lotes muy grandes (miles), es OBLIGATORIO poner un límite.
#
# ¿QUÉ PASA CON LOS RATE LIMITS (429)?
#   · Si llegan demasiadas peticiones simultáneas, OpenAI responde 429.
#   · `.batch()` sin límite puede disparar muchos 429.
#   · Soluciones:
#       1. `max_concurrency=5` (sencillo).
#       2. `max_retries=N` en ChatOpenAI (reintentos automáticos).
#       3. Backoff exponencial personalizado (avanzado, con `retry` lib).
#
# ¿CUÁNDO .batch() ES MEJOR QUE UN BUCLE .invoke()?
#   · LATENCIA: aquí sí (paraleliza).
#   · CÓDIGO: más limpio y declarativo.
#   · ERRORES: fallback y return_exceptions más controlados.
#   · COSTE: NO cambia (sigues pagando por token).
#
# ¿CUÁNDO NO USAR .batch()?
#   · Si el orden no importa pero sí la latencia de cada uno (usa
#     `batch_as_completed` para mostrar resultados según llegan).
#   · Si necesitas streaming de cada item (usa `astream` en bucle).
#   · Si los textos son MUY dependientes entre sí (no deberían ir en
#     paralelo: p. ej. si cada uno depende del anterior).
#   · Si la API del proveedor no soporta muchas peticiones concurrentes
#     sin penalización (local models).
#
# ⚠️ CUIDADO CON EL "SILENT TRUNCATION":
#   `.batch()` no avisa si el lote es enorme. Si pasas 10.000 items,
#   intentará lanzar 10.000 peticiones (a menos que pongas
#   `max_concurrency`). Puedes agotar cuotas, tumbar la cuenta, o
#   simplemente colgar la máquina. SIEMPRE pon un límite en producción.
#
# 💡 COMPARACIÓN DE LATENCIA (n=3, ~2s por item):
#   Bucle .invoke():    ~6s (secuencial)
#   .batch():           ~2s (paralelo, si no hay throttling)
#   .batch(max=1):      ~6s (paralelo "falso": 1 a la vez)
#   .batch(max=5):      ~2s (3 items caben en 5)
#   .batch(max=2):      ~4s (2 + 1)
#   → La latencia depende del mínimo entre nº items y max_concurrency.


# ==============================================================================
# 10. COMPARATIVA DIRECTA: BUCLE vs .batch() (bonus didáctico)
# ==============================================================================
# Este bloque NO se ejecuta (está en un if False), pero es útil LEERLO
# para interiorizar la diferencia. Descoméntalo si quieres medir tiempos.
# ------------------------------------------------------------------------------
if False:  # Cambia a True para probar
    import time

    # --- OPCIÓN A: bucle con .invoke (secuencial) ---
    t0 = time.perf_counter()
    resultados_loop = [chain.invoke(r) for r in reviews_batch]
    t_loop = time.perf_counter() - t0

    # --- OPCIÓN B: .batch (concurrente) ---
    t0 = time.perf_counter()
    resultados_batch = chain.batch(reviews_batch)
    t_batch = time.perf_counter() - t0

    # --- OPCIÓN C: .batch con límite de concurrencia ---
    t0 = time.perf_counter()
    resultados_batch_5 = chain.batch(reviews_batch, max_concurrency=5)
    t_batch_5 = time.perf_counter() - t0

    # --- OPCIÓN D: .batch con errores encapsulados ---
    t0 = time.perf_counter()
    resultados_batch_exc = chain.batch(reviews_batch, return_exceptions=True)
    t_batch_exc = time.perf_counter() - t0

    print(f"Bucle .invoke:      {t_loop:.2f}s")
    print(f".batch():           {t_batch:.2f}s  ← ganancia ≈ {t_loop/t_batch:.1f}x")
    print(f".batch(max=5):      {t_batch_5:.2f}s")
    print(f".batch(exc=True):   {t_batch_exc:.2f}s")
    print(f"Resultados idénticos?: {resultados_loop == resultados_batch}")

# 🔎 Para profundizar
#   • Nota que `resultados_loop == resultados_batch` puede ser True o False:
#     con `temperature=0` DEBERÍA ser True, pero la API NO garantiza
#     determinismo perfecto (factores de red y balanceo). Si quieres
#     garantías, compara las claves y valida con tolerancia.
#   • El "speedup" varía con el nº de items, la latencia de la API y los
#     rate limits. En local (modelo pequeño) el speedup es MENOR porque el
#     cuello de botella es la CPU/GPU, no la red.
#   • `time.perf_counter()` es la opción correcta para medir (no `time.time()`).


# ==============================================================================
# 📊 RESUMEN DE PATRONES REUTILIZABLES (para tu estudio)
# ==============================================================================
#
# 1. PATRÓN "SECRETOS VÍA .env"  🆕
#    load_dotenv() + fail-fast con os.getenv. Nunca hardcodees claves.
#
# 2. PATRÓN "MISMO RUNNABLE, MÚLTIPLES MÉTODOS"
#    Una cadena LCEL soporta invoke/batch/stream/ainvoke/abatch/astream.
#    Cambias el método sin tocar la cadena.
#
# 3. PATRÓN "BATCH CON CONCURRENCIA CONTROLADA"
#    Usa `max_concurrency` SIEMPRE en producción. 5–10 es un buen default.
#    Sin límite → 429, cuotas agotadas, cuentas suspendidas.
#
# 4. PATRÓN "BATCH CON ERRORES ENCAPSULADOS"
#    `return_exceptions=True` + chequeo `isinstance(r, Exception)` permite
#    procesar parcialmente. Útil para lotes grandes donde algún item
#    inevitablemente falla.
#
# 5. PATRÓN "FALLBACK EN LA RAMA, NO EN EL BATCH"
#    Encapsula los errores DENTRO de cada rama (try/except + valor por
#    defecto). Así `.batch()` no necesita `return_exceptions` y el
#    contrato de datos se mantiene plano y uniforme.
#
# 6. PATRÓN "MEDIR ANTES DE OPTIMIZAR"
#    `time.perf_counter()` + comparar bucle vs batch. La ganancia depende
#    del caso real, no de la teoría.
#
# 7. PATRÓN "ORDEN PRESERVADO"
#    `.batch()` garantiza el orden aunque internamente sea asíncrono.
#    Si necesitas "según van terminando", usa `.batch_as_completed()`.
#
# ==============================================================================
# 🎯 EJERCICIOS PROPUESTOS (Parte 2)
# ==============================================================================
# A) Añade un CUARTO texto al lote y compara la salida. ¿Se mantiene el
#    orden? ¿Cuánto tarda vs. los 3 primeros?
#
# B) Añade un texto VACÍO ("") al lote. ¿Qué sentimiento detecta? ¿Por qué?
#    Pista: piensa en lo que hace tu `preprocess_text`.
#
# C) Ejecuta el bloque del apartado 10 (`if False` → `if True`). Comenta
#    los tiempos obtenidos y explica la diferencia entre las 4 opciones.
#
# D) Cambia `chain.batch(reviews_batch)` por
#    `chain.batch(reviews_batch, max_concurrency=1)` y luego `=2` y `=5`.
#    ¿Cómo cambia el tiempo? Dibuja una tabla tiempo vs max_concurrency.
#
# E) Añade un texto que SEGURO rompa el prompt (p. ej. con muchos `{` y `}`)
#    y observa cómo se comporta el JSON. ¿Cae en el `except` interno?
#    ¿O rompe `.batch()`? Justifica el resultado.
#
# F) (Avanzado) Sustituye `.batch` por `await chain.abatch(...)` dentro de
#    un `async def main_async()`. Necesitarás `asyncio.run(main_async())`.
#    ¿Qué diferencia de tiempo observas vs. `.batch` síncrono?
#
# G) 🆕 Comprueba que el fail-fast funciona: comenta temporalmente
#    `OPENAI_API_KEY` en tu `.env` (o renómbrala) y ejecuta el script.
#    Deberías ver el RuntimeError con el mensaje claro ANTES de cualquier
#    llamada a la API.
#
# ==============================================================================
# 🔗 CONEXIÓN CON LA PARTE 1
# ==============================================================================
# Todo lo aprendido en la Parte 1 (RunnableLambda, RunnableParallel, LCEL,
# temperature=0, fallback por rama, merge separado, prompt R-T-C) sigue
# siendo válido. La Parte 2 SOLO añade:
#   · Un método de invocación nuevo: `.batch()`.
#   · La carga segura de secretos vía `dotenv` + fail-fast.  🆕
#
# Es decir:
#   Parte 1 = "construir la cadena"       ← ya lo sabes
#   Parte 2 = "invocar en lote + secrets" ← nueva capa
#   Parte 3 (futura) = "streaming"        ← mismo patrón, otro método
# ==============================================================================