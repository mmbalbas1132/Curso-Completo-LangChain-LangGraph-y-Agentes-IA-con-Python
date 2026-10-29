# ═══════════════════════════════════════════════════════════════════════════
# MATERIAL DE ESTUDIO: INTRODUCCIÓN A LCEL Y Runnables CON RunnableLambda
#
# Objetivo: Entender cómo encadenar funciones de Python personalizadas
# usando el operador tubería (|) de LangChain Expression Language (LCEL).
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ───────────────────────────────────────────────────────────────────────────

# dotenv: Nos permite cargar variables de entorno desde un archivo .env.
# Aunque en este script no usamos ninguna API Key, es una buena práctica
# tenerlo al inicio de cualquier proyecto de LangChain.
from dotenv import load_dotenv
import os  # (Nota: 'os' no se usa en este ejemplo, pero se importa por costumbre)

# Cargamos las variables de entorno (busca el archivo .env en el directorio).
load_dotenv()

# 💡 CONCEPTO CLAVE: RunnableLambda
# LangChain necesita que todos los componentes de una cadena (prompts, modelos,
# parsers) sean de un tipo llamado "Runnable".
# RunnableLambda es un "adaptador" que convierte cualquier función normal de
# Python (o función lambda) en un Runnable, permitiéndonos usarla dentro de LCEL.
from langchain_core.runnables import RunnableLambda


# ───────────────────────────────────────────────────────────────────────────
# 2. DEFINICIÓN DE LOS PASOS (PASO 1 Y PASO 2)
# ───────────────────────────────────────────────────────────────────────────

# PASO 1: Usando una función lambda (anónima)
# - Recibe un valor 'x' (en este caso, el número 43).
# - Devuelve un string formateado: "Numero 43".
# - Al envolverlo en RunnableLambda, ahora 'paso1' es un eslabón válido
#   para nuestra cadena LCEL.
paso1 = RunnableLambda(lambda x: f"Numero {x}")


# PASO 2: Usando una función clásica de Python (def)
# - Recibe un texto (que será el output del paso1).
# - Devuelve una lista que contiene ese texto repetido 2 veces.
# - Ejemplo: "Numero 43" se convierte en ["Numero 43", "Numero 43"].
def duplicar_texto(texto):
    return [texto] * 2

# Envolvemos la función 'duplicar_texto' en un RunnableLambda.
paso2 = RunnableLambda(duplicar_texto)


# ───────────────────────────────────────────────────────────────────────────
# 3. CREACIÓN DE LA CADENA LCEL
# ───────────────────────────────────────────────────────────────────────────

# 💡 CONCEPTO CLAVE: El operador tubería (|)
# En LCEL, el símbolo "|" conecta componentes. Funciona igual que en Linux:
# la salida (output) de un componente se pasa automáticamente como entrada (input)
# al siguiente componente.
# El flujo de datos es estrictamente de izquierda a derecha.
cadena = paso1 | paso2


# ───────────────────────────────────────────────────────────────────────────
# 4. EJECUCIÓN DE LA CADENA
# ───────────────────────────────────────────────────────────────────────────

# 💡 CONCEPTO CLAVE: .invoke()
# Todos los Runnables en LangChain tienen el método .invoke() para ejecutarse.
# Le pasamos el valor inicial (43). Este valor entra en 'paso1',
# y el resultado fluye hacia 'paso2'.
resultado = cadena.invoke(43)


# ───────────────────────────────────────────────────────────────────────────
# 5. RESULTADO
# ───────────────────────────────────────────────────────────────────────────

# Imprimimos el resultado final en la consola.
# La salida esperada es: ['Numero 43', 'Numero 43']
print(resultado)