# ============================================================
# BLOQUE 0 — IMPORTS
# ------------------------------------------------------------
# Importamos dos "document loaders" del paquete
# `langchain_community.document_loaders`.
#
# ¿Qué es un document loader?
#   Es un objeto de LangChain cuya única responsabilidad es
#   convertir una FUENTE EXTERNA (un PDF, una URL, un CSV,
#   una base de datos, Notion, etc.) en una lista de objetos
#   `Document`. Ese `Document` es el "ladrillo" con el que
#   luego se construyen splitters, embeddings, vector stores
#   y, finalmente, un RAG.
#
# ¿Por qué `langchain_community` y no `langchain`?
#   LangChain se dividió en varios paquetes:
#     - `langchain-core`: abstracciones base (Document, Runnable…)
#     - `langchain`: cadenas y utilidades de alto nivel
#     - `langchain-community`: integraciones mantenidas por la
#       comunidad (PDF, web, Notion, etc.)
#     - `langchain-openai`, `langchain-anthropic`…: integraciones
#       de proveedores concretos.
#   Los loaders de PDF y web viven, por tanto, en `community`.
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader


# ============================================================
# BLOQUE 1 — RAMA A: CARGAR UN PDF LOCAL
# ------------------------------------------------------------
# 1.1) Construimos el loader.
#      `PyPDFLoader` recibe la RUTA al fichero PDF.
#      Ojo: aquí solo se prepara el loader; el PDF todavía
#      NO se ha leído del disco. Es "perezoso" por diseño.
#
#      Requisito de instalación: `pip install pypdf`
#      (PyPDFLoader usa pypdf por debajo).
# ============================================================
loader = PyPDFLoader("/home/mmbal/proyectos/curso-langchain/Tema3/CV_Manuel_Balbas_Naveira_2026-11-10.pdf")

# 1.2) `load()` dispara la lectura real.
#      Devuelve una LISTA de `Document`, típicamente
#      UN Document POR PÁGINA del PDF.
#      Cada `Document` tiene dos atributos:
#        - `page_content`: el texto extraído de esa página.
#        - `metadata`: un dict con info de contexto, p. ej.
#          {"source": "...CV...pdf", "page": 0, "total_pages": N}.
#      El campo `page` es 0-based: la primera página es la 0.
pages = loader.load()

# 1.3) Iteramos la lista de Documents.
#      `enumerate(pages)` nos da a la vez el índice (i) y el
#      propio Document (page). Usamos `i + 1` SOLO para que
#      el mensaje al usuario sea 1-based ("Página 1" en vez
#      de "Página 0"). El dato real en metadata sigue siendo
#      0-based: no lo confundas.
#      `page` es un `Document`, así que accedemos a sus dos
#      atributos: `page_content` y `metadata`.
for i, page in enumerate(pages):
    print(f"Página {i + 1}:")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")


# ============================================================
# BLOQUE 2 — RAMA B: CARGAR UNA PÁGINA WEB
# ------------------------------------------------------------
# 2.1) Mismo patrón, distinta fuente.
#      `WebBaseLoader` recibe una URL y por debajo:
#        - descarga el HTML con `requests`,
#        - lo limpia/parsea (BeautifulSoup),
#        - extrae el texto visible.
#
#      Requisito de instalación:
#        `pip install beautifulsoup4 lxml requests`
#      (o el extra `langchain-community[web]`).
#
#      Nota: aquí la variable se llama `loader2` porque en el
#      bloque anterior ya usamos `loader`. Se mantiene el
#      nombre original por respeto al código del alumno.
# ============================================================
loader2 = WebBaseLoader("https://es.wikipedia.org/wiki/Inteligencia_artificial")
pages2 = loader2.load()

# 2.2) Mismo bucle que en la rama A.
#      En la práctica, `WebBaseLoader` suele devolver
#      UN solo Document (uno por URL), no uno por sección.
#      Los metadatos típicos son {"source": url, "title": ...,
#      "language": ...}. Fíjate en que las CLAVES del dict
#      de metadata NO son las mismas que en el PDF: eso es
#      normal y es una lección importante.
for i, page in enumerate(pages2):
    print(f"Página {i + 1}:")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")


# ============================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ============================================================
# 1) Los dos imports podrían unificarse en una sola línea:
#       from langchain_community.document_loaders import (
#           PyPDFLoader, WebBaseLoader,
#       )
#    No lo aplico por la regla de preservación.
#
# 2) `loader` y `loader2` solo se usan una vez, justo antes
#    de `load()`. Podrían encadenarse:
#       pages = PyPDFLoader(...).load()
#    Es más compacto, pero menos legible para un principiante.
#    No lo aplico.
#
# 3) Los dos bucles `for` son idénticos. Se podrían extraer a
#    una función auxiliar `imprimir_paginas(pages)`.
#    No lo aplico: cambiaría la estructura del código original.
#
# 4) No hay `try/except` ni `if __name__ == "__main__":`.
#    En un script real conviene añadirlos, pero aquí el
#    objetivo es didáctico y lineal. No lo aplico.