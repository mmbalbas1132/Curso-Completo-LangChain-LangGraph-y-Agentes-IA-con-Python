# 3.1 Cargar documentos: `PyPDFLoader` y `WebBaseLoader`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.1_document_loaders_limpio.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)* = la documentación o lo observado difiere del archivo; ✅ *(ejecutado)* = comportamiento observado al ejecutar código con `langchain-community` 0.4.2, `langchain-core` 1.7.7 y `pypdf` 7.1.0. El PDF del tema es un currículum personal: **se comprobó su estructura (número de páginas y claves de metadatos), pero su contenido no se reproduce aquí**. `WebBaseLoader` no se probó contra Wikipedia (sin acceso a esa web desde el entorno de pruebas); se probó contra una página HTML local de ejemplo.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es un *document loader* y qué devuelve (`Document`).
2. Cargar un PDF con `PyPDFLoader`: un `Document` por página.
3. Cargar una página web con `WebBaseLoader`.
4. Leer `page_content` (el texto) y `metadata` (los datos sobre el texto).

### Qué necesitas

- Python básico (listas, `for`, f-strings).
- `pip install -U langchain-community pypdf beautifulsoup4` (comandos de las páginas oficiales de cada loader).
- Un PDF y acceso a internet para la parte web. **No hace falta clave de API.**

### Mapa del tema

```text
PDF  ──► PyPDFLoader(ruta).load()    ──► [Document, Document, ...]   (uno por página)
Web  ──► WebBaseLoader(url).load()   ──► [Document]                   (uno por URL)
                                              │
                                  .page_content (texto)  +  .metadata (diccionario)
```

### Glosario

| Término | Significado |
|---|---|
| **Document loader** | Objeto que lee una fuente (PDF, web, Drive…) y la convierte en `Document`. |
| **`Document`** | Objeto con dos campos: `page_content` (texto) y `metadata` (diccionario). |
| **`PyPDFLoader`** | Loader de PDFs basado en `pypdf`. |
| **`WebBaseLoader`** | Loader que descarga una página web y extrae su texto. |
| **Metadatos** | Datos sobre el texto: ruta, página, título, idioma… |

---

## Tutorial

### Lección 1: importaciones (líneas 1 y 2)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
```

Ambos loaders se importan del paquete `langchain_community`, como indican las páginas oficiales de cada uno.

⚠️ *(observación)* Al importar `langchain_community` con la versión 0.4.2, Python mostró un aviso `DeprecationWarning` que dice que `langchain-community` "is being sunset and is no longer actively maintained" y remite a una guía de migración hacia paquetes de integración independientes. ✅ Las páginas oficiales consultadas siguen documentando estos imports. El script funciona, pero conviene saberlo para el futuro.

---

### Lección 2: cargar un PDF (líneas 4 y 6)

```python
loader = PyPDFLoader("/home/mmbal/proyectos/curso-langchain/Tema3/CV_Manuel_Balbas_Naveira_2026-11-10.pdf")

pages = loader.load()
```

| Línea | Qué hace |
|---|---|
| `PyPDFLoader("ruta")` | Crea el loader apuntando al archivo. Todavía no lee nada. |
| `loader.load()` | Lee el PDF y devuelve una **lista** de `Document`. |

**Comprobado** ✅: con el PDF del tema, `load()` devolvió una lista de 2 `Document`, uno por página (el PDF tiene 2 páginas). La documentación oficial confirma que cada página es un `Document` propio.

🔀 *(matiz)* La ruta del script es absoluta y pertenece al ordenador de quien lo escribió (`/home/mmbal/.../Tema3/...`). En otra máquina o carpeta dará error de archivo no encontrado. Una ruta relativa al propio script (como hace 3.2 con `Path(__file__)`) es más portable.

---

### Lección 3: recorrer las páginas (líneas 8 a 11)

```python
for i, page in enumerate(pages):
    print(f"Página {i + 1}:")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")
```

| Elemento | Qué hace |
|---|---|
| `enumerate(pages)` | Da el índice `i` (desde 0) y cada `Document`. |
| `i + 1` | Muestra la página contando desde 1, más natural para leer. |
| `page.page_content` | Texto extraído de esa página (un `str`). ✅ |
| `page.metadata` | Diccionario con datos del documento. |

**Comprobado** ✅: las claves de `metadata` que devolvió `PyPDFLoader` en este PDF son:

```text
author, creationdate, creator, page, page_label, producer, source, title, total_pages
```

Y la numeración de páginas: la primera tiene `page: 0` y `page_label: '1'`. La documentación oficial confirma `source` (ruta) y `page` (índice desde cero).

**Puntos clave.**

- `page` en los metadatos empieza en **0**; el script suma 1 solo para mostrarlo.
- Qué claves hay depende del PDF y de la versión (aquí aparecen 9).

### Lección 4: cargar una página web (líneas 14 a 20)

```python
loader2 = WebBaseLoader("https://es.wikipedia.org/wiki/Inteligencia_artificial")
pages2 = loader2.load()

for i, page in enumerate(pages2):
    print(f"Página {i + 1}:")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")
```

El patrón es el mismo: crear el loader, `load()` y recorrer.

**Comprobado** ✅ (con una página HTML local de prueba, no con Wikipedia): `WebBaseLoader(url).load()` devolvió **1** `Document` con estos metadatos:

```text
{'source': <url>, 'title': 'Prueba', 'description': 'desc', 'language': 'es'}
```

Coincide con la documentación oficial, que indica que para HTML se obtienen `source`, `title`, `description` y `language` (según la página).

⚠️ *(observación)* En la prueba, el `page_content` fue `'PruebaHola mundo'`: el título y el cuerpo quedaron pegados sin separador. Con una página real como Wikipedia el texto será mucho más largo y también incluirá menús y otros elementos de la página (no se comprobó).

⚠️ *(observación)* Al crear el loader apareció el aviso `USER_AGENT environment variable not set, consider setting it to identify your requests.` ✅ La documentación oficial consultada no menciona esa variable; el aviso lo emite la propia librería.

Una página web da **un** `Document`, así que el texto "Página 1" del script es solo un contador del bucle.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo cargo un PDF?

```python
docs = PyPDFLoader("ruta/al/archivo.pdf").load()
```

### ¿Cómo sé cuántas páginas tiene?

`len(docs)`. ✅ (2 en el PDF del tema).

### ¿Cómo leo solo los metadatos?

`docs[0].metadata`.

### ¿Cómo cargo varias URLs?

La documentación oficial indica que `WebBaseLoader` también acepta una lista de URLs.

### ¿Cómo evito cargar todo en memoria?

Ambos loaders ofrecen `.lazy_load()`, que entrega los documentos de uno en uno (confirmado en las páginas oficiales de cada loader).

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `FileNotFoundError` / error al abrir el PDF | La ruta absoluta del script no existe en tu equipo | Usar la ruta de tu PDF |
| `ModuleNotFoundError: langchain_community` | Paquete no instalado | `pip install -U langchain-community` |
| Error con `pypdf` o `bs4` | Falta una dependencia | Instalar `pypdf` y `beautifulsoup4` |
| La web no carga | Sin conexión o el servidor bloquea la petición | Revisar la conexión; la documentación advierte de que subir `requests_per_second` puede provocar bloqueos |

---

## Referencia

### `PyPDFLoader`

| Elemento | Detalle |
|---|---|
| Import | `from langchain_community.document_loaders import PyPDFLoader` |
| Constructor | Ruta del PDF (único argumento usado en el script) |
| `.load()` | Lista de `Document`, uno por página ✅ |
| `.lazy_load()` | Documentos de uno en uno |
| Metadatos observados ✅ | `author, creationdate, creator, page, page_label, producer, source, title, total_pages` |
| Credenciales | Ninguna |

### `WebBaseLoader`

| Elemento | Detalle |
|---|---|
| Import | `from langchain_community.document_loaders import WebBaseLoader` |
| Constructor | Una URL o una lista de URLs |
| `.load()` | Lista de `Document` ✅ (1 por URL) |
| `.aload()` | Carga concurrente |
| `.lazy_load()` / `.alazy_load()` | De uno en uno |
| Metadatos (HTML) ✅ | `source, title, description, language` |
| `requests_per_second` | Por defecto 2 (según la documentación oficial) |
| Credenciales | Ninguna |

### `Document`

| Campo | Tipo observado ✅ | Contenido |
|---|---|---|
| `page_content` | `str` | Texto |
| `metadata` | `dict` | Datos sobre el texto |

---

## Explicación

### ¿Por qué convertir todo a `Document`?

Porque así PDFs, webs y otras fuentes quedan en el **mismo formato** (texto + metadatos) y el resto de la aplicación no necesita saber de dónde vino cada texto. 📘 Es la base para trocear, indexar y consultar después.

### ¿Por qué un `Document` por página en el PDF?

Es el comportamiento de `PyPDFLoader` ✅: permite saber de qué página sale cada texto gracias a `metadata["page"]`.

### Lo que este documento no cubre

- La salida real de Wikipedia con `WebBaseLoader` (no se pudo comprobar desde el entorno de pruebas).
- El contenido del PDF del tema.
- Qué hacer después con los documentos (trocear, indexar): no aparece en este archivo.

### Observaciones sobre el archivo

- Ruta absoluta del PDF ligada a otro equipo (ver Lección 2).
- El bucle imprime **todo** el contenido; con Wikipedia saldrá muchísimo texto.
- Las dos partes (PDF y web) repiten el mismo patrón; podrían ir en una función, pero el archivo las deja separadas por claridad.

---

## Autoevaluación

1. ¿Qué dos campos tiene un `Document`?
2. ¿Cuántos `Document` devuelve `PyPDFLoader` para un PDF de 2 páginas?
3. ¿Desde qué número cuenta `metadata["page"]`?
4. ¿Por qué el script imprime `i + 1`?
5. ¿Qué metadatos devuelve `WebBaseLoader` para una página HTML según la documentación?
6. ¿Qué aviso da `langchain_community` al importarse en la versión probada?
7. ¿Qué problema tiene la ruta del PDF en el script?
8. ¿Qué método permite cargar sin llenar la memoria?

### Respuestas

1. `page_content` y `metadata`.
2. Dos, uno por página ✅.
3. Desde 0.
4. Para mostrar la página contando desde 1.
5. `source`, `title`, `description` y `language` (según la página).
6. Que `langchain-community` está siendo retirado y ya no se mantiene activamente ✅.
7. Es absoluta y pertenece a otro equipo; no funcionará en otra máquina.
8. `.lazy_load()`.

---

## Resumen

| Concepto | En una línea |
|---|---|
| Loader | Lee una fuente y devuelve `Document` |
| `PyPDFLoader` | Un `Document` por página ✅ |
| `WebBaseLoader` | Un `Document` por URL ✅ |
| `Document` | `page_content` + `metadata` |
| `metadata["page"]` | Índice de página desde 0 ✅ |

---

## Fuentes consultadas

- `PyPDFLoader`: <https://python.langchain.com/docs/integrations/document_loaders/pypdfloader>
- `WebBaseLoader`: <https://docs.langchain.com/oss/python/integrations/document_loaders/web_base>
- Material del tema: `3.1_document_loaders_limpio.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
