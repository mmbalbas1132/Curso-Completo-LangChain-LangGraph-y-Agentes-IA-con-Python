# 3.2 Cargar PDFs desde Google Drive con `GoogleDriveLoader`

> **Cómo usar este texto.** Léelo en orden: *Antes de empezar*, *Tutorial* y, cuando lo necesites, *Guía práctica*, *Referencia* y *Explicación*. Al final hay una *Autoevaluación* con respuestas.
>
> **Procedencia.** Documento elaborado a partir de `3.2_google_drive_loader_limpio.py` y de la documentación oficial de LangChain. Marcas: 📘 *(material del curso)* = procede solo del archivo; ⚠️ *(no contrastado)* = no se pudo confirmar en las páginas oficiales consultadas; 🔀 *(discrepancia)*; ✅ *(ejecutado o comprobado)* = observado al ejecutar código o al leer el código fuente instalado de `langchain-google-community` 5.1.1 (con `langchain-core` 1.7.7 y `pypdf` 7.1.0). **El script completo no se ejecutó**: necesita el flujo OAuth de Google en un navegador y acceso a tu Drive. Tampoco se abrieron `credentials.json` ni `token.json` (son credenciales privadas). Sí se comprobó, sin credenciales, la clase `DrivePyPDFLoader` sobre el PDF del tema (solo número de páginas y metadatos) y los valores por defecto de `GoogleDriveLoader`.

---

## Antes de empezar

### Qué vas a aprender

1. Qué es OAuth y para qué sirven `credentials.json` y `token.json`.
2. Configurar `GoogleDriveLoader` para leer PDFs de una carpeta de Drive (con subcarpetas).
3. Escribir un cargador propio (`DrivePyPDFLoader`) y pasárselo con `file_loader_cls`.
4. Controlar errores con `try/except`.

### Qué necesitas

- Haber visto el documento 3.1 (`Document`, `page_content`, `metadata`).
- Un proyecto de Google Cloud con la API de Drive activada y credenciales de tipo aplicación de escritorio (pasos de la documentación oficial).
- `pip install -qU "langchain-google-community[drive]" pypdf python-dotenv`.

### Mapa del tema

```text
credentials.json ──► (1ª vez) ventana de consentimiento en el navegador ──► token.json
                                                                              │
GoogleDriveLoader(folder_id, scopes, recursive, file_types=["pdf"], file_loader_cls)
   │  descarga cada PDF de la carpeta (y subcarpetas)
   ▼
DrivePyPDFLoader(file=<PDF en memoria>).load()  ──► [Document por página]
   │  GoogleDriveLoader añade a cada metadato: source y title
   ▼
documents  (lista de Document)
```

### Glosario

| Término | Significado |
|---|---|
| **OAuth** | Sistema por el que autorizas a una aplicación a acceder a tu Drive sin darle tu contraseña. |
| **`credentials.json`** | Archivo que identifica a tu aplicación ante Google (se descarga de Google Cloud). |
| **`token.json`** | Autorización ya concedida por ti; se genera tras aceptar el consentimiento. |
| **Scope (alcance)** | Permiso que se solicita. Aquí, `drive.readonly` (solo lectura). |
| **`folder_id`** | Identificador de la carpeta de Drive (parte final de su URL). |
| **`file_loader_cls`** | Clase que sabe convertir el archivo descargado en `Document`. |

---

## Tutorial

### Lección 1: importaciones (líneas 1 a 6)

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader
```

| Importación | Para qué se usa |
|---|---|
| `os`, `Path` | Variables de entorno y rutas de archivos |
| `load_dotenv` | Leer el `.env` |
| `PdfReader` (de `pypdf`) | Leer PDFs |
| `Document` | Crear los documentos de LangChain a mano |
| `GoogleDriveLoader` | Loader de Drive (`from langchain_google_community import GoogleDriveLoader` es el import de la documentación oficial) |

---

### Lección 2: un cargador de PDF propio (líneas 9 a 24)

```python
class DrivePyPDFLoader:
    def __init__(self, file=None, **kwargs):
        self.file = file

    def load(self):
        if self.file is None:
            return []
        reader = PdfReader(self.file)
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"page": i + 1}
            )
            for i, page in enumerate(reader.pages)
        ]
```

| Parte | Qué hace |
|---|---|
| `__init__(self, file=None, **kwargs)` | Guarda el archivo recibido. `**kwargs` absorbe argumentos extra sin usarlos. |
| `if self.file is None: return []` | Sin archivo, devuelve lista vacía. |
| `PdfReader(self.file)` | Abre el PDF. |
| `page.extract_text() or ""` | Texto de la página; si no hay texto, cadena vacía. |
| `Document(page_content=..., metadata={"page": i + 1})` | Un `Document` por página; `page` cuenta desde **1**. |

**¿Por qué existe esta clase?** `GoogleDriveLoader` descarga cada archivo a memoria y se lo pasa a la clase indicada en `file_loader_cls`. ✅ En el código fuente instalado, la llamada es literalmente `self.file_loader_cls(file=fh, **self.file_loader_kwargs)`, donde `fh` es el archivo descargado en memoria (`BytesIO`). Por eso `DrivePyPDFLoader` recibe un argumento llamado `file`.

**Comprobado** ✅: `DrivePyPDFLoader` leyendo en memoria (como hace Drive) el PDF del tema devolvió 2 `Document` con metadatos `{'page': 1}` y texto de tipo `str`.

🔀 *(diferencia con 3.1)* Aquí `page` cuenta desde 1; `PyPDFLoader` (3.1) lo cuenta desde 0 y añade más metadatos.

**Punto clave.** `GoogleDriveLoader` descarga; `DrivePyPDFLoader` convierte el PDF en `Document`.

---

### Lección 3: variables de entorno y rutas (líneas 27 a 37)

```python
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

current_dir = Path(__file__).resolve().parent
CREDENTIALS_PATH = current_dir / "credentials.json"
TOKEN_PATH = current_dir / "token.json"
```

| Línea | Qué hace |
|---|---|
| `base_dir = ...parent.parent` | Carpeta **superior** a la del script (la raíz del proyecto). |
| `load_dotenv(dotenv_path=base_dir / ".env")` | Busca el `.env` en esa raíz. |
| `GOOGLE_APPLICATION_CREDENTIALS = ""` | Fuerza el uso de OAuth ignorando credenciales de nube por defecto. La documentación oficial indica este mismo paso. |
| `current_dir` | Carpeta del propio script. |
| `CREDENTIALS_PATH` / `TOKEN_PATH` | `credentials.json` y `token.json` junto al script. |

La documentación oficial indica que por defecto el loader busca estos archivos en `~/.credentials/`; aquí se sobrescriben con `credentials_path` y `token_path`. ✅ (valores por defecto observados: `~/.credentials/credentials.json` y `~/.credentials/token.json`).

---

### Lección 4: el ID de la carpeta (línea 40)

```python
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "<ID_de_carpeta_por_defecto>")
```

> En este documento se sustituye el ID real por un marcador; el script lleva el ID de una carpeta concreta como valor por defecto.

`os.getenv(nombre, valor_por_defecto)` lee la variable `GOOGLE_DRIVE_FOLDER_ID` del `.env`; si no existe, usa el valor por defecto. 📘

---

### Lección 5: configurar el loader (líneas 43 a 51)

```python
loader = GoogleDriveLoader(
    folder_id=FOLDER_ID,
    credentials_path=str(CREDENTIALS_PATH),
    token_path=str(TOKEN_PATH),
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
    recursive=True,
    file_types=["pdf"],
    file_loader_cls=DrivePyPDFLoader
)
```

| Parámetro | Valor | Qué hace |
|---|---|---|
| `folder_id` | `FOLDER_ID` | Carpeta de Drive a leer |
| `credentials_path` | ruta a `credentials.json` | Identifica la aplicación |
| `token_path` | ruta a `token.json` | Donde se guarda/lee la autorización |
| `scopes` | `drive.readonly` | Permiso de **solo lectura** de todo el Drive 📘 |
| `recursive` | `True` | Incluye subcarpetas (por defecto es `False`) ✅ |
| `file_types` | `["pdf"]` | Solo PDFs |
| `file_loader_cls` | `DrivePyPDFLoader` | Clase que convierte cada PDF en `Document` |

Comprobaciones:

- ✅ Valor por defecto de `scopes` en la versión probada: `['https://www.googleapis.com/auth/drive.file']`. El script lo cambia a `drive.readonly`. La página oficial consultada **no documenta** el argumento `scopes` ni sus valores por defecto. ⚠️
- ✅ En el código fuente, `file_types` solo se acepta si se indica `folder_id`, y los nombres cortos como `"pdf"` se convierten al tipo MIME completo.
- La documentación oficial confirma `folder_id`, `credentials_path`, `token_path`, `recursive`, `file_types` y `file_loader_cls`.

🔀 *(inconsistencia de la propia documentación oficial)* La página dice primero que "solo se admiten Google Docs" y más abajo que por defecto se cargan documentos, hojas de cálculo y PDFs de una carpeta. El script depende de poder cargar PDFs mediante `file_loader_cls`; en el código fuente instalado hay una rama específica para `application/pdf` y para cuando `file_loader_cls` está definido ✅.

**Primera ejecución.** Según la documentación oficial, se abre una ventana de consentimiento en el navegador; si ya existe un `token.json` en esa ruta, **no se vuelve a pedir autenticación**.

---

### Lección 6: cargar y mostrar (líneas 54 a 66)

```python
try:
    print("Iniciando carga desde Google Drive...")
    documents = loader.load()

    print(f"✅ Páginas cargadas: {len(documents)}")
    if documents:
        print("\n--- Metadatos de la primera página ---")
        print(documents[0].metadata)
        print("\n--- Contenido (primeros 200 caracteres) ---")
        print(documents[0].page_content[:200] + "...")

except Exception as e:
    print(f"❌ Error al cargar documentos: {e}")
```

| Parte | Qué hace |
|---|---|
| `try / except Exception as e` | Si algo falla (credenciales, red, permisos), muestra el error sin que el programa se rompa con un traceback. |
| `loader.load()` | Descarga los PDFs y devuelve la lista de `Document`. |
| `len(documents)` | Total de **páginas** (no de archivos): cada página es un `Document`. |
| `if documents:` | Solo muestra datos si la lista no está vacía. |
| `documents[0].metadata` | Metadatos de la primera página. |
| `page_content[:200]` | Primeros 200 caracteres. |

**Metadatos esperados** ✅ *(deducido del código fuente instalado, no ejecutado contra Drive)*: tras cargar con `file_loader_cls`, `GoogleDriveLoader` añade a cada `Document`:

- `source` = `https://drive.google.com/file/d/<id_del_archivo>/view`
- `title` = nombre del archivo en Drive (si el loader propio no puso ya un `title`)

Por tanto `documents[0].metadata` debería contener `page` (del loader propio) más `source` y `title`. No se reproduce una salida real porque requiere tu Drive.

**Puntos clave.**

- "Páginas cargadas" cuenta `Document`, que son páginas.
- Un `try/except` amplio oculta la causa exacta si no se lee el mensaje.

### Ejercicios

Este archivo no incluye ejercicios propuestos.

---

## Guía práctica

### ¿Cómo preparo las credenciales?

Según la documentación oficial: crear o elegir un proyecto de Google Cloud, activar la API de Google Drive, crear credenciales para aplicación de escritorio y descargar el archivo como `credentials.json`; después poner `GOOGLE_APPLICATION_CREDENTIALS` a cadena vacía.

### ¿Cómo cambio de carpeta?

Define `GOOGLE_DRIVE_FOLDER_ID=<id>` en el `.env` (parte final de la URL de la carpeta).

### ¿Cómo cargo otros tipos de archivo?

Cambia `file_types` y proporciona un `file_loader_cls` adecuado a ese tipo (la documentación oficial menciona, por ejemplo, `UnstructuredFileIOLoader`).

### ¿Cómo evito repetir el consentimiento?

Conserva `token.json`: mientras exista en `token_path`, no se pide autenticación. 

### Seguridad

`credentials.json` y `token.json` son privados: **no los subas a repositorios ni los compartas**. Añádelos al `.gitignore`. Valora no dejar un ID de carpeta real como valor por defecto en el código.

### Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `❌ Error al cargar documentos: ...` | Credenciales ausentes o caducadas, falta de permisos, red | Leer el mensaje; revisar `credentials.json`, `token.json` y el `folder_id` |
| Se abre el navegador cada vez | No se guarda `token.json` en la ruta esperada | Comprobar `token_path` |
| Lista vacía | La carpeta no tiene PDFs o `file_types` no coincide | Revisar la carpeta y `recursive` |
| No encuentra el `.env` | Se busca en la carpeta superior al script | Colocarlo ahí o cambiar `dotenv_path` |
| `ModuleNotFoundError` | Faltan paquetes | Instalar `langchain-google-community[drive]` y `pypdf` |

---

## Referencia

### `GoogleDriveLoader`: parámetros usados

| Parámetro | Valor en el script | Por defecto observado ✅ | Documentación oficial |
|---|---|---|---|
| `folder_id` | `FOLDER_ID` | `None` | Sí |
| `credentials_path` | `credentials.json` junto al script | `~/.credentials/credentials.json` | Sí |
| `token_path` | `token.json` junto al script | `~/.credentials/token.json` | Sí |
| `scopes` | `drive.readonly` | `drive.file` | No documentado ⚠️ |
| `recursive` | `True` | `False` | Sí |
| `file_types` | `["pdf"]` | `None` | Sí |
| `file_loader_cls` | `DrivePyPDFLoader` | `None` | Sí |

### Metadatos

| Clave | Quién la añade | Valor |
|---|---|---|
| `page` | `DrivePyPDFLoader` | Número de página desde 1 ✅ |
| `source` | `GoogleDriveLoader` | URL de Drive del archivo ✅ (código fuente) |
| `title` | `GoogleDriveLoader` | Nombre del archivo ✅ (código fuente) |

### Archivos y variables

| Elemento | Uso |
|---|---|
| `credentials.json` | Identifica la aplicación (privado) |
| `token.json` | Autorización concedida (privado) |
| `.env` en la carpeta superior | `GOOGLE_DRIVE_FOLDER_ID` |
| `GOOGLE_APPLICATION_CREDENTIALS=""` | Forzar OAuth |

---

## Explicación

### ¿Por qué un loader propio para los PDFs?

Porque `GoogleDriveLoader` solo sabe convertir Google Docs y similares por sí mismo; para otros archivos delega en la clase que le indiques con `file_loader_cls`. 📘 Aquí se usa `pypdf` y se genera un `Document` por página.

### ¿Por qué `drive.readonly`?

Es un permiso de solo lectura, suficiente para cargar documentos y más seguro que uno de escritura. El valor por defecto del loader (`drive.file`) es distinto. ✅ ⚠️ No se contrastó qué archivos concretos permite ver cada alcance.

### ¿Por qué se vacía `GOOGLE_APPLICATION_CREDENTIALS`?

Para que LangChain use el flujo OAuth del usuario y no unas credenciales de nube que pudiera haber en el entorno. La documentación oficial incluye este mismo paso.

### Lo que este documento no cubre

- La ejecución real contra tu Drive y su salida.
- Cómo crear el proyecto en Google Cloud paso a paso con capturas.
- Qué hacer con los documentos después (trocear, indexar).

### Observaciones sobre el archivo

- El script usa un ID de carpeta real como valor por defecto.
- `load_dotenv` busca el `.env` en la carpeta **superior** al script, pero `credentials.json` y `token.json` se buscan en la carpeta del script.
- `DrivePyPDFLoader.__init__` ignora `**kwargs`; sirve para que el loader no falle si Drive le pasa argumentos extra (`file_loader_kwargs`).
- Los `Document` de este loader cuentan páginas desde 1, a diferencia de `PyPDFLoader` (desde 0).

---

## Autoevaluación

1. ¿Para qué sirven `credentials.json` y `token.json`?
2. ¿Qué hace `GOOGLE_APPLICATION_CREDENTIALS = ""`?
3. ¿Qué es `file_loader_cls` y qué recibe?
4. ¿Qué permiso solicita el script y qué significa?
5. ¿Qué hace `recursive=True`?
6. ¿Qué cuenta `len(documents)`?
7. ¿Qué metadatos añade `GoogleDriveLoader` a cada página según el código fuente?
8. ¿Por qué no se vuelve a abrir el navegador en la segunda ejecución?
9. ¿Qué ficheros no deben subirse a un repositorio?

### Respuestas

1. El primero identifica la aplicación ante Google; el segundo guarda la autorización concedida.
2. Fuerza el uso de OAuth ignorando credenciales de nube por defecto.
3. La clase que convierte el archivo descargado en `Document`; recibe `file=` con el archivo en memoria.
4. `drive.readonly`: solo lectura.
5. Incluye las subcarpetas.
6. El número de páginas (cada página es un `Document`).
7. `source` (URL de Drive) y `title` (nombre del archivo).
8. Porque ya existe `token.json` en `token_path`.
9. `credentials.json` y `token.json`.

---

## Resumen

| Concepto | En una línea |
|---|---|
| OAuth | `credentials.json` → consentimiento → `token.json` |
| `GoogleDriveLoader` | Descarga archivos de una carpeta de Drive |
| `file_loader_cls` | Clase que convierte cada archivo en `Document` |
| `DrivePyPDFLoader` | Un `Document` por página con `pypdf` ✅ |
| `drive.readonly` | Permiso de solo lectura |
| Seguridad | No compartir credenciales ni token |

---

## Fuentes consultadas

- `GoogleDriveLoader`: <https://docs.langchain.com/oss/python/integrations/document_loaders/google_drive>
- Código fuente instalado de `langchain-google-community` 5.1.1 (valores por defecto y comportamiento de `file_loader_cls`).
- Material del tema: `3.2_google_drive_loader_limpio.py`.

*Documento generado a partir de los materiales del tema y la documentación oficial de LangChain.*
