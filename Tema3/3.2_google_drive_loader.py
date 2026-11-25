# ============================================================
# BLOQUE 0 — IMPORTS
# ------------------------------------------------------------
# - os: lo usamos para leer una variable de entorno
#       (`os.getenv`) y para escribir en `os.environ`.
# - Path: ruta "moderna" de Python (pathlib). Nos permite
#       construir rutas con `/` sin depender del SO, y
#       resolver la ruta absoluta del script con `.resolve()`.
# - load_dotenv: lee un fichero `.env` y vuelca sus claves
#       como variables de entorno del proceso.
# - PdfReader: de la librería `pypdf`. Es el parseador puro
#       de PDF, sin LangChain por medio. Lo usamos dentro de
#       nuestro loader personalizado.
# - Document: la MISMA clase que viste en 3.1. Es la unidad
#       atómica de LangChain (`page_content` + `metadata`).
#       Importante: viene de `langchain_core.documents`, que
#       es el paquete base, no de `langchain_community`.
# - GoogleDriveLoader: el loader oficial de LangChain para
#       Google Drive. Vive en `langchain_google_community`.
# ============================================================
import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader


# ============================================================
# BLOQUE 1 — CLASE PERSONALIZADA: DrivePyPDFLoader
# ------------------------------------------------------------
# ¿Por qué definimos esta clase ANTES de usarla?
#   Porque en Python los nombres deben existir en el momento
#   en que se evalúan. Cuando más abajo escribimos
#   `file_loader_cls=DrivePyPDFLoader`, el intérprete necesita
#   que el nombre `DrivePyPDFLoader` YA esté definido.
#   Por eso el `class` va arriba del todo.
#
# ¿Qué es `file_loader_cls`?
#   `GoogleDriveLoader` localiza los ficheros del Drive y,
#   para CADA uno, delega la lectura real en otra clase
#   (el "file loader"). Por defecto usa un loader interno
#   (basado en `unstructured`), pero aquí lo SUSTITUIMOS por
#   nuestra propia clase ligera basada en `pypdf`.
#
# ¿Qué gana el alumno con esto?
#   1. Menos dependencias pesadas (`unstructured` arrastra
#      mucho: Tesseract, Poppler, etc.).
#   2. Control total sobre `page_content` y `metadata`.
# ============================================================
class DrivePyPDFLoader:
    def __init__(self, file=None, **kwargs):
        # Guardamos la referencia al fichero que nos pase
        # GoogleDriveLoader. Puede ser una ruta local
        # (tras descargarlo a un tmp) o un objeto BytesIO.
        # `file=None` por defecto permite instanciar la
        # clase "vacía" sin que reviente; el `load()`
        # devolverá [] en ese caso.
        #
        # `**kwargs` acepta cualquier argumento extra que
        # GoogleDriveLoader decida pasar (por ejemplo
        # `file_path`, `metadata`…). Aunque aquí NO los
        # usemos, es importante aceptarlos para que la
        # firma sea compatible con lo que espera el
        # orquestador. Si los quitáramos, GoogleDriveLoader
        # podría lanzar `TypeError: unexpected keyword`.
        self.file = file

    def load(self):
        # Defensa: si no nos han dado fichero, no hay nada
        # que leer. Devolvemos una lista vacía en lugar de
        # lanzar excepción. Esto es deliberado: un loader
        # debe poder "no encontrar nada" sin romper el
        # pipeline.
        if self.file is None:
            return []

        # Abrimos el PDF con pypdf. `PdfReader` es la API
        # clásica: lee el fichero, indexa páginas y expone
        # `reader.pages`, que es iterable.
        reader = PdfReader(self.file)

        # Construimos la lista de Documents.
        # Fíjate en el `or ""`: `extract_text()` puede
        # devolver `None` cuando la página no tiene capa
        # de texto (p. ej. un escaneo). Para que
        # `page_content` sea SIEMPRE un str (contrato de
        # `Document`), caemos a string vacío.
        #
        # Fíjate también en `metadata={"page": i + 1}`.
        # ¡Ojo! Aquí es 1-based, mientras que el
        # `PyPDFLoader` de 3.1 pone `page` en 0-based.
        # No es un bug: es una decisión de diseño del
        # alumno. Los metadatos son "lo que tú decidas".
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"page": i + 1}
            )
            for i, page in enumerate(reader.pages)
        ]


# ============================================================
# BLOQUE 2 — PREPARACIÓN DEL ENTORNO (pasos 1 a 5)
# ============================================================

# 1. Cargar variables de entorno (buscando el .env en la raíz del proyecto)
#    ------------------------------------------------------------
#    `Path(__file__)` es la ruta de ESTE fichero .py.
#    `.resolve()` la convierte en absoluta (resuelve `..`
#    y symlinks).
#    `.parent` sube un nivel.
#    `.parent.parent` sube DOS niveles.
#    Conclusión: si este script está en
#    `.../Tema3/3.2_google_drive_loader.py`, `base_dir`
#    será `.../curso-langchain`. Ahí esperamos el `.env`.
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")

# 2. Forzar a LangChain a usar OAuth (ignorar credenciales por defecto de la nube)
#    ------------------------------------------------------------
#    `GOOGLE_APPLICATION_CREDENTIALS` es la variable
#    estándar de Google Cloud para indicar un fichero de
#    cuenta de servicio (service account).
#    Si esa variable tiene una ruta válida, las librerías
#    de Google la usan AUTOMÁTICAMENTE y se saltan el flujo
#    OAuth interactivo (que es el que abre el navegador y
#    pide "Permitir acceso").
#    Al ponerla a string vacío, forzamos a que Google
#    IGNORE cualquier credencial de entorno y use el par
#    `credentials.json` + `token.json` que pasamos más
#    abajo. Es un "truco" muy habitual en tutoriales de
#    OAuth en local.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# 3. Definir rutas relativas al propio script
#    ------------------------------------------------------------
#    De nuevo `Path(__file__).resolve().parent`: esta vez
#    subimos solo UN nivel (la carpeta del script).
#    Ahí viven `credentials.json` (descargado de Google
#    Cloud Console) y `token.json` (generado la primera
#    vez que se completa el flujo OAuth; en adelante se
#    reutiliza para no volver a pedir permiso).
current_dir = Path(__file__).resolve().parent
CREDENTIALS_PATH = current_dir / "credentials.json"
TOKEN_PATH = current_dir / "token.json"

# 4. Obtener el ID de la carpeta desde .env (o usar el valor por defecto)
#    ------------------------------------------------------------
#    `os.getenv(clave, default)`:
#      - si la variable existe en el entorno → la devuelve;
#      - si no existe → devuelve el segundo argumento.
#    Ese ID de carpeta se puede sacar de la URL de Drive:
#    https://drive.google.com/drive/folders/<ESTE_ES_EL_ID>
#    Es una cadena larga y opaca. Por eso se recomienda
#    tenerla en `.env` y no hardcodeada, pero aquí se
#    mantiene el literal por respeto al código original.
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "18I97qr1prZMAzj_3YpB3L2yIboH7FHHd")

# 5. Configurar el loader con permiso de lectura completa (drive.readonly)
#    ------------------------------------------------------------
#    Parámetros:
#      - folder_id: qué carpeta de Drive escanear.
#      - credentials_path / token_path: el par OAuth
#        descargado/generado (ver comentario del paso 3).
#      - scopes: permisos OAuth. `drive.readonly` es
#        el MÍNIMO privilegio necesario para LEER. Si
#        quisiéramos descargar y escribir, habría que
#        añadir otros scopes (drive.file, drive…).
#      - recursive=True: entra también en subcarpetas.
#      - file_types=["pdf"]: filtra por MIME/extensión,
#        ignorando docs, hojas de cálculo, etc.
#      - file_loader_cls=DrivePyPDFLoader: LO CLAVE.
#        Le decimos que use NUESTRA clase para parsear
#        cada PDF, en lugar del loader por defecto.
loader = GoogleDriveLoader(
    folder_id=FOLDER_ID,
    credentials_path=str(CREDENTIALS_PATH),
    token_path=str(TOKEN_PATH),
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
    recursive=True,
    file_types=["pdf"],
    file_loader_cls=DrivePyPDFLoader
)


# ============================================================
# BLOQUE 3 — EJECUCIÓN (paso 6)
# ------------------------------------------------------------
# `loader.load()` es el momento en que ocurre TODO:
#   1. OAuth (abre navegador la primera vez y guarda
#      token.json; en adelante lo reutiliza).
#   2. Listado recursivo de la carpeta (API de Drive).
#   3. Descarga temporal de cada PDF.
#   4. Parseo de cada PDF con `DrivePyPDFLoader`
#      → 1 Document por página con metadata {"page": i+1}.
#   5. Aplanado de todos los Documents en una sola lista.
# El resultado es una `List[Document]` exactamente igual
# que en 3.1: el tipo de retorno es el mismo, cambia la
# fuente. Ese es EL punto didáctico del script.
try:
    print("Iniciando carga desde Google Drive...")
    documents = loader.load()

    print(f"✅ Páginas cargadas: {len(documents)}")
    if documents:
        print("\n--- Metadatos de la primera página ---")
        # `documents[0]` es un `Document`: mostramos su
        # metadata (para ver que `page` está en 1-based,
        # tal y como lo definimos en DrivePyPDFLoader).
        print(documents[0].metadata)
        print("\n--- Contenido (primeros 200 caracteres) ---")
        # `[:200]` es slicing de string: cortamos a 200
        # caracteres para no inundar la consola con texto.
        # `+ "..."` es cosmético: indica que hay más.
        print(documents[0].page_content[:200] + "...")

except Exception as e:
    # Captura genérica. En producción convendría afinar
    # por tipo de error (OAuth, red, parseo…), pero para
    # un script didáctico está bien: asegura que un fallo
    # se lea como mensaje y no como traceback gigante.
    print(f"❌ Error al cargar documentos: {e}")


# ============================================================
# 💡 SUGERENCIAS (no aplicadas — el código original se respeta)
# ============================================================
# 1) `**kwargs` en `__init__` se acepta pero no se usa. Yo
#    lo dejaría igual (es intencional: compatibilidad con
#    lo que GoogleDriveLoader espere pasar). Solo lo
#    documento.  No lo aplico.
#
# 2) La `metadata` de `DrivePyPDFLoader` usa `{"page": i+1}`
#    (1-based), mientras que en 3.1 `PyPDFLoader` usaba
#    0-based. Conviene elegir una convención y documentarla.
#    No lo aplico: cambiaría el comportamiento observable.
#
# 3) El `except Exception` es muy amplio. Podría dividirse
#    en `except (google.auth.exceptions.RefreshError,
#    FileNotFoundError, ...)`. No lo aplico.
#
# 4) `os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""`
#    a veces es más limpio `os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)`.
#    No lo aplico: cambiaría la semántica (vacío vs.
#    inexistente).
#
# 5) El ID de carpeta por defecto es un literal "opaco".
#    Mejor obligar a que venga del `.env`:
#        FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
#        if not FOLDER_ID: raise SystemExit(...)
#    No lo aplico.
#
# 6) No hay `if __name__ == "__main__":`. En un script
#    didáctico está bien; en producción convendría.
#    No lo aplico.