import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader


# Cargador ligero basado en pypdf (1 Document por página)
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


# 1. Cargar variables de entorno (buscando el .env en la raíz del proyecto)
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")

# 2. Forzar a LangChain a usar OAuth (ignorar credenciales por defecto de la nube)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# 3. Definir rutas relativas al propio script
current_dir = Path(__file__).resolve().parent
CREDENTIALS_PATH = current_dir / "credentials.json"
TOKEN_PATH = current_dir / "token.json"

# 4. Obtener el ID de la carpeta desde .env (o usar el valor por defecto)
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "18I97qr1prZMAzj_3YpB3L2yIboH7FHHd")

# 5. Configurar el loader con permiso de lectura completa (drive.readonly)
loader = GoogleDriveLoader(
    folder_id=FOLDER_ID,
    credentials_path=str(CREDENTIALS_PATH),
    token_path=str(TOKEN_PATH),
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
    recursive=True,
    file_types=["pdf"],
    file_loader_cls=DrivePyPDFLoader
)

# 6. Cargar los documentos
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