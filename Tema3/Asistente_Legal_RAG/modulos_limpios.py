# ==============================================================================
# modulos_limpios.py — hace que los archivos numerados se importen entre sí
# ==============================================================================
# Problema: los archivos de estudio se llaman "1.1_config_limpio.py", "1.2_prompts_limpio.py",
# etc. En Python un módulo cuyo nombre empieza por un número y contiene un punto NO se puede
# importar con "import" / "from ... import" (el punto se interpreta como separador de paquetes).
#
# Solución: este archivo registra un "buscador de módulos" que traduce los nombres que usa el
# código (config, prompts, rag_system) a los archivos numerados _limpio:
#
#     config      -> 1.1_config_limpio.py
#     prompts     -> 1.2_prompts_limpio.py
#     rag_system  -> 1.4_rag_system_limpio.py
#
# Así el código de 1.3, 1.4 y 1.5 conserva "from config import ...", "from prompts import *" y
# "from rag_system import ...", pero lo que se importa SIEMPRE son las versiones _limpio.
# Solo hay que escribir "import modulos_limpios" ANTES de esas líneas.
#
# Los archivos sin número (config.py, prompts.py, rag_system.py, app.py, build_vectorstore.py)
# no usan este archivo: se importan entre sí de la forma normal.
# ==============================================================================
import sys
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location
from pathlib import Path

CARPETA = Path(__file__).resolve().parent

EQUIVALENCIAS = {
    "config": "1.1_config_limpio.py",
    "prompts": "1.2_prompts_limpio.py",
    "rag_system": "1.4_rag_system_limpio.py",
}


class BuscadorLimpios(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        archivo = EQUIVALENCIAS.get(fullname)
        if archivo is None:
            return None
        return spec_from_file_location(fullname, CARPETA / archivo)


# Se coloca el primero de la lista para que tenga prioridad sobre config.py, prompts.py y
# rag_system.py (si existen en la misma carpeta). Solo se registra una vez.
if not any(isinstance(b, BuscadorLimpios) for b in sys.meta_path):
    sys.meta_path.insert(0, BuscadorLimpios())
