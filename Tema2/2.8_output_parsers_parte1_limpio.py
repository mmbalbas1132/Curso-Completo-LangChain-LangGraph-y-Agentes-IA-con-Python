"""
Ejemplo de validación y serialización con Pydantic.
"""

from pydantic import BaseModel


class Usuario(BaseModel):
    id: int
    nombre: str
    activo: bool = True


if __name__ == "__main__":
    data = {"id": "123", "nombre": "Ana"}
    usuario = Usuario(**data)

    print(usuario.model_dump_json())