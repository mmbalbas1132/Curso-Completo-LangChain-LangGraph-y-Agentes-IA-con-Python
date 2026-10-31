# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
# Importamos BaseModel, la clase fundamental de Pydantic.
from pydantic import BaseModel


# ==============================================================================
# 2. DEFINICIÓN DEL MODELO (El "Esquema" de los datos)
# ==============================================================================
# ➕ Añadido: Heredar de BaseModel convierte esta clase en un "validador de datos".
# No solo define la estructura, sino que fuerza a que los datos cumplan las reglas.
class Usuario(BaseModel):
    id: int          # Exige un entero. Si llega un string "123", lo convertirá.
    nombre: str      # Exige un string.
    activo: bool = True  # Exige un booleano. Como tiene un '= True', es opcional.


# ==============================================================================
# 3. DATOS DE ENTRADA
# ==============================================================================
# ➕ Añadido: Fíjate en el truco: el 'id' es un STRING ("123"), no un entero.
# Y el campo 'activo' brilla por su ausencia.
data = {"id": "123", "nombre": "Ana"}


# ==============================================================================
# 4. INSTANCIACIÓN Y VALIDACIÓN (El "Filtro" de Pydantic)
# ==============================================================================
# ➕ Añadido: El operador `**data` desempaqueta el diccionario. Esto equivale
# a escribir: Usuario(id="123", nombre="Ana").
usuario = Usuario(**data)

# ¿Qué ha pasado aquí dentro?
# 1. Pydantic vio que 'id' era un string, pero se esperaba un int. Hizo la conversión.
# 2. Pydantic vio que 'activo' no estaba. Le asignó el valor por defecto (True).


# ==============================================================================
# 5. SERIALIZACIÓN (La "Salida" en formato JSON)
# ==============================================================================
# ➕ Añadido: model_dump_json() toma el objeto (ya validado) y lo convierte
# de vuelta a un string en formato JSON.
print(usuario.model_dump_json())
