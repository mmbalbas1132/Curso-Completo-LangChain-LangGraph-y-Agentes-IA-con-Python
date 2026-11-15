import pdfplumber
from io import BytesIO


def extraer_texto_pdf(archivo_pdf):
    try:
        # Leemos el archivo una sola vez
        pdf_bytes = archivo_pdf.read()
        texto_completo = ""

        # Usamos pdfplumber (mejor para tablas y layouts complejos que PyPDF2)
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for numero_pagina, pagina in enumerate(pdf.pages, 1):
                texto_pagina = pagina.extract_text()
                if texto_pagina and texto_pagina.strip():
                    texto_completo += f"\n--- PÁGINA {numero_pagina} ---\n"
                    texto_completo += texto_pagina + "\n"

        texto_completo = texto_completo.strip()

        if not texto_completo:
            # Lanzamos excepción en lugar de devolver un string
            raise ValueError(
                "El PDF parece estar vacío o contener solo imágenes. Asegúrate de que el texto sea seleccionable.")

        return texto_completo

    except Exception as e:
        # Re-lanzamos la excepción para que la UI la capture
        raise Exception(f"Error al procesar el archivo PDF: {str(e)}")