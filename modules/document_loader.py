"""
document_loader.py — Ingesta y preparación de documentos PDF

Responsabilidad: leer los PDFs de la carpeta de documentos,
limpiar el texto extraído y dividirlo en fragmentos (chunks)
manejables para el sistema de recuperación semántica.

Flujo:
    carpeta data/documents/
        → extracción de texto por página (pypdf)
        → limpieza básica del texto
        → división en chunks con solapamiento
        → lista de Document con metadatos
"""

import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Añadir la raíz del proyecto al path para importar config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def clean_text(text: str) -> str:
    """
    Limpia el texto extraído de un PDF.

    Elimina artefactos habituales en PDFs institucionales:
    saltos de línea internos en párrafos, espacios múltiples
    y líneas muy cortas que suelen ser encabezados o pies de página.

    Args:
        text: Texto en bruto extraído del PDF.

    Returns:
        Texto limpio listo para ser dividido en chunks.
    """
    # Sustituye saltos de línea simples por espacios
    # (los párrafos PDF suelen partir líneas en mitad de la frase)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Colapsa espacios múltiples en uno solo
    text = re.sub(r" {2,}", " ", text)

    # Elimina líneas muy cortas (< 4 caracteres): números de página,
    # encabezados repetitivos, pies de firma, etc.
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) >= 4]
    text = "\n".join(lines)

    return text.strip()


def load_documents(documents_path: str = config.DOCUMENTS_PATH):
    """
    Carga todos los PDFs de la carpeta indicada y devuelve
    una lista de Document de LangChain, uno por página.

    Cada Document incluye metadatos:
        - source: nombre del archivo PDF
        - page:   número de página (base 0)

    Args:
        documents_path: Ruta a la carpeta con los PDFs.

    Returns:
        Lista de Document con el texto por página.
    """
    path = Path(documents_path)

    if not path.exists():
        raise FileNotFoundError(f"Carpeta de documentos no encontrada: {path}")

    pdf_files = sorted(path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No se encontraron archivos PDF en: {path}")

    all_pages: list[Document] = []

    for pdf_file in pdf_files:
        print(f"  Cargando: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.page_content = clean_text(page.page_content)

            # Ignorar páginas vacías tras la limpieza
            if len(page.page_content.strip()) < 50:
                continue

            page.metadata["source"] = pdf_file.name
            all_pages.append(page)

    print(f"  Total de páginas cargadas: {len(all_pages)}")
    return all_pages


def split_documents(
    pages,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """
    Divide las páginas en fragmentos (chunks) más pequeños.

    Usa RecursiveCharacterTextSplitter, que respeta la estructura
    del texto: corta primero por párrafos, luego por frases,
    y como último recurso por caracteres.

    El solapamiento entre chunks garantiza que el contexto
    no se pierda en los bordes del corte.

    Args:
        pages:         Lista de Document por página.
        chunk_size:    Tamaño máximo de cada fragmento en caracteres.
        chunk_overlap: Caracteres que se repiten entre chunks consecutivos.

    Returns:
        Lista de Document con los chunks y sus metadatos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(pages)

    print(f"  Total de chunks generados: {len(chunks)}")
    print(f"  Tamaño: {chunk_size} chars | Solapamiento: {chunk_overlap} chars")

    return chunks


def ingest_documents(documents_path: str = config.DOCUMENTS_PATH):
    """
    Función principal de ingesta: carga → limpieza → chunking.

    Es el punto de entrada que usan los demás módulos.

    Args:
        documents_path: Ruta a la carpeta con los PDFs.

    Returns:
        Lista de chunks listos para generar embeddings.
    """
    print("\n[Ingesta documental]")
    pages = load_documents(documents_path)
    chunks = split_documents(pages)
    print(f"  Ingesta completada: {len(chunks)} fragmentos listos.\n")
    return chunks


# ----------------------------------------------------------------
# Prueba rápida: python modules/document_loader.py
# ----------------------------------------------------------------
if __name__ == "__main__":
    chunks = ingest_documents()

    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Fuente  : {chunk.metadata.get('source', 'desconocida')}")
        print(f"Página  : {chunk.metadata.get('page', '?')}")
        print(f"Longitud: {len(chunk.page_content)} caracteres")
        print(f"Texto   : {chunk.page_content[:300]}...")
