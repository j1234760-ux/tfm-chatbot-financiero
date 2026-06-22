"""
embeddings.py — Generación del índice vectorial con FAISS

Responsabilidad: convertir los chunks de texto en vectores numéricos
(embeddings) y construir un índice FAISS que permita búsquedas
semánticas rápidas.

Flujo (una sola vez):
    chunks → sentence-transformers → vectores → índice FAISS → disco

Flujo (en cada consulta):
    índice cargado desde disco → búsqueda en memoria
"""

import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Carga el modelo de embeddings desde HuggingFace.

    paraphrase-multilingual-MiniLM-L12-v2 es compacto (~120 MB),
    rápido en CPU y tiene buen rendimiento en español.
    Se descarga automáticamente la primera vez y queda en caché.

    Returns:
        Modelo de embeddings listo para usar.
    """
    print(f"  Cargando modelo de embeddings: {config.EMBEDDING_MODEL}")

    model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # Normalizar facilita el cálculo de similitud coseno
        encode_kwargs={"normalize_embeddings": True},
    )

    return model


def build_vectorstore(chunks=None) -> FAISS:
    """
    Construye el índice FAISS y lo guarda en disco.

    Llama a esta función una sola vez, o cuando añadas nuevos documentos.

    Args:
        chunks: Fragmentos de texto. Si es None, ejecuta la ingesta
                automáticamente desde data/documents/.

    Returns:
        Índice FAISS cargado en memoria.
    """
    from document_loader import ingest_documents

    print("\n[Construcción del índice FAISS]")

    if chunks is None:
        chunks = ingest_documents()

    embedding_model = get_embedding_model()

    print(f"  Generando embeddings para {len(chunks)} chunks...")
    print("  (Puede tardar unos minutos la primera vez)")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    vectorstore_path = config.VECTORSTORE_PATH
    Path(vectorstore_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(vectorstore_path)

    print(f"  Índice guardado en: {vectorstore_path}")
    print(f"  Vectores indexados: {vectorstore.index.ntotal}\n")

    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Carga el índice FAISS desde disco.

    Es la función que usa el pipeline RAG en cada consulta.
    No regenera embeddings: lee el índice ya construido.

    Returns:
        Índice FAISS cargado en memoria.

    Raises:
        FileNotFoundError: Si el índice no existe.
                           Solución: ejecutar build_vectorstore() primero.
    """
    vectorstore_path = config.VECTORSTORE_PATH
    index_file = Path(vectorstore_path) / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Índice FAISS no encontrado en: {vectorstore_path}\n"
            "Genera el índice ejecutando:\n"
            "  python modules/embeddings.py"
        )

    embedding_model = get_embedding_model()

    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    print(f"  Índice cargado: {vectorstore.index.ntotal} vectores")
    return vectorstore


# ----------------------------------------------------------------
# Prueba rápida: python modules/embeddings.py
# ----------------------------------------------------------------
if __name__ == "__main__":
    build_vectorstore()
    print("Verificando carga del índice...")
    vs = load_vectorstore()
    print(f"OK — {vs.index.ntotal} vectores disponibles para búsqueda.")
