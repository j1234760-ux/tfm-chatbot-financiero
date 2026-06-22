"""
config.py — Parámetros globales del proyecto

Lee las variables del archivo .env y las expone como constantes
para que el resto de módulos las importen desde aquí.

Nota sobre reproducibilidad:
    El modelo LLM se fija a una versión concreta (gpt-4o-2024-08-06)
    para garantizar que los resultados sean reproducibles con independencia
    de las actualizaciones que el proveedor aplique al alias genérico.
"""

import os
from dotenv import load_dotenv

# Carga el archivo .env en las variables de entorno del proceso
load_dotenv()

# --- API de OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Versión fija del modelo para garantizar reproducibilidad.
# No usar alias genéricos como "gpt-4o", ya que el proveedor puede
# actualizar el modelo subyacente sin previo aviso.
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")

# --- Parámetros del LLM ---
# Temperatura baja para favorecer consistencia sobre variedad expresiva,
# criterio adecuado en un sistema educativo donde la coherencia importa.
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Rutas del proyecto ---
DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "data/documents")
VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

# --- Modelo de embeddings ---
# Modelo multilingüe optimizado para similitud semántica en español.
# Ejecutable en CPU sin infraestructura GPU dedicada.
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Parámetros de chunking ---
# chunk_size=600 y overlap=100 establecidos empíricamente:
# valores mayores introducían ruido documental en la recuperación;
# valores menores perdían el contexto semántico del fragmento.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Parámetros de recuperación ---
# top_k=5: equilibrio entre exhaustividad y precisión para el tamaño
# actual del corpus (~40 documentos).
# score_threshold=0.4: umbral de similitud coseno obtenido tras convertir
# la distancia L2 de FAISS mediante similitud = 1 - (d^2 / 2).
# Fragmentos por debajo de este valor raramente aportaban información
# útil durante las pruebas del prototipo.
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_SCORE_THRESHOLD: float = float(
    os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.4")
)


def validate_config() -> None:
    """
    Comprueba que las variables críticas están configuradas.
    Se llama al arrancar la aplicación.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY no está configurada. "
            "Copia .env.example a .env y añade tu clave de API."
        )
