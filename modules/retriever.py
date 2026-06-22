"""
retriever.py — Recuperación semántica sobre el índice FAISS

Responsabilidad: recibir una pregunta del usuario, convertirla
al mismo espacio vectorial que los documentos y devolver los
fragmentos más relevantes ordenados por similitud.

Flujo:
    pregunta
        → embedding (mismo modelo que en la ingesta)
        → búsqueda de los K vecinos más cercanos en FAISS
        → filtrado por umbral mínimo de similitud coseno
        → lista de ResultadoRecuperacion ordenada por relevancia
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@dataclass
class ResultadoRecuperacion:
    """
    Agrupa la información de un fragmento recuperado.

    Usar un dataclass hace el código más legible que trabajar
    con diccionarios o tuplas.
    """
    fragmento: str       # Texto del chunk recuperado
    fuente: str          # Nombre del archivo PDF de origen
    pagina: int          # Número de página dentro del PDF (base 0)
    puntuacion: float    # Similitud coseno (0.0 = ninguna, 1.0 = idéntico)


def recuperar_fragmentos(
    pregunta: str,
    vectorstore: FAISS,
    top_k: int = config.RETRIEVER_TOP_K,
    umbral: float = config.RETRIEVER_SCORE_THRESHOLD,
) -> list[ResultadoRecuperacion]:
    """
    Busca los fragmentos más relevantes para una pregunta.

    FAISS devuelve distancias L2. Como los vectores están normalizados,
    convertimos a similitud coseno con:
        similitud = 1 - (distancia² / 2)

    Args:
        pregunta:    Consulta del usuario.
        vectorstore: Índice FAISS ya cargado en memoria.
        top_k:       Número máximo de fragmentos a recuperar.
        umbral:      Puntuación mínima para incluir un fragmento.

    Returns:
        Lista de ResultadoRecuperacion de mayor a menor similitud.
        Puede estar vacía si ningún fragmento supera el umbral.
    """
    if not pregunta.strip():
        return []

    resultados_raw = vectorstore.similarity_search_with_score(
        query=pregunta,
        k=top_k,
    )

    resultados: list[ResultadoRecuperacion] = []

    for documento, distancia in resultados_raw:
        # Convertir distancia L2 a similitud coseno (rango 0–1)
        similitud = round(1 - (distancia ** 2) / 2, 4)

        if similitud < umbral:
            continue

        resultados.append(ResultadoRecuperacion(
            fragmento=documento.page_content,
            fuente=documento.metadata.get("source", "desconocida"),
            pagina=documento.metadata.get("page", 0),
            puntuacion=similitud,
        ))

    resultados.sort(key=lambda r: r.puntuacion, reverse=True)
    return resultados


def formatear_contexto(resultados: list[ResultadoRecuperacion]) -> str:
    """
    Convierte los fragmentos recuperados en un bloque de texto
    estructurado para incluir en el prompt del LLM.

    Cada fragmento va etiquetado con su fuente y página.

    Args:
        resultados: Lista de ResultadoRecuperacion.

    Returns:
        Texto con los fragmentos numerados y etiquetados.
        Cadena vacía si no hay resultados.
    """
    if not resultados:
        return ""

    bloques = []
    for i, r in enumerate(resultados, start=1):
        bloque = (
            f"[Fragmento {i} | Fuente: {r.fuente} | Página: {r.pagina + 1}]\n"
            f"{r.fragmento}"
        )
        bloques.append(bloque)

    return "\n\n".join(bloques)


# ----------------------------------------------------------------
# Prueba rápida: python modules/retriever.py
# ----------------------------------------------------------------
if __name__ == "__main__":
    from embeddings import load_vectorstore

    PREGUNTA = "¿Qué es el fondo de emergencia y cuánto dinero debería tener?"

    print(f"\n[Recuperación semántica]\nPregunta: {PREGUNTA}\n")

    vs = load_vectorstore()
    resultados = recuperar_fragmentos(PREGUNTA, vs)

    if not resultados:
        print("No se encontraron fragmentos por encima del umbral.")
        print(f"Umbral configurado: {config.RETRIEVER_SCORE_THRESHOLD}")
    else:
        print(f"Fragmentos recuperados: {len(resultados)}\n")
        print("-" * 60)
        for i, r in enumerate(resultados, start=1):
            print(f"Resultado {i}")
            print(f"  Fuente     : {r.fuente}")
            print(f"  Página     : {r.pagina + 1}")
            print(f"  Puntuación : {r.puntuacion:.4f}")
            print(f"  Fragmento  : {r.fragmento[:200]}...")
            print("-" * 60)

    print("\n[Contexto formateado para el prompt]")
    print(formatear_contexto(resultados))
