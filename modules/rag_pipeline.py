import re
from dataclasses import dataclass

from modules.retriever import recuperar_fragmentos, formatear_contexto
from modules.prompt_builder import construir_prompt
from modules.llm_client import llamar_llm


# ---------------------------------------------------------------------------
# Filtro de asesoramiento personalizado
# ---------------------------------------------------------------------------
#
# Detecta consultas que solicitan recomendaciones personalizadas antes de
# llegar al LLM. Si la consulta coincide, se devuelve una respuesta fija
# sin invocar al modelo ni al retriever.
#
# Este enfoque es más fiable que confiar únicamente en las instrucciones
# del prompt, ya que el LLM tiende a usar los fragmentos recuperados para
# construir orientaciones implícitas aunque se le indique que no lo haga.

_PATRONES_ASESORAMIENTO = [
    # Preguntas directas sobre qué hacer con el dinero
    r"\b(en qu[eé]|qu[eé]).*(deb[oe]r[ií]a|conviene|recomend|mejor|aconsej).*(invert|ahorr|poner|meter|coloc)",
    r"\b(dónde|donde|cómo|como|cuál|cual).*(invert|ahorr|poner|meter|coloc).*(dinero|ahorros|capital|euros)",
    r"\b(me\s+)?recomend.*(producto|fondo|acci[oó]n|dep[oó]sito|plan|invers)",
    r"\bqu[eé]\s+(producto|fondo|acci[oó]n|dep[oó]sito|plan)\s+(me\s+)?(recomiend|conviene|deb[eé]r[ií]a)",
    # Peticiones directas de consejo
    r"\b(aconséjame|aconse[jg]a|dime\s+qu[eé]\s+hac|qu[eé]\s+har[ií]as\s+t[uú])",
    r"\b(qu[eé]|cómo)\s+(debo|debería|puedo|podría)\s+(hacer|invertir|ahorrar)\s+(con|mis|el)",
    # Preguntas sobre acciones o inversiones específicas
    r"\ben\s+qu[eé]\s+(acciones?|fondos?|valores?|activos?)\s+(debo|debería|conviene|recomend)",
    r"\b(comprar|vender|contratar|suscribir)\s+(acciones?|fondos?|valores?|productos?)",
    # Dónde meter/poner los ahorros
    r"\b(d[oó]nde|con qu[eé])\s+(meto|pongo|coloco|guardo|deposito)\s+(mis|el|los|este)\s+(dinero|ahorros|capital)",
    # "me conviene / me interesa / merece la pena contratar X"
    r"\b(me\s+)?(conviene|interesa|merece\s+la\s+pena)\s+(contratar|abrir|pedir|solicitar|suscribir)",
    # "qué me interesa más" / "qué me conviene más" / "qué me sale mejor"
    r"\bqu[eé]\s+me\s+(interesa|conviene|viene\s+mejor|sale\s+mejor)",
]

_RESPUESTA_ASESORAMIENTO = (
    "Mi función es educativa, así que no puedo recomendarte en qué invertir "
    "ni orientarte sobre qué hacer con tu dinero en tu caso concreto. "
    "Esa decisión depende de tu situación personal, tus objetivos y tu tolerancia "
    "al riesgo, y requiere un asesor financiero certificado que te conozca.\n\n"
    "Lo que sí puedo hacer es explicarte cómo funcionan los conceptos relacionados: "
    "qué es la renta variable, qué implica el riesgo financiero, qué significa "
    "diversificar o cómo comparar productos de ahorro e inversión. "
    "¿Quieres que profundice en alguno de esos conceptos?"
)


def _es_consulta_asesoramiento(pregunta: str) -> bool:
    """
    Devuelve True si la pregunta solicita asesoramiento financiero personalizado.
    La comparación es insensible a mayúsculas/minúsculas.
    """
    texto = pregunta.lower()
    return any(re.search(patron, texto) for patron in _PATRONES_ASESORAMIENTO)


# ---------------------------------------------------------------------------
# Dataclass de respuesta
# ---------------------------------------------------------------------------

@dataclass
class RespuestaRAG:
    respuesta: str
    fuentes: list


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def ejecutar_pipeline(
    pregunta: str,
    vectorstore,
    nivel_usuario: str = "BASICO",
) -> RespuestaRAG:
    """
    Orquesta el flujo completo de una consulta RAG.

    Flujo:
        1. Filtro de asesoramiento: si la consulta pide recomendación
           personalizada, se devuelve respuesta fija sin invocar al LLM.
        2. Recuperación semántica sobre el índice FAISS.
        3. Construcción del prompt con contexto y nivel de usuario.
        4. Generación de respuesta mediante el LLM.

    Args:
        pregunta:     Consulta del usuario.
        vectorstore:  Índice FAISS cargado en memoria.
        nivel_usuario: Nivel declarado: "basico", "medio" o "avanzado".

    Returns:
        RespuestaRAG con la respuesta generada y las fuentes utilizadas.
    """

    # — Paso 1: filtro previo de asesoramiento personalizado —
    if _es_consulta_asesoramiento(pregunta):
        return RespuestaRAG(
            respuesta=_RESPUESTA_ASESORAMIENTO,
            fuentes=[],
        )

    # — Paso 2: recuperación semántica —
    resultados = recuperar_fragmentos(
        pregunta=pregunta,
        vectorstore=vectorstore,
    )

    # — Paso 3: construcción del prompt —
    contexto = formatear_contexto(resultados)
    mensajes = construir_prompt(
        pregunta=pregunta,
        contexto=contexto,
        nivel=nivel_usuario,
    )

    # — Paso 4: generación de respuesta —
    respuesta = llamar_llm(mensajes)

    return RespuestaRAG(
        respuesta=respuesta,
        fuentes=resultados,
    )
