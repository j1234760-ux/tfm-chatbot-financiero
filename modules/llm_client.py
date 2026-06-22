"""
llm_client.py — Cliente para la API de OpenAI

Responsabilidad: recibir los mensajes construidos por prompt_builder,
enviarlos al modelo y devolver la respuesta como texto limpio.

Es el único módulo que realiza llamadas externas de red.
Todo lo demás (embeddings, FAISS) funciona en local.
"""

import sys
from pathlib import Path

from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def get_client() -> OpenAI:
    """
    Crea y devuelve el cliente de OpenAI.

    Returns:
        Cliente OpenAI configurado con la clave de API.

    Raises:
        ValueError: Si la clave de API no está configurada.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY no está configurada. Revisa tu archivo .env."
        )
    return OpenAI(api_key=config.OPENAI_API_KEY)


def llamar_llm(mensajes: list[dict]) -> str:
    """
    Envía los mensajes al modelo y devuelve la respuesta como texto.

    En caso de error devuelve un mensaje explicativo en lugar de
    lanzar una excepción, para que la interfaz pueda mostrarlo
    directamente al usuario.

    Args:
        mensajes: Lista en formato OpenAI:
                  [{"role": "system", "content": ...},
                   {"role": "user",   "content": ...}]

    Returns:
        Texto de la respuesta del modelo, o mensaje de error.
    """
    try:
        client = get_client()

        respuesta = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=mensajes,
            temperature=config.LLM_TEMPERATURE,   # 0.3: coherente sin ser rígido
            max_tokens=config.LLM_MAX_TOKENS,      # 1024: suficiente para respuestas educativas
        )

        return respuesta.choices[0].message.content.strip()

    except AuthenticationError:
        return (
            "Error de autenticación con OpenAI. "
            "Comprueba que tu OPENAI_API_KEY es válida y está activa."
        )
    except RateLimitError:
        return (
            "Se ha superado el límite de solicitudes de la API. "
            "Espera unos segundos y vuelve a intentarlo."
        )
    except APIConnectionError:
        return (
            "No se ha podido conectar con la API de OpenAI. "
            "Comprueba tu conexión a internet."
        )
    except Exception as e:
        return f"Error inesperado al llamar al modelo: {str(e)}"


# ----------------------------------------------------------------
# Prueba rápida: python modules/llm_client.py
# Requiere OPENAI_API_KEY configurada en .env
# ----------------------------------------------------------------
if __name__ == "__main__":
    from prompt_builder import construir_prompt

    contexto_prueba = (
        "[Fragmento 1 | Fuente: guia_ahorro.pdf | Página: 7]\n"
        "El fondo de emergencia es una reserva de dinero líquido destinada a cubrir "
        "gastos imprevistos. Los expertos recomiendan entre tres y seis meses de gastos fijos.\n\n"
        "[Fragmento 2 | Fuente: guia_ahorro.pdf | Página: 8]\n"
        "Este fondo debe mantenerse en un producto de alta liquidez."
    )

    pregunta_prueba = "¿Qué es un fondo de emergencia y cuánto debería tener?"

    print(f"\n[Prueba llm_client]\nPregunta: {pregunta_prueba}\n")

    for nivel in ["basico", "avanzado"]:
        print(f"--- Nivel: {nivel.upper()} ---")
        mensajes = construir_prompt(pregunta_prueba, contexto_prueba, nivel)
        respuesta = llamar_llm(mensajes)
        print(respuesta)
        print()
