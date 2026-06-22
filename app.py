"""
app.py — Interfaz Streamlit del chatbot de educación financiera

Punto de entrada de la aplicación. Gestiona la sesión del usuario,
el historial de conversación y delega toda la lógica al pipeline RAG.

Uso:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from modules.embeddings import load_vectorstore
from modules.rag_pipeline import ejecutar_pipeline
import config


# ---------------------------------------------------------------------------
# Configuración de la página — debe ser la primera llamada a Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Educación Financiera",
    page_icon="💶",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Carga del índice vectorial — una sola vez por sesión
# Se almacena en st.session_state para no recargarlo en cada interacción
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def cargar_vectorstore():
    """
    Carga el índice FAISS en memoria.
    @st.cache_resource garantiza que esto ocurre una sola vez
    aunque el usuario haga múltiples preguntas.
    """
    return load_vectorstore()


# ---------------------------------------------------------------------------
# Inicialización del estado de la sesión
# ---------------------------------------------------------------------------

if "historial" not in st.session_state:
    # Lista de mensajes: cada elemento es {"role": "user"|"assistant", "content": str}
    st.session_state.historial = []

if "fuentes_ultima_respuesta" not in st.session_state:
    st.session_state.fuentes_ultima_respuesta = []


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuración")

    nivel = st.selectbox(
        "Nivel de explicación",
        options=["basico", "medio", "avanzado"],
        index=1,                          # "medio" por defecto
        format_func=lambda x: x.capitalize(),
        help="Ajusta el lenguaje de las respuestas a tu nivel de conocimiento financiero.",
    )

    st.divider()

    # Fuentes de la última respuesta
    st.subheader("Fuentes utilizadas")
    if st.session_state.fuentes_ultima_respuesta:
        for f in st.session_state.fuentes_ultima_respuesta:
            st.caption(
                f"📄 {f.fuente} · p. {f.pagina + 1} · "
                f"similitud: {f.puntuacion:.2f}"
            )
    else:
        st.caption("Las fuentes documentales aparecerán aquí tras cada respuesta.")

    st.divider()

    # Aviso educativo fijo — presente en toda la sesión
    st.info(
        "**Aviso importante**\n\n"
        "Este chatbot tiene finalidad **exclusivamente educativa**. "
        "Las respuestas se basan en documentación institucional pública "
        "y **no constituyen asesoramiento financiero personalizado**.\n\n"
        "Para decisiones sobre tu dinero, consulta a un asesor financiero certificado.",
        icon="ℹ️",
    )

    # Botón para limpiar el historial
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.historial = []
        st.session_state.fuentes_ultima_respuesta = []
        st.rerun()


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------

st.title("💶 Educación Financiera")
st.caption("Resuelve tus dudas sobre finanzas personales con información de fuentes institucionales.")

# Cargar el vectorstore (desde caché tras la primera carga)
try:
    vectorstore = cargar_vectorstore()
except FileNotFoundError:
    st.error(
        "No se encontró el índice vectorial. "
        "Genera el índice ejecutando:\n\n"
        "```bash\npython modules/embeddings.py\n```"
    )
    st.stop()

# Mostrar el historial de conversación
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# ---------------------------------------------------------------------------
# Campo de entrada y procesamiento de la pregunta
# ---------------------------------------------------------------------------

pregunta = st.chat_input("Escribe tu pregunta sobre finanzas personales...")

if pregunta:
    # Mostrar la pregunta del usuario en el chat
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Guardar en el historial
    st.session_state.historial.append({"role": "user", "content": pregunta})

    # Llamar al pipeline y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            resultado = ejecutar_pipeline(pregunta, vectorstore, nivel)

        st.markdown(resultado.respuesta)

        # Guardar fuentes para mostrarlas en la barra lateral
        st.session_state.fuentes_ultima_respuesta = resultado.fuentes

    # Guardar la respuesta en el historial
    st.session_state.historial.append({
        "role": "assistant",
        "content": resultado.respuesta,
    })

    # Refrescar la barra lateral con las nuevas fuentes
    st.rerun()
