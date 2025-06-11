import streamlit as st

st.set_page_config(page_title="ClaudeAcadémico Dashboard", layout="wide")

st.title("📊 Dashboard ClaudeAcadémico")
st.markdown("Bienvenido al panel de monitoreo del sistema de traducción académica.")

with st.expander("ℹ️ Estado del sistema"):
    st.success("✅ FastAPI Backend operativo")
    st.info("🚧 Integraciones en desarrollo")

st.metric("Traducciones procesadas", "0", delta="+0 desde ayer")
st.metric("Sugerencias terminológicas", "0", delta="+0 desde ayer")
