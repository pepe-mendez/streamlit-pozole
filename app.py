import streamlit as st
import pandas as pd

from tabs import entender, limpiar, descriptiva, predictiva, prescriptiva, decisiones


# ----------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------

st.set_page_config(
    page_title="Análisis de Ventas en Pozolería",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análisis de Ventas en Pozolería 🍲   ")

st.markdown("""
### Pregunta inicial

**¿Qué clientes tienen mayor probabilidad de volver a comprar y qué tamaño de porción elegirán?**
""")


# ----------------------------------
# CARGAR DATASET
# ----------------------------------

df = pd.read_csv(
    "data/dataset_clientes_sinteticos_ventas_pozole.csv"
)


# ----------------------------------
# CREAR PESTAÑAS
# ----------------------------------

tab_entender, tab_limpiar, tab_descriptiva, tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs([
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones"
])


# ----------------------------------
# 1. ENTENDER LOS DATOS
# ----------------------------------

with tab_entender:
    entender.mostrar(df)


# ----------------------------------
# PESTAÑAS PENDIENTES
# ----------------------------------

with tab_limpiar:
    limpiar.mostrar(df)


with tab_descriptiva:
    descriptiva.mostrar(df)


with tab_predictiva:
    predictiva.mostrar(df)


with tab_prescriptiva:
    prescriptiva.mostrar(df)


with tab_decisiones:
    decisiones.mostrar(df)

# -------------------------------
# COLUMNAS
# -------------------------------

