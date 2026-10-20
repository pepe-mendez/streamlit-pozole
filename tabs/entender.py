import streamlit as st
import pandas as pd


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("1. Entender los datos")

    # ----------------------------------
    # DIMENSIONES DEL DATASET
    # ----------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Número de clientes",
            len(df)
        )

    with col2:
        st.metric(
            "Número de variables",
            len(df.columns)
        )


    # ----------------------------------
    # VISTA PREVIA
    # ----------------------------------

    st.subheader("Vista previa del dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # ----------------------------------
    # DIAGNÓSTICO
    # ----------------------------------

    st.subheader("Diagnóstico del dataset")

    diagnostico = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values,
        "Valores faltantes": df.isnull().sum().values
    })

    st.dataframe(
        diagnostico,
        use_container_width=True
    )


    # ----------------------------------
    # DUPLICADOS
    # ----------------------------------

    st.subheader("Problemas potenciales")

    duplicados = df.duplicated().sum()

    st.metric(
        "Registros duplicados",
        duplicados
    )


    # ----------------------------------
    # EXPLORADOR DE VARIABLES
    # ----------------------------------

    st.subheader("Explorar una variable")

    variable = st.selectbox(
        "Selecciona una variable:",
        df.columns,
        key="variable_entender"
    )


    # ----------------------------------
    # VARIABLE NUMÉRICA
    # ----------------------------------

    if pd.api.types.is_numeric_dtype(df[variable]):

        st.write("**Tipo:** Variable numérica")

        estadisticas = df[variable].describe()

        st.dataframe(
            estadisticas,
            use_container_width=True
        )


    # ----------------------------------
    # VARIABLE CATEGÓRICA
    # ----------------------------------

    else:

        st.write("**Tipo:** Variable categórica")

        frecuencias = (
            df[variable]
            .value_counts(dropna=False)
            .reset_index()
        )

        frecuencias.columns = [
            "Valor",
            "Frecuencia"
        ]

        st.dataframe(
            frecuencias,
            use_container_width=True
        )