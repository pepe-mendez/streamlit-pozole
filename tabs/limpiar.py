"""Herramientas para revisar y preparar el dataset de clientes."""

import pandas as pd
import streamlit as st


def _tabla_faltantes(df):
    resumen = pd.DataFrame(
        {
            "Variable": df.columns,
            "Valores faltantes": df.isna().sum().values,
            "Porcentaje faltante": (df.isna().mean().values * 100).round(2),
        }
    )
    return resumen[resumen["Valores faltantes"] > 0].sort_values(
        "Valores faltantes", ascending=False
    )


def _revisar_rangos(df):
    """Señala valores negativos en medidas que deberían ser no negativas."""
    columnas_no_negativas = [
        "edad",
        "ingreso_mensual",
        "visitas_restaurante_30d",
        "compras_12m",
        "gasto_total_12m",
        "dias_desde_ultima_compra",
        "monto_ultima_compra",
    ]
    hallazgos = []
    for columna in columnas_no_negativas:
        if columna in df.columns and pd.api.types.is_numeric_dtype(df[columna]):
            cantidad = int((df[columna] < 0).sum())
            if cantidad:
                hallazgos.append(
                    {"Variable": columna, "Valores fuera de rango": cantidad}
                )
    return pd.DataFrame(hallazgos, columns=["Variable", "Valores fuera de rango"])


def mostrar(df):
    st.header("2. Limpiar los datos")
    st.markdown(
        "Revisa la calidad de los registros y decide qué preparar. "
        "Los cambios se aplican a una copia; el dataset original permanece intacto."
    )

    duplicados = int(df.duplicated().sum())
    faltantes = int(df.isna().sum().sum())
    columnas_texto = int(
        sum(
            not pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_datetime64_any_dtype(df[col])
            for col in df.columns
        )
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", f"{len(df):,}")
    col2.metric("Valores faltantes", f"{faltantes:,}")
    col3.metric("Filas duplicadas", f"{duplicados:,}")
    col4.metric("Columnas de texto", f"{columnas_texto:,}")

    st.subheader("1. Valores faltantes")
    tabla_faltantes = _tabla_faltantes(df)
    if tabla_faltantes.empty:
        st.success("No se encontraron valores faltantes.")
    else:
        st.dataframe(tabla_faltantes, hide_index=True, use_container_width=True)
        st.info(
            "Antes de eliminar o completar valores, confirma por qué faltan y qué "
            "significan en el negocio."
        )

    st.subheader("2. Duplicados y rangos")
    col_duplicados, col_rangos = st.columns(2)
    with col_duplicados:
        st.write("**Filas idénticas**")
        if duplicados:
            st.warning(f"Hay {duplicados:,} filas duplicadas exactas.")
        else:
            st.success("No hay filas duplicadas exactas.")
    with col_rangos:
        st.write("**Valores negativos en medidas**")
        tabla_rangos = _revisar_rangos(df)
        if tabla_rangos.empty:
            st.success("No se detectaron valores negativos en las medidas revisadas.")
        else:
            st.dataframe(tabla_rangos, hide_index=True, use_container_width=True)
            st.caption(
                "Son alertas para investigar; no se eliminan ni corrigen automáticamente."
            )

    st.subheader("3. Preparar una copia")
    st.caption(
        "Selecciona únicamente cambios justificados. La tabla y la descarga muestran "
        "la copia resultante de esta sesión."
    )
    eliminar_duplicados = st.checkbox(
        "Eliminar filas duplicadas exactas de la copia",
        value=False,
        disabled=duplicados == 0,
        key="limpiar_eliminar_duplicados",
    )

    posibles_fechas = [
        col
        for col in df.columns
        if "fecha" in str(col).lower() and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    convertir_fecha = False
    columna_fecha = None
    if posibles_fechas:
        columna_fecha = st.selectbox(
            "Columna de fecha para convertir",
            posibles_fechas,
            key="limpiar_columna_fecha",
        )
        convertir_fecha = st.checkbox(
            "Convertir esta columna a fecha (día/mes/año)",
            value=False,
            key="limpiar_convertir_fecha",
        )

    df_limpio = df.copy()
    cambios = []
    if eliminar_duplicados:
        filas_antes = len(df_limpio)
        df_limpio = df_limpio.drop_duplicates().copy()
        cambios.append(f"Se quitaron {filas_antes - len(df_limpio):,} filas duplicadas.")
    if convertir_fecha and columna_fecha:
        fecha_convertida = pd.to_datetime(
            df_limpio[columna_fecha], format="%d/%m/%y", errors="coerce"
        )
        no_convertidas = int((fecha_convertida.isna() & df_limpio[columna_fecha].notna()).sum())
        df_limpio[columna_fecha] = fecha_convertida
        cambios.append(
            f"Se convirtió `{columna_fecha}` a fecha; {no_convertidas:,} valores no se pudieron interpretar."
        )

    if cambios:
        for cambio in cambios:
            st.write(f"- {cambio}")
    else:
        st.info("Aún no has seleccionado transformaciones para la copia.")

    st.subheader("Vista previa de la copia")
    st.dataframe(df_limpio.head(10), hide_index=True, use_container_width=True)
    st.download_button(
        "Descargar copia preparada (CSV)",
        data=df_limpio.to_csv(index=False).encode("utf-8-sig"),
        file_name="clientes_pozole_preparado.csv",
        mime="text/csv",
        key="limpiar_descargar_csv",
    )
