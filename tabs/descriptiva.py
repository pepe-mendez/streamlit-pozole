"""Indicadores descriptivos para explorar las ventas de la pozolería."""

import pandas as pd
import streamlit as st


def _serie_numerica(df, columna):
    """Devuelve una columna numérica cuando existe; en otro caso, una serie vacía."""
    if columna not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _resumen_por_grupo(df, grupo):
    datos = df.copy()
    datos["_compra_semanal"] = _serie_numerica(datos, "compro_ultima_semana")
    datos["_gasto_12m"] = _serie_numerica(datos, "gasto_total_12m")
    datos["_compras_12m"] = _serie_numerica(datos, "compras_12m")
    datos["_visitas_30d"] = _serie_numerica(datos, "visitas_restaurante_30d")

    resumen = (
        datos.groupby(grupo, dropna=False)
        .agg(
            clientes=("cliente_id", "size") if "cliente_id" in datos.columns else (grupo, "size"),
            compradores_semana=("_compra_semanal", "sum"),
            gasto_promedio_12m=("_gasto_12m", "mean"),
            compras_promedio_12m=("_compras_12m", "mean"),
            visitas_promedio_30d=("_visitas_30d", "mean"),
        )
        .reset_index()
    )
    resumen["Tasa compra semanal (%)"] = (
        resumen["compradores_semana"] / resumen["clientes"] * 100
    )
    resumen = resumen.rename(
        columns={
            grupo: "Grupo",
            "gasto_promedio_12m": "Gasto promedio por cliente (12 meses)",
            "compras_promedio_12m": "Compras promedio por cliente (12 meses)",
            "visitas_promedio_30d": "Visitas promedio por cliente (30 días)",
        }
    )
    return resumen[
        [
            "Grupo",
            "clientes",
            "compradores_semana",
            "Tasa compra semanal (%)",
            "Gasto promedio por cliente (12 meses)",
            "Compras promedio por cliente (12 meses)",
            "Visitas promedio por cliente (30 días)",
        ]
    ].rename(
        columns={"clientes": "Clientes", "compradores_semana": "Compraron última semana"}
    )


def mostrar(df):
    st.header("3. Estadística descriptiva")
    st.markdown(
        "Resume el comportamiento observado de los clientes. Los indicadores describen "
        "este dataset y no prueban que una variable cause otra."
    )

    if df.empty:
        st.warning("No hay registros para calcular indicadores.")
        return

    n_clientes = len(df)
    compra = _serie_numerica(df, "compro_ultima_semana")
    gasto = _serie_numerica(df, "gasto_total_12m")
    compras = _serie_numerica(df, "compras_12m")
    ultima_compra = _serie_numerica(df, "monto_ultima_compra")

    compradores = int(compra.eq(1).sum())
    tasa_compra = compradores / n_clientes * 100 if n_clientes else 0
    gasto_promedio = gasto.mean()
    compras_promedio = compras.mean()
    ultima_compra_promedio = ultima_compra.mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Clientes", f"{n_clientes:,}")
    m2.metric("Compraron la última semana", f"{compradores:,}", f"{tasa_compra:.1f}% de clientes")
    m3.metric(
        "Compras promedio por cliente (12 meses)",
        f"{compras_promedio:.1f}" if pd.notna(compras_promedio) else "Sin dato",
    )

    m4, m5, m6 = st.columns(3)
    m4.metric(
        "Gasto promedio por cliente (12 meses)",
        f"{gasto_promedio:,.2f}" if pd.notna(gasto_promedio) else "Sin dato",
    )
    m5.metric(
        "Monto promedio de última compra",
        f"{ultima_compra_promedio:,.2f}" if pd.notna(ultima_compra_promedio) else "Sin dato",
    )
    m6.metric(
        "Tasa de compra en la última semana",
        f"{tasa_compra:.1f}%",
        help="Clientes con compro_ultima_semana = 1 dividido entre todos los registros.",
    )

    st.subheader("Cómo se calculan")
    definiciones = pd.DataFrame(
        [
            {"Indicador": "Tasa de compra semanal", "Numerador": "Registros con compro_ultima_semana = 1", "Denominador": "Todos los clientes del dataset"},
            {"Indicador": "Compras promedio por cliente", "Numerador": "Suma de compras_12m", "Denominador": "Clientes con valor numérico en compras_12m"},
            {"Indicador": "Gasto promedio por cliente", "Numerador": "Suma de gasto_total_12m", "Denominador": "Clientes con valor numérico en gasto_total_12m"},
            {"Indicador": "Monto promedio de última compra", "Numerador": "Suma de monto_ultima_compra", "Denominador": "Clientes con monto de última compra disponible"},
        ]
    )
    st.dataframe(definiciones, hide_index=True, use_container_width=True)

    columnas_disponibles = [
        columna
        for columna in ["tamano_preferido", "genero"]
        if columna in df.columns
    ]
    if columnas_disponibles:
        st.subheader("Comparar grupos de clientes")
        grupo = st.selectbox(
            "Agrupar por",
            columnas_disponibles,
            format_func=lambda col: {
                "tamano_preferido": "Tamaño preferido",
                "genero": "Género",
            }.get(col, col),
            key="descriptiva_grupo",
        )
        resumen = _resumen_por_grupo(df, grupo)
        st.dataframe(
            resumen.style.format(
                {
                    "Tasa compra semanal (%)": "{:.1f}%",
                    "Gasto promedio por cliente (12 meses)": "{:,.2f}",
                    "Compras promedio por cliente (12 meses)": "{:.1f}",
                    "Visitas promedio por cliente (30 días)": "{:.1f}",
                },
                na_rep="—",
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(resumen.set_index("Grupo")["Tasa compra semanal (%)"])

    st.subheader("Distribuciones")
    graf1, graf2 = st.columns(2)
    with graf1:
        if "tamano_preferido" in df.columns:
            st.write("**Tamaño de porción preferido**")
            conteo_tamanos = df["tamano_preferido"].value_counts(dropna=False)
            st.bar_chart(conteo_tamanos)
        else:
            st.info("El dataset no incluye `tamano_preferido`.")
    with graf2:
        if "visitas_restaurante_30d" in df.columns:
            st.write("**Visitas al restaurante (últimos 30 días)**")
            visitas = _serie_numerica(df, "visitas_restaurante_30d").dropna()
            if not visitas.empty:
                histograma = pd.cut(visitas, bins=10).value_counts(sort=False)
                histograma.index = histograma.index.astype(str)
                st.bar_chart(histograma)
            else:
                st.info("No hay valores numéricos disponibles para las visitas.")
        else:
            st.info("El dataset no incluye `visitas_restaurante_30d`.")

    st.caption(
        "La tasa semanal corresponde al periodo indicado por la variable; los indicadores "
        "de compras y gasto usan una ventana de 12 meses. Las comparaciones entre grupos "
        "son descriptivas y no explican por sí solas las diferencias observadas."
    )
