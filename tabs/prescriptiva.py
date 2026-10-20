"""Recomendaciones explicables por segmento o propensión estimada."""

import pandas as pd
import streamlit as st


ACCIONES_POR_SEGMENTO = {
    "Reactivación": (
        "Probar un mensaje de reactivación y, si está disponible, mencionar el tamaño "
        "que suele preferir. Medir recompra después del contacto."
    ),
    "Compra frecuente": (
        "Facilitar una nueva compra usando el tamaño preferido como referencia. "
        "Comparar la respuesta con un grupo sin contacto."
    ),
    "Seguimiento": (
        "Probar una comunicación sencilla para conocer qué facilitaría su próxima visita; "
        "registrar respuesta y recompra."
    ),
}

ACCIONES_POR_PROBABILIDAD = {
    "Alta": "Facilitar la recompra con una comunicación breve basada en su tamaño preferido, si se conoce.",
    "Media": "Probar un recordatorio y medir si cambia la recompra frente a un grupo de comparación.",
    "Baja": "Explorar barreras de regreso mediante una prueba de contacto pequeña y medir la respuesta.",
}


def _serie_numerica(df, columna):
    if columna not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _mostrar_por_segmento(df):
    columnas_requeridas = {"dias_desde_ultima_compra", "compras_12m"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        st.warning("Faltan variables para crear segmentos: " + ", ".join(sorted(faltantes)))
        return

    recencia = _serie_numerica(df, "dias_desde_ultima_compra")
    frecuencia = _serie_numerica(df, "compras_12m")
    dias_max = int(recencia.max()) if recencia.notna().any() else 180
    dias_max = max(dias_max, 1)
    frecuencia_sugerida = int(round(frecuencia.quantile(0.75))) if frecuencia.notna().any() else 1
    frecuencia_max = int(frecuencia.max()) if frecuencia.notna().any() else max(frecuencia_sugerida, 1)

    col1, col2 = st.columns(2)
    with col1:
        umbral_recencia = st.slider(
            "Considerar para reactivación desde (días sin compra)",
            min_value=1,
            max_value=dias_max,
            value=min(60, dias_max),
            key="prescriptiva_umbral_recencia",
        )
    with col2:
        umbral_frecuencia = st.number_input(
            "Compras en 12 meses para considerar compra frecuente",
            min_value=0,
            max_value=max(frecuencia_max, 1),
            value=min(max(frecuencia_sugerida, 0), max(frecuencia_max, 1)),
            step=1,
            help="El valor inicial corresponde aproximadamente al percentil 75 de compras_12m.",
            key="prescriptiva_umbral_frecuencia",
        )

    recomendaciones = df.copy()
    recomendaciones["Días desde última compra"] = recencia
    recomendaciones["Compras en 12 meses"] = frecuencia
    condicion_reactivacion = recencia.ge(umbral_recencia)
    condicion_frecuente = frecuencia.ge(umbral_frecuencia)
    recomendaciones["Segmento"] = "Seguimiento"
    recomendaciones.loc[condicion_frecuente, "Segmento"] = "Compra frecuente"
    recomendaciones.loc[condicion_reactivacion, "Segmento"] = "Reactivación"
    recomendaciones["Acción recomendada"] = recomendaciones["Segmento"].map(
        ACCIONES_POR_SEGMENTO
    )
    if "tamano_preferido" in recomendaciones.columns:
        recomendaciones["Evidencia"] = (
            "Compras 12m: "
            + recomendaciones["Compras en 12 meses"].fillna(0).astype(int).astype(str)
            + "; días sin compra: "
            + recomendaciones["Días desde última compra"].fillna(-1).astype(int).astype(str)
            + "; tamaño preferido: "
            + recomendaciones["tamano_preferido"].fillna("sin dato").astype(str)
        )
    else:
        recomendaciones["Evidencia"] = (
            "Compras 12m: "
            + recomendaciones["Compras en 12 meses"].fillna(0).astype(int).astype(str)
            + "; días sin compra: "
            + recomendaciones["Días desde última compra"].fillna(-1).astype(int).astype(str)
        )

    st.markdown(
        "La asignación usa reglas editables. Si un cliente cumple ambas reglas, "
        "se asigna a **Reactivación** por prioridad de recencia."
    )
    resumen = (
        recomendaciones.groupby("Segmento", dropna=False)
        .agg(
            Clientes=("Segmento", "size"),
            Compras_promedio_12m=("Compras en 12 meses", "mean"),
            Dias_promedio_sin_compra=("Días desde última compra", "mean"),
        )
        .reindex(["Reactivación", "Compra frecuente", "Seguimiento"])
        .dropna(how="all")
        .reset_index()
    )
    if "compro_ultima_semana" in df.columns:
        resultado = _serie_numerica(recomendaciones, "compro_ultima_semana")
        tasa = recomendaciones.assign(_resultado=resultado).groupby("Segmento")["_resultado"].mean() * 100
        resumen["Compra última semana (%)"] = resumen["Segmento"].map(tasa)
        resumen = resumen.rename(
            columns={"Compra última semana (%)": "Tasa observada última semana (%)"}
        )

    st.subheader("Resumen de segmentos")
    st.dataframe(
        resumen.style.format(
            {
                "Compras_promedio_12m": "{:.1f}",
                "Dias_promedio_sin_compra": "{:.1f}",
                "Tasa observada última semana (%)": "{:.1f}%",
            },
            na_rep="—",
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "La tasa semanal es una observación del dataset dentro de cada segmento; "
        "no mide el efecto de la acción recomendada."
    )

    st.subheader("Acciones y evidencia")
    acciones = pd.DataFrame(
        [
            {"Segmento": nombre, "Regla": regla, "Acción recomendada": accion}
            for nombre, regla, accion in [
                (
                    "Reactivación",
                    f"Días desde última compra ≥ {umbral_recencia}",
                    ACCIONES_POR_SEGMENTO["Reactivación"],
                ),
                (
                    "Compra frecuente",
                    f"Compras en 12 meses ≥ {int(umbral_frecuencia)} y no cumple reactivación",
                    ACCIONES_POR_SEGMENTO["Compra frecuente"],
                ),
                (
                    "Seguimiento",
                    "No cumple las reglas anteriores",
                    ACCIONES_POR_SEGMENTO["Seguimiento"],
                ),
            ]
        ]
    )
    st.dataframe(acciones, hide_index=True, use_container_width=True)

    columnas_salida = [
        columna
        for columna in ["cliente_id", "Segmento", "Evidencia", "Acción recomendada"]
        if columna in recomendaciones.columns
    ]
    with st.expander("Ver recomendaciones por cliente"):
        st.dataframe(
            recomendaciones[columnas_salida],
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Descargar recomendaciones por segmento",
            data=recomendaciones[columnas_salida].to_csv(index=False).encode("utf-8-sig"),
            file_name="recomendaciones_por_segmento.csv",
            mime="text/csv",
            key="prescriptiva_descargar_segmentos",
        )


def _mostrar_por_probabilidad(df):
    try:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from tabs.predictiva import _construir_pipeline
    except ImportError:
        st.error("La modalidad por probabilidad requiere scikit-learn.")
        st.code("python -m pip install -r requirements.txt")
        return

    objetivo = "compro_ultima_semana"
    if objetivo not in df.columns:
        st.warning("No se encontró `compro_ultima_semana` para estimar la propensión.")
        return

    candidatas = [
        columna
        for columna in df.columns
        if columna not in {"cliente_id", "fecha_alta", objetivo}
    ]
    historial = {
        "visitas_restaurante_30d",
        "compras_12m",
        "gasto_total_12m",
        "dias_desde_ultima_compra",
        "monto_ultima_compra",
    }
    predictores_iniciales = [
        columna
        for columna in candidatas
        if columna not in historial and columna != "tamano_preferido"
    ]
    predictores = st.multiselect(
        "Variables para estimar la compra semanal",
        options=candidatas,
        default=predictores_iniciales,
        help=(
            "Se excluyen inicialmente ID, fecha e historial de compras. "
            "Incluye variables solo si estaban disponibles antes de la última semana."
        ),
        key="prescriptiva_predictores_probabilidad",
    )
    historial_usado = [col for col in predictores if col in historial]
    if historial_usado or "tamano_preferido" in predictores:
        potenciales = historial_usado + (
            ["tamano_preferido"] if "tamano_preferido" in predictores else []
        )
        st.warning(
            "Confirma que estas variables eran conocidas antes del periodo de predicción: "
            + ", ".join(f"`{col}`" for col in potenciales)
            + "."
        )
    if not predictores:
        st.warning("Selecciona al menos una variable predictora.")
        return

    datos = df[predictores + [objetivo]].dropna(subset=[objetivo]).copy()
    y = pd.to_numeric(datos[objetivo], errors="coerce")
    mascara = y.isin([0, 1])
    datos = datos.loc[mascara]
    y = y.loc[mascara].astype(int)
    clases = y.value_counts()
    if len(clases) < 2 or clases.min() < 2:
        st.warning("Se necesitan al menos dos casos de cada clase para calcular probabilidades de prueba.")
        return

    test_size = st.slider(
        "Proporción para prueba",
        min_value=0.2,
        max_value=0.4,
        value=0.25,
        step=0.05,
        key="prescriptiva_proporcion_prueba",
    )
    umbral_bajo, umbral_alto = st.slider(
        "Rangos de probabilidad para definir niveles",
        min_value=0.1,
        max_value=0.9,
        value=(0.4, 0.7),
        step=0.05,
        key="prescriptiva_umbrales_probabilidad",
    )

    X = datos[predictores].copy()
    columnas_numericas = [
        columna for columna in predictores if pd.api.types.is_numeric_dtype(X[columna])
    ]
    columnas_categoricas = [columna for columna in predictores if columna not in columnas_numericas]
    try:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
        modelo = _construir_pipeline(columnas_numericas, columnas_categoricas, False)
        modelo.fit(X_entrenamiento, y_entrenamiento)
        indice_positivo = list(modelo.classes_).index(1)
        probabilidad = modelo.predict_proba(X_prueba)[:, indice_positivo]
    except Exception as error:
        st.error(f"No se pudieron calcular las probabilidades: {error}")
        return

    # Las recomendaciones se generan solo para los registros reservados para prueba.
    recomendaciones = datos.loc[X_prueba.index].copy()
    recomendaciones["Probabilidad de compra semanal"] = probabilidad
    recomendaciones["Nivel"] = "Baja"
    recomendaciones.loc[
        recomendaciones["Probabilidad de compra semanal"] >= umbral_bajo, "Nivel"
    ] = "Media"
    recomendaciones.loc[
        recomendaciones["Probabilidad de compra semanal"] >= umbral_alto, "Nivel"
    ] = "Alta"
    recomendaciones["Acción recomendada"] = recomendaciones["Nivel"].map(
        ACCIONES_POR_PROBABILIDAD
    )

    try:
        predicciones = modelo.predict(X_prueba)
        exactitud_balanceada = balanced_accuracy_score(y_prueba, predicciones)
        auc = roc_auc_score(y_prueba, probabilidad)
        m1, m2, m3 = st.columns(3)
        m1.metric("Clientes en prueba", f"{len(recomendaciones):,}")
        m2.metric("Exactitud balanceada en prueba", f"{exactitud_balanceada:.1%}")
        m3.metric("ROC-AUC en prueba", f"{auc:.3f}")
    except ValueError:
        st.caption("No se pudieron calcular todas las métricas para esta partición.")

    resumen = (
        recomendaciones.groupby("Nivel", observed=False)
        .agg(
            Clientes=("Nivel", "size"),
            Probabilidad_promedio=("Probabilidad de compra semanal", "mean"),
        )
        .reindex(["Baja", "Media", "Alta"])
        .reset_index()
    )
    st.subheader("Acciones por nivel de probabilidad")
    st.dataframe(
        resumen.style.format({"Probabilidad_promedio": "{:.1%}"}, na_rep="—"),
        hide_index=True,
        use_container_width=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Nivel": nivel, "Regla": regla, "Acción recomendada": ACCIONES_POR_PROBABILIDAD[nivel]}
                for nivel, regla in [
                    ("Baja", f"Probabilidad < {umbral_bajo:.0%}"),
                    ("Media", f"{umbral_bajo:.0%} ≤ probabilidad < {umbral_alto:.0%}"),
                    ("Alta", f"Probabilidad ≥ {umbral_alto:.0%}"),
                ]
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "Una probabilidad alta indica propensión estimada, no que una acción vaya a "
        "causar la compra. Para medir el efecto de una acción se necesita una prueba "
        "con grupo de comparación. Las recomendaciones se calculan solo para los "
        "registros de prueba, que no se usaron para ajustar el modelo."
    )

    columnas_salida = [
        columna
        for columna in ["cliente_id", "tamano_preferido"]
        if columna in df.columns
    ]
    # `recomendaciones` conserva solo predictores y objetivo; recuperar los datos
    # identificativos desde el DataFrame original usando el índice de prueba.
    recomendaciones_salida = df.loc[
        recomendaciones.index, columnas_salida
    ].copy()
    recomendaciones_salida["Probabilidad de compra semanal"] = recomendaciones[
        "Probabilidad de compra semanal"
    ]
    recomendaciones_salida["Nivel"] = recomendaciones["Nivel"]
    recomendaciones_salida["Acción recomendada"] = recomendaciones["Acción recomendada"]
    st.subheader("Recomendaciones por cliente en el conjunto de prueba")
    st.dataframe(
        recomendaciones_salida.sort_values(
            "Probabilidad de compra semanal", ascending=False
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Descargar recomendaciones por probabilidad",
        data=recomendaciones_salida.to_csv(index=False).encode("utf-8-sig"),
        file_name="recomendaciones_por_probabilidad.csv",
        mime="text/csv",
        key="prescriptiva_descargar_probabilidad",
    )

    st.divider()
    st.subheader("Estimación del tipo de porción a elegir")
    objetivo_porcion = "tamano_preferido"
    if objetivo_porcion not in df.columns:
        st.warning("No se encontró `tamano_preferido` para estimar la porción.")
        return

    candidatas_porcion = [
        columna
        for columna in df.columns
        if columna
        not in {
            "cliente_id",
            "fecha_alta",
            objetivo_porcion,
            "compro_ultima_semana",
        }
    ]
    predictores_porcion_iniciales = [
        columna
        for columna in candidatas_porcion
        if columna not in historial
    ]
    predictores_porcion = st.multiselect(
        "Variables para estimar el tipo de porción",
        options=candidatas_porcion,
        default=predictores_porcion_iniciales,
        help=(
            "Se excluyen por defecto el identificador, la fecha, la compra semanal y "
            "el historial. Selecciona solo datos conocidos antes de elegir la porción."
        ),
        key="prescriptiva_predictores_porcion",
    )
    historial_porcion = [
        columna for columna in predictores_porcion if columna in historial
    ]
    if historial_porcion:
        st.warning(
            "Confirma que estas variables estaban disponibles antes de elegir la porción: "
            + ", ".join(f"`{columna}`" for columna in historial_porcion)
            + "."
        )

    if not predictores_porcion:
        st.warning("Selecciona al menos una variable predictora para la porción.")
        return

    datos_porcion = df[predictores_porcion + [objetivo_porcion]].copy()
    etiquetas_normalizadas = (
        datos_porcion[objetivo_porcion]
        .astype("string")
        .str.strip()
        .str.casefold()
        .map({"chico": "Chico", "mediano": "Mediano", "grande": "Grande"})
    )
    filas_sin_tamano_valido = int(etiquetas_normalizadas.isna().sum())
    datos_porcion[objetivo_porcion] = etiquetas_normalizadas
    datos_porcion = datos_porcion.dropna(subset=[objetivo_porcion]).copy()
    y_porcion = datos_porcion[objetivo_porcion]
    conteos_porcion = y_porcion.value_counts()
    tamanos_esperados = {"Chico", "Mediano", "Grande"}
    tamanos_ausentes = tamanos_esperados - set(conteos_porcion.index)
    if filas_sin_tamano_valido:
        st.warning(
            f"Se excluirán {filas_sin_tamano_valido} registros sin un tamaño reconocido "
            "(Chico, Mediano o Grande)."
        )
    if tamanos_ausentes:
        st.warning(
            "No se puede entrenar la estimación de tres porciones porque faltan clases "
            "en el dataset: " + ", ".join(sorted(tamanos_ausentes))
        )
        return
    if conteos_porcion.min() < 2:
        st.warning("Cada tamaño necesita al menos dos registros para separarlo en entrenamiento y prueba.")
        return

    proporcion_prueba_porcion = st.slider(
        "Proporción de datos reservada para evaluar porciones",
        min_value=0.2,
        max_value=0.4,
        value=0.25,
        step=0.05,
        key="prescriptiva_proporcion_prueba_porcion",
    )
    st.info(
        "El modelo asigna el mismo peso a Chico, Mediano y Grande durante el "
        "entrenamiento para considerar las tres clases. Esto no garantiza que las "
        "tres se predigan correctamente; revisa las métricas por tamaño."
    )
    X_porcion = datos_porcion[predictores_porcion].copy()
    numericas_porcion = [
        columna
        for columna in predictores_porcion
        if pd.api.types.is_numeric_dtype(X_porcion[columna])
    ]
    categoricas_porcion = [
        columna
        for columna in predictores_porcion
        if columna not in numericas_porcion
    ]

    try:
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_porcion,
            y_porcion,
            test_size=proporcion_prueba_porcion,
            random_state=42,
            stratify=y_porcion,
        )
        modelo_porcion = _construir_pipeline(
            numericas_porcion, categoricas_porcion, True
        )
        modelo_porcion.fit(X_train_p, y_train_p)
        probabilidades_porcion = modelo_porcion.predict_proba(X_test_p)
        clases_porcion = modelo_porcion.classes_
        posiciones_ganadoras = probabilidades_porcion.argmax(axis=1)
        porciones_estimadas = clases_porcion[posiciones_ganadoras]
        confianza_porcion = probabilidades_porcion.max(axis=1)
        aciertos_porcion = porciones_estimadas == y_test_p.to_numpy()
    except Exception as error:
        st.error(f"No se pudo estimar el tipo de porción: {error}")
        return

    resultados_porcion = pd.DataFrame(
        {
            "Porción estimada": porciones_estimadas,
            "Confianza del modelo": confianza_porcion,
            "Acierto observado en prueba": aciertos_porcion,
        },
        index=X_test_p.index,
    )
    resultados_porcion["Acción recomendada"] = resultados_porcion[
        "Porción estimada"
    ].map(
        lambda porcion: (
            f"Hacer visible la opción {porcion} como preferencia estimada y "
            "confirmar disponibilidad antes de sugerirla."
        )
    )
    metrica_col1, metrica_col2, metrica_col3 = st.columns(3)
    metrica_col1.metric("Clientes evaluados", f"{len(X_test_p):,}")
    metrica_col2.metric(
        "Exactitud",
        f"{aciertos_porcion.mean():.1%}",
        help="Proporción de porciones correctamente clasificadas en la prueba.",
    )
    metrica_col3.metric(
        "Exactitud balanceada",
        f"{balanced_accuracy_score(y_test_p, porciones_estimadas):.1%}",
        help="Promedio del recall de cada clase en los registros reservados para prueba.",
    )
    clases_estimadas = pd.Series(porciones_estimadas).nunique()
    if clases_estimadas < len(clases_porcion):
        st.warning(
            f"El modelo asignó solo {clases_estimadas} de {len(clases_porcion)} tipos "
            "de porción en la prueba. Revisa la exactitud balanceada y las probabilidades "
            "por clase antes de usar estas recomendaciones."
        )

    resumen_porcion = (
        resultados_porcion.groupby("Porción estimada")
        .agg(
            Clientes=("Porción estimada", "size"),
            Confianza_promedio=("Confianza del modelo", "mean"),
            Acierto_observado=("Acierto observado en prueba", "mean"),
        )
        .reindex(clases_porcion)
        .reset_index()
        .rename(
            columns={
                "Confianza_promedio": "Confianza promedio",
                "Acierto_observado": "Porción correcta en prueba (%)",
            }
        )
    )
    resumen_porcion["Clientes"] = resumen_porcion["Clientes"].fillna(0).astype(int)
    resumen_porcion["Porción correcta en prueba (%)"] *= 100
    st.dataframe(
        resumen_porcion.style.format(
            {
                "Confianza promedio": "{:.1%}",
                "Porción correcta en prueba (%)": "{:.1f}%",
            },
            na_rep="—",
        ),
        hide_index=True,
        use_container_width=True,
    )

    probabilidades_porcion_df = pd.DataFrame(
        probabilidades_porcion,
        index=X_test_p.index,
        columns=[f"Probabilidad {clase}" for clase in clases_porcion],
    )
    columnas_salida_porcion = [
        columna for columna in ["cliente_id"] if columna in df.columns
    ]
    salida_porcion = df.loc[X_test_p.index, columnas_salida_porcion].copy()
    salida_porcion["Porción estimada"] = porciones_estimadas
    salida_porcion["Confianza del modelo"] = confianza_porcion
    salida_porcion = salida_porcion.join(probabilidades_porcion_df)
    salida_porcion["Acción recomendada"] = resultados_porcion[
        "Acción recomendada"
    ]
    st.dataframe(
        salida_porcion.sort_values("Confianza del modelo", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Descargar recomendaciones de porción",
        data=salida_porcion.to_csv(index=False).encode("utf-8-sig"),
        file_name="recomendaciones_porcion.csv",
        mime="text/csv",
        key="prescriptiva_descargar_porcion",
    )
    st.caption(
        "La confianza es la probabilidad más alta que asigna el modelo entre las clases; "
        "la tabla conserva las probabilidades de Chico, Mediano y Grande para comparar "
        "alternativas. No garantiza que esa porción vaya a elegirse. La exactitud "
        "observada se calcula en registros de prueba que no se usaron para entrenar."
    )


def mostrar(df):
    st.header("5. Estadística prescriptiva")
    st.markdown(
        "Convierte los patrones observados o las probabilidades estimadas en acciones "
        "concretas y medibles. Estas recomendaciones son hipótesis para probar, no "
        "garantías de compra."
    )
    modo = st.radio(
        "¿Cómo quieres formar los grupos?",
        ["Reglas por segmento", "Niveles de probabilidad"],
        horizontal=True,
        key="prescriptiva_modo",
    )
    if modo == "Reglas por segmento":
        _mostrar_por_segmento(df)
    else:
        _mostrar_por_probabilidad(df)
