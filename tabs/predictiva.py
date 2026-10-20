"""Modelos de clasificación exploratorios para la pestaña predictiva."""

import sys

import pandas as pd
import streamlit as st


VARIABLES_HISTORICAS = {
    "visitas_restaurante_30d",
    "compras_12m",
    "gasto_total_12m",
    "dias_desde_ultima_compra",
    "monto_ultima_compra",
}


def _construir_pipeline(columnas_numericas, columnas_categoricas, balancear):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformadores = []
    if columnas_numericas:
        pipeline_numerica = Pipeline(
            [
                ("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler()),
            ]
        )
        transformadores.append(("numericas", pipeline_numerica, columnas_numericas))
    if columnas_categoricas:
        pipeline_categorica = Pipeline(
            [
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("codificar", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformadores.append(("categoricas", pipeline_categorica, columnas_categoricas))

    preparacion = ColumnTransformer(transformadores, remainder="drop")
    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if balancear else None,
        random_state=42,
    )
    return Pipeline([("preparacion", preparacion), ("clasificador", modelo)])


def mostrar(df):
    st.header("4. Estadística predictiva")
    st.markdown(
        "Entrena un clasificador para estimar una etiqueta del dataset. "
        "El resultado es un ejercicio de aprendizaje con datos sintéticos, "
        "no una previsión validada para tomar decisiones reales."
    )

    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
    except ImportError:
        st.error(
            "La aplicación no encuentra scikit-learn en el intérprete que está ejecutando Streamlit."
        )
        st.caption(f"Intérprete activo: `{sys.executable}`")
        st.code(f'"{sys.executable}" -m pip install -r requirements.txt')
        return

    objetivos = [
        columna
        for columna in ["compro_ultima_semana", "tamano_preferido"]
        if columna in df.columns
    ]
    if not objetivos:
        st.warning("No se encontraron las variables objetivo esperadas en el dataset.")
        return

    objetivo = st.selectbox(
        "¿Qué queremos estimar?",
        objetivos,
        format_func=lambda col: {
            "compro_ultima_semana": "Compra durante la última semana (clasificación binaria)",
            "tamano_preferido": "Tamaño de porción preferido (clasificación multiclase)",
        }.get(col, col),
        key="predictiva_objetivo",
    )

    columnas_excluidas = {"cliente_id", "fecha_alta", objetivo}
    candidatas = [col for col in df.columns if col not in columnas_excluidas]
    seguras_por_defecto = [
        col
        for col in candidatas
        if col not in VARIABLES_HISTORICAS and col != "compro_ultima_semana"
    ]
    seleccionadas = st.multiselect(
        "Variables predictoras",
        options=candidatas,
        default=seguras_por_defecto,
        help=(
            "Se excluyen por defecto identificador, fecha y variables de historial. "
            "Incluye una variable solo si estaría disponible antes del momento en que "
            "se desea hacer la predicción."
        ),
        key="predictiva_variables",
    )
    historial_seleccionado = [col for col in seleccionadas if col in VARIABLES_HISTORICAS]
    if historial_seleccionado:
        st.warning(
            "Revisa el momento de medición de estas variables para evitar fuga de información: "
            + ", ".join(f"`{col}`" for col in historial_seleccionado)
            + "."
        )
    if objetivo == "tamano_preferido" and "compro_ultima_semana" in seleccionadas:
        st.warning(
            "`compro_ultima_semana` puede reflejar información posterior al momento "
            "de elegir el tamaño; confirma que sea válida como predictor."
        )
    if objetivo == "compro_ultima_semana" and "tamano_preferido" in seleccionadas:
        st.warning(
            "Confirma que `tamano_preferido` se conocía antes de la última semana "
            "para descartar fuga de información."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        proporcion_prueba = st.slider(
            "Proporción para prueba",
            min_value=0.2,
            max_value=0.4,
            value=0.25,
            step=0.05,
            key="predictiva_proporcion_prueba",
        )
    with col2:
        semilla = st.number_input(
            "Semilla aleatoria",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            key="predictiva_semilla",
        )
    with col3:
        if objetivo == "tamano_preferido":
            balancear = st.checkbox(
                "Balancear Chico, Mediano y Grande",
                value=True,
                help=(
                    "Pondera inversamente la frecuencia de cada tamaño durante el "
                    "entrenamiento. No garantiza que todas las clases se predigan bien."
                ),
                key="predictiva_balancear_tamanos",
            )
        else:
            balancear = st.checkbox(
                "Balancear peso de las clases",
                value=False,
                help="Da más peso a las clases menos frecuentes durante el ajuste.",
                key="predictiva_balancear_compra",
            )

    st.info(
        "El modelo de compra semanal usa regresión logística. El dataset no establece "
        "cuándo se registró cada predictor; los resultados pueden sobreestimar el "
        "desempeño si las variables contienen información posterior al resultado."
    )

    if not seleccionadas:
        st.warning("Selecciona al menos una variable predictora.")
        return

    datos = df[seleccionadas + [objetivo]].copy()
    datos[objetivo] = datos[objetivo].where(datos[objetivo].notna())
    datos = datos.dropna(subset=[objetivo])
    clases = datos[objetivo].value_counts()
    if len(clases) < 2:
        st.warning("Se requieren al menos dos clases con registros para entrenar.")
        return
    if clases.min() < 2:
        st.warning("Cada clase necesita al menos dos registros para separar entrenamiento y prueba.")
        return

    X = datos[seleccionadas].copy()
    y = datos[objetivo].copy()
    columnas_numericas = [
        col for col in seleccionadas if pd.api.types.is_numeric_dtype(X[col])
    ]
    columnas_categoricas = [col for col in seleccionadas if col not in columnas_numericas]
    if not columnas_numericas and not columnas_categoricas:
        st.warning("No hay variables predictoras utilizables.")
        return

    try:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X,
            y,
            test_size=proporcion_prueba,
            random_state=int(semilla),
            stratify=y,
        )
        pipeline = _construir_pipeline(
            columnas_numericas, columnas_categoricas, balancear
        )
        pipeline.fit(X_entrenamiento, y_entrenamiento)
        predicciones = pipeline.predict(X_prueba)
    except Exception as error:
        st.error(f"No se pudo entrenar el modelo: {error}")
        return

    accuracy = accuracy_score(y_prueba, predicciones)
    balanced_accuracy = balanced_accuracy_score(y_prueba, predicciones)
    exactitud_base = y_entrenamiento.value_counts(normalize=True).max()
    m1, m2, m3 = st.columns(3)
    m1.metric("Registros de entrenamiento", f"{len(X_entrenamiento):,}")
    m2.metric("Registros de prueba", f"{len(X_prueba):,}")
    m3.metric("Exactitud balanceada", f"{balanced_accuracy:.1%}")
    m4, m5 = st.columns(2)
    m4.metric("Exactitud del modelo", f"{accuracy:.1%}")
    m5.metric(
        "Referencia: siempre la clase más común",
        f"{exactitud_base:.1%}",
        help="Exactitud que se obtendría prediciendo para todos los casos la clase más frecuente del conjunto de entrenamiento.",
    )

    st.subheader("Evaluación en datos de prueba")
    st.write(
        "Compara la exactitud del modelo con la referencia mayoritaria. Si el modelo "
        "no la supera, no mejora la proporción global de aciertos frente a predecir "
        "siempre la clase más común. La exactitud balanceada da el mismo peso a cada clase."
    )
    etiquetas = pipeline.classes_
    matriz = confusion_matrix(y_prueba, predicciones, labels=etiquetas)
    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real: {etiqueta}" for etiqueta in etiquetas],
        columns=[f"Predicho: {etiqueta}" for etiqueta in etiquetas],
    )
    st.write("**Matriz de confusión**")
    st.dataframe(matriz_df, use_container_width=True)

    reporte = classification_report(
        y_prueba,
        predicciones,
        labels=etiquetas,
        output_dict=True,
        zero_division=0,
    )
    resumen_clases = pd.DataFrame(reporte).T
    resumen_clases.index.name = "Clase / promedio"
    resumen_clases = resumen_clases.reset_index()
    st.write("**Precisión, recall y F1 por clase**")
    st.dataframe(
        resumen_clases.style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"},
            na_rep="—",
        ),
        hide_index=True,
        use_container_width=True,
    )

    if len(etiquetas) == 2:
        try:
            probabilidades = pipeline.predict_proba(X_prueba)[:, 1]
            auc = roc_auc_score(y_prueba, probabilidades)
            st.metric(
                f"ROC-AUC (clase positiva: {etiquetas[1]})",
                f"{auc:.3f}",
                help="Mide la capacidad de ordenar positivos por encima de negativos; 0.5 equivale al azar.",
            )
        except ValueError:
            st.caption("No se pudo calcular ROC-AUC para esta partición de prueba.")

    elif objetivo == "tamano_preferido":
        predicciones_unicas = pd.Series(predicciones).nunique()
        if predicciones_unicas < len(etiquetas):
            st.warning(
                f"El modelo asignó {predicciones_unicas} de {len(etiquetas)} tipos "
                "de porción en la prueba. Revisa el recall de cada clase antes de "
                "interpretar las recomendaciones."
            )

        probabilidades_tamanos = pipeline.predict_proba(X_prueba)
        tabla_probabilidades = pd.DataFrame(
            probabilidades_tamanos,
            index=X_prueba.index,
            columns=[f"Probabilidad {etiqueta}" for etiqueta in etiquetas],
        )
        tabla_recomendaciones = pd.DataFrame(index=X_prueba.index)
        if "cliente_id" in df.columns:
            tabla_recomendaciones["Cliente"] = df.loc[X_prueba.index, "cliente_id"]
        tabla_recomendaciones["Tamaño real"] = y_prueba
        tabla_recomendaciones["Tamaño estimado"] = predicciones
        tabla_recomendaciones = tabla_recomendaciones.join(tabla_probabilidades)
        st.subheader("Probabilidades para Chico, Mediano y Grande")
        st.caption(
            "Cada fila muestra la probabilidad estimada para cada clase; el tamaño "
            "estimado es el de mayor probabilidad."
        )
        st.dataframe(
            tabla_recomendaciones.head(20),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Descargar probabilidades por tamaño",
            data=tabla_recomendaciones.to_csv(index=False).encode("utf-8-sig"),
            file_name="probabilidades_tamano_porcion.csv",
            mime="text/csv",
            key="predictiva_descargar_probabilidades_tamano",
        )

    st.caption(
        "La imputación, el escalado y la codificación se ajustan solo con el conjunto "
        "de entrenamiento. La separación está estratificada y usa la semilla elegida. "
        "Una partición de prueba de datos sintéticos no valida el modelo para clientes reales."
    )
