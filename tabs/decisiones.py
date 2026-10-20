"""Vista de decisión con acción, mensaje y justificación por segmento."""

import pandas as pd
import streamlit as st


CONTENIDO_SEGMENTOS = {
    "Reactivación": {
        "accion": (
            "Probar una comunicación de reactivación con un grupo de control. "
            "Si se personaliza, usar solo el tamaño preferido que ya esté registrado."
        ),
        "mensaje": (
            "Hace tiempo que no registramos una compra tuya. Si decides volver, "
            "cuéntanos qué tamaño de pozole prefieres: {{tamano_preferido}}."
        ),
        "justificacion": (
            "El segmento agrupa clientes cuya última compra fue hace al menos "
            "el umbral de días seleccionado. La recencia indica tiempo desde una "
            "compra observada; no demuestra por qué dejó de comprar."
        ),
        "medicion": "Comparar la tasa de recompra del grupo contactado con la del grupo de control.",
    },
    "Compra frecuente": {
        "accion": (
            "Facilitar una nueva compra y comprobar si la preferencia de tamaño "
            "registrada sigue vigente. Medir el resultado frente a un grupo sin contacto."
        ),
        "mensaje": (
            "Gracias por elegirnos con frecuencia. Tenemos registrado que prefieres "
            "el tamaño {{tamano_preferido}}. ¿Sigue siendo tu elección para tu próxima compra?"
        ),
        "justificacion": (
            "El segmento reúne clientes con compras en 12 meses iguales o superiores "
            "al umbral definido, siempre que no hayan quedado antes en Reactivación."
        ),
        "medicion": "Medir recompra y confirmar si se mantiene el tamaño preferido.",
    },
    "Seguimiento": {
        "accion": (
            "Probar una pregunta breve para conocer la preferencia o la necesidad "
            "del cliente antes de diseñar una acción posterior."
        ),
        "mensaje": (
            "Para tu próxima visita, ¿qué tamaño de pozole prefieres? "
            "Tu respuesta nos ayudará a tener en cuenta tu elección."
        ),
        "justificacion": (
            "Este grupo no cumple las reglas priorizadas de Reactivación ni de "
            "Compra frecuente. Se propone aprender sobre su preferencia sin asumir "
            "una causa o una intención de compra."
        ),
        "medicion": "Registrar respuestas y observar la recompra en el periodo definido.",
    },
}


def _serie_numerica(df, columna):
    if columna not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[columna], errors="coerce")


def _segmentar(df):
    recencia = _serie_numerica(df, "dias_desde_ultima_compra")
    frecuencia = _serie_numerica(df, "compras_12m")
    dias_max = int(recencia.max()) if recencia.notna().any() else 180
    dias_max = max(dias_max, 1)
    q75 = int(round(frecuencia.quantile(0.75))) if frecuencia.notna().any() else 1
    max_frecuencia = int(frecuencia.max()) if frecuencia.notna().any() else 1

    umbral_recencia = int(
        st.session_state.get("prescriptiva_umbral_recencia", min(60, dias_max))
    )
    umbral_frecuencia = int(
        st.session_state.get(
            "prescriptiva_umbral_frecuencia",
            min(max(q75, 0), max(max_frecuencia, 1)),
        )
    )
    # Mantener los límites dentro de los valores permitidos por los controles de pestaña 5.
    umbral_recencia = min(max(umbral_recencia, 1), dias_max)
    umbral_frecuencia = min(max(umbral_frecuencia, 0), max(max_frecuencia, 1))

    segmento = pd.Series("Seguimiento", index=df.index, dtype="object")
    segmento.loc[frecuencia.ge(umbral_frecuencia)] = "Compra frecuente"
    segmento.loc[recencia.ge(umbral_recencia)] = "Reactivación"
    return segmento, recencia, frecuencia, umbral_recencia, umbral_frecuencia


def mostrar(df):
    st.header("6. Toma de decisiones")
    st.markdown(
        "Selecciona un segmento para revisar la acción propuesta, un mensaje de ejemplo "
        "y los datos que justifican la decisión. El mensaje es un borrador; no se envía desde esta app."
    )

    requeridas = {"dias_desde_ultima_compra", "compras_12m"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        st.warning(
            "Para asignar segmentos se necesitan estas variables: "
            + ", ".join(sorted(faltantes))
        )
        return
    if df.empty:
        st.warning("No hay registros para resumir decisiones.")
        return

    segmentos, recencia, frecuencia, umbral_recencia, umbral_frecuencia = _segmentar(df)
    opciones = [
        segmento
        for segmento in ["Reactivación", "Compra frecuente", "Seguimiento"]
        if segmentos.eq(segmento).any()
    ]
    seleccionado = st.selectbox(
        "Selecciona un segmento",
        opciones,
        key="decisiones_segmento",
    )
    mascara = segmentos.eq(seleccionado)
    cantidad = int(mascara.sum())
    clientes = df.loc[mascara]

    tasa_compra = None
    if "compro_ultima_semana" in df.columns and cantidad:
        resultado = _serie_numerica(clientes, "compro_ultima_semana")
        if resultado.notna().any():
            tasa_compra = resultado.mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clientes en el segmento", f"{cantidad:,}")
    col2.metric(
        "Días promedio desde última compra",
        f"{recencia.loc[mascara].mean():.1f}"
        if recencia.loc[mascara].notna().any()
        else "Sin dato",
    )
    col3.metric(
        "Compras promedio (12 meses)",
        f"{frecuencia.loc[mascara].mean():.1f}"
        if frecuencia.loc[mascara].notna().any()
        else "Sin dato",
    )
    col4.metric(
        "Compra observada última semana",
        f"{tasa_compra:.1f}%" if tasa_compra is not None else "Sin dato",
        help="Porcentaje observado dentro del segmento; no mide el efecto de la acción.",
    )

    contenido = CONTENIDO_SEGMENTOS[seleccionado]
    with st.container(border=True):
        st.subheader(seleccionado)
        st.markdown("**Acción recomendada**")
        st.write(contenido["accion"])
        st.markdown("**Mensaje de ejemplo**")
        st.info(contenido["mensaje"])
        st.caption(
            "`{{tamano_preferido}}` es un campo de personalización: reemplázalo solo "
            "cuando exista un valor registrado para ese cliente."
        )
        st.markdown("**Justificación**")
        st.write(contenido["justificacion"])
        st.markdown("**Cómo medir la acción**")
        st.write(contenido["medicion"])

    st.subheader("Reglas de asignación")
    st.write(
        f"- Reactivación: `dias_desde_ultima_compra` ≥ {umbral_recencia}.\n"
        f"- Compra frecuente: `compras_12m` ≥ {umbral_frecuencia}, excepto quienes "
        "ya cumplen Reactivación.\n"
        "- Seguimiento: no cumple las reglas anteriores."
    )
    st.caption(
        "Los umbrales se comparten con la pestaña 5. La tasa de compra observada es "
        "descriptiva: para atribuir un cambio a la acción hace falta un grupo de comparación. "
        "El dataset no contiene canal de contacto ni permisos; valida ambos antes de usar un mensaje."
    )
