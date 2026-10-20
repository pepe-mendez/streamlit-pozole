# Guía del proyecto: análisis de ventas en una pozolería

## Propósito

Proyecto didáctico de análisis de datos con Python, pandas y Streamlit. La aplicación organiza el trabajo en seis etapas para explorar clientes de una pozolería y apoyar decisiones de marketing basadas en los datos disponibles.

Pregunta central de la aplicación:

> **¿Qué clientes tienen mayor probabilidad de volver a comprar y qué tamaño de porción elegirán?**

El dataset contiene una etiqueta de compra durante la última semana y una preferencia de tamaño. Estas variables permiten plantear ejercicios predictivos, pero sus resultados deben interpretarse como exploratorios: los datos son sintéticos y no demuestran que una acción de marketing cause una compra.

## Principios de trabajo

- Avanzar por etapas: **Entender → Limpiar → Descriptiva → Predictiva → Prescriptiva → Decisiones**.
- Revisar los datos antes de transformarlos y justificar cada tratamiento.
- Distinguir asociaciones y predicciones de relaciones causales.
- Usar las predicciones como apoyo, explicando errores, supuestos y límites.
- Mantener el código legible y usar nombres y textos en español en la interfaz.

## Estado actual

`app.py` configura Streamlit, carga el dataset, presenta la pregunta central y crea seis pestañas. Cada pestaña llama a su módulo: **Entender**, **Limpiar**, **Descriptiva**, **Predictiva**, **Prescriptiva** y **Decisiones**.

La estructura esperada para que la aplicación actual funcione es:

```text
ventas_pozole/
├── data/
│   └── dataset_clientes_sinteticos_ventas_pozole.csv
├── tabs/
│   ├── __init__.py
│   ├── entender.py
│   ├── limpiar.py
│   ├── descriptiva.py
│   ├── predictiva.py
│   ├── prescriptiva.py
│   └── decisiones.py
├── app.py
├── requirements.txt
├── project.md
└── project-pozole.md
```

La estructura descrita arriba sí está presente: `tabs/__init__.py` y `tabs/entender.py` existen. El módulo define `mostrar(df)` y ya implementa la exploración inicial: muestra el número de clientes y variables, una vista previa de diez filas, tipos, valores únicos y faltantes, duplicados exactos y un explorador con estadísticas numéricas o frecuencias categóricas. El selector usa la clave estable `variable_entender`, y la función recibe el DataFrame desde `app.py` sin recargar el CSV. La ruta del dataset que usa `app.py` coincide con el archivo disponible.

Mantener `app.py` como orquestador: configuración, carga, pestañas y llamadas a módulos. Pasar el DataFrame explícitamente a cada función; los módulos no deben volver a cargar el CSV ni depender de variables globales.

API sugerida para la primera pestaña:

```python
# tabs/entender.py
def mostrar(df):
    ...
```

La pestaña **Entender** presenta el propósito, filas y columnas, una muestra, tipos y valores ausentes, duplicados exactos y exploración de variables. Para variables numéricas, muestra estadísticos descriptivos; para categóricas, frecuencias, incluyendo ausencias si las hubiera. Usa la clave estable `variable_entender`.

La pestaña **Limpiar** revisa valores faltantes, duplicados exactos y valores negativos en medidas que deberían ser no negativas. Permite opcionalmente quitar duplicados exactos y convertir una columna cuyo nombre contenga `fecha` desde día/mes/año. Los cambios se aplican a una copia, se previsualizan y pueden descargarse como CSV; no se imputan faltantes ni se corrigen valores automáticamente. Los widgets tienen claves únicas.

La pestaña **Descriptiva** muestra clientes, compradores de la última semana y tasa de compra semanal, compras promedio por cliente en 12 meses, gasto promedio por cliente en 12 meses y monto promedio de última compra. La tasa semanal usa todos los registros como denominador; los demás promedios usan solo los valores numéricos disponibles. También permite comparar la tasa y promedios por tamaño preferido o género y muestra la distribución del tamaño preferido y de visitas en 30 días. Las unidades monetarias no se presuponen. Las comparaciones son descriptivas, no causales.

La pestaña **Predictiva** permite elegir entre `compro_ultima_semana` y `tamano_preferido`, seleccionar predictores y ajustar la proporción de prueba. Excluye `cliente_id` y `fecha_alta` de los predictores; las variables de historial quedan fuera por defecto y generan una advertencia si se incluyen. Usa regresión logística, imputación/escalado/codificación dentro de un `Pipeline`, partición estratificada reproducible, matriz de confusión, comparación con una referencia mayoritaria, exactitud, exactitud balanceada, precisión, recall, F1 y ROC-AUC para clasificación binaria. Para `tamano_preferido`, balancea por defecto Chico, Mediano y Grande, muestra las probabilidades por clase y avisa si alguna clase no se predice en los datos de prueba. Requiere `scikit-learn`, incluido en `requirements.txt`.

La pestaña **Prescriptiva** ofrece reglas por segmento y estimaciones probabilísticas. Las reglas priorizan reactivación según días desde la última compra y después compra frecuente según `compras_12m`; los umbrales se pueden ajustar. La modalidad probabilística incluye dos apartados: estima `compro_ultima_semana` y clasifica su probabilidad como baja, media o alta; además, estima `tamano_preferido` como una clase multiclase. El modelo de porciones valida las clases Chico, Mediano y Grande, balancea sus pesos durante el entrenamiento y muestra las probabilidades de cada clase para compararlas, además de la porción con mayor probabilidad y su confianza. Cada estimación tiene su propia selección de predictores y evalúa registros reservados para prueba. Las acciones incluyen evidencia y pueden descargarse por cliente. Son hipótesis para medir, no efectos garantizados; la propensión no equivale a impacto incremental.

La pestaña **Decisiones** ofrece un selector entre Reactivación, Compra frecuente y Seguimiento. Reutiliza los umbrales y las mismas prioridades de la segmentación por reglas de la pestaña 5. Para el segmento seleccionado presenta cantidad de clientes, indicadores observados, acción recomendada, mensaje de ejemplo con `{{tamano_preferido}}`, justificación y una forma de medir el resultado. El mensaje es un borrador: el dataset no incluye canales ni permisos de contacto.

## Dataset disponible

Archivo: `data/dataset_clientes_sinteticos_ventas_pozole.csv`.

Hechos observados en el archivo actual:

- 1.200 filas y 12 columnas.
- No hay valores ausentes ni filas duplicadas exactas.
- `tamano_preferido`: Chico (316), Mediano (595), Grande (289).
- `compro_ultima_semana`: 0 (586), 1 (614).
- `fecha_alta` aparece en formato día/mes/año (por ejemplo `29/12/23`) y pandas la leerá como texto si no se convierte explícitamente.

| Campo | Uso posible |
|---|---|
| `cliente_id` | Identificador; excluir como predictor salvo justificación clara. |
| `edad`, `genero`, `ingreso_mensual` | Perfil del cliente. Confirmar unidades y significado con quien diseñó los datos. |
| `fecha_alta` | Antigüedad del cliente; convertir a fecha antes de calcular duración. |
| `visitas_restaurante_30d` | Actividad reciente en el restaurante durante 30 días. |
| `compras_12m`, `gasto_total_12m` | Frecuencia y gasto acumulado anual. |
| `dias_desde_ultima_compra` | Recencia de compra. |
| `tamano_preferido` | Categoría de porción preferida; posible objetivo multiclase si se define el momento de predicción. |
| `monto_ultima_compra` | Valor de la última compra, sujeto a confirmar unidad y periodo. |
| `compro_ultima_semana` | Indicador binario (0/1) de compra durante la última semana; posible objetivo de clasificación. |

Las interpretaciones se infieren de los nombres y ejemplos, no de un diccionario de datos validado. Confirmar unidades, ventanas temporales y cómo se generaron las etiquetas antes de sacar conclusiones.

## Hoja de ruta didáctica

### 1. Entender los datos

Identificar qué describe cada variable, sus tipos, rangos, frecuencias y calidad. Revisar el equilibrio de `compro_ultima_semana` y la distribución de `tamano_preferido`. No recomendar campañas todavía.

### 2. Limpiar los datos

Aunque el archivo no contiene ausencias ni duplicados exactos, revisar tipos, formatos de fecha, rangos imposibles y valores atípicos. No eliminar valores extremos solo por ser grandes. Conservar el DataFrame original y aplicar cambios a una copia documentando cada decisión.

### 3. Estadística descriptiva

La pestaña implementada calcula clientes, compradores y tasa de compra semanal, compras promedio por cliente, gasto promedio por cliente en 12 meses y monto promedio de última compra. Presenta numerador, denominador o población de referencia y periodo para cada indicador. Incluye comparación por tamaño preferido o género y distribuciones de tamaño y visitas recientes. No trata el gasto acumulado como ingreso de una sola compra ni atribuye causalidad a las diferencias observadas.

### 4. Estadística predictiva

La pestaña implementada permite explorar `compro_ultima_semana` como objetivo binario y `tamano_preferido` como objetivo multiclase. Antes de interpretar el resultado, definir qué información estaría disponible al momento de predecir. `cliente_id` y `fecha_alta` se excluyen; el historial se excluye por defecto por riesgo de fuga temporal y se puede incluir tras revisarlo. El preprocesamiento se ajusta solo en entrenamiento, y la evaluación usa una partición de prueba estratificada y reproducible con matriz de confusión, exactitud, exactitud balanceada, precisión, recall, F1 y ROC-AUC binaria. Los resultados sintéticos son educativos, no una previsión validada para operar el negocio.

### 5. Estadística prescriptiva

La pestaña implementada permite usar reglas transparentes por recencia y frecuencia o niveles de probabilidad de compra semanal. Muestra la regla, evidencia y acción sugerida por grupo, además de recomendaciones descargables por cliente. Las acciones son hipótesis que deben medirse con un grupo de comparación; no presentar una probabilidad como impacto causal ni inventar promociones, productos o beneficios.

### 6. Toma de decisiones

La pestaña implementada permite seleccionar Reactivación, Compra frecuente o Seguimiento y muestra el tamaño del grupo, indicadores descriptivos, acción recomendada, ejemplo de mensaje, justificación, regla de asignación y métrica para evaluar la prueba. Los umbrales se comparten con la pestaña 5. Distinguir los resultados observados de los efectos que se esperan comprobar; validar canal y permisos antes de usar mensajes.

## Entorno y ejecución

Ejecutar desde la carpeta que contiene `app.py`:

```bash
python -m venv pozole_train
```

Activar el entorno:

```bash
# macOS / Linux
source pozole_train/bin/activate
```

```powershell
# Windows PowerShell
pozole_train\Scripts\Activate.ps1
```

Instalar dependencias y ejecutar:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

En este proyecto existe un entorno `pozole_train`; la guía no asume que sea necesario recrearlo. `app.py` usa una ruta relativa al directorio de trabajo. Iniciar Streamlit desde la raíz del proyecto para que encuentre el CSV.

## Guía para agentes de programación

1. Leer esta guía y revisar el árbol y el contenido real de los archivos antes de editar.
2. Respetar el flujo didáctico por etapas y limitar cada cambio a la etapa solicitada.
3. Preservar el CSV original; crear copias o variables separadas para las transformaciones.
4. Usar imports explícitos, nombres descriptivos en español y claves únicas para widgets.
5. No afirmar validez predictiva o causal con el dataset sintético.
6. Comprobar que el nombre y la ruta del dataset coinciden con `app.py`; resolver la ruta relativa al archivo como mejora futura si se necesita ejecutar desde otros directorios.
