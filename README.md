# Clasificador de Calidad de Vinos con KNN

Este proyecto desarrolla un modelo de Machine Learning capaz de clasificar la calidad de vinos tintos a partir de sus características químicas. Para ello se utiliza el algoritmo K-Vecinos más Cercanos, conocido como KNN, uno de los modelos supervisados más intuitivos para tareas de clasificación.

El objetivo es analizar si, a partir de variables como la acidez, el nivel de alcohol, los sulfatos, el pH o la densidad, es posible predecir si un vino pertenece a una categoría de baja, media o alta calidad.

## Objetivo del Proyecto

El propósito principal de este proyecto es entrenar, evaluar y optimizar un modelo KNN para predecir la calidad de un vino tinto utilizando datos reales del Wine Quality Data Set del UCI Machine Learning Repository.

La variable objetivo se representa como `label`:

| Label | Categoría |
| --- | --- |
| `0` | Baja calidad |
| `1` | Calidad media |
| `2` | Alta calidad |

El dataset original contiene la variable `quality`, que fue transformada en tres clases para facilitar el problema de clasificación:

- Calidad baja: vinos con `quality <= 4`.
- Calidad media: vinos con `quality` entre `5` y `6`.
- Calidad alta: vinos con `quality >= 7`.

## Dataset

El conjunto de datos contiene mediciones fisicoquímicas de vinos tintos. Cada fila representa un vino y cada columna describe una característica química.

Variables utilizadas:

| Variable | Descripción |
| --- | --- |
| `fixed acidity` | Acidez fija del vino |
| `volatile acidity` | Acidez volátil |
| `citric acid` | Cantidad de ácido cítrico |
| `residual sugar` | Azúcar residual |
| `chlorides` | Nivel de cloruros |
| `free sulfur dioxide` | Dióxido de azufre libre |
| `total sulfur dioxide` | Dióxido de azufre total |
| `density` | Densidad del vino |
| `pH` | Nivel de pH |
| `sulphates` | Sulfatos |
| `alcohol` | Porcentaje de alcohol |

## Metodología

El flujo de trabajo seguido en el proyecto fue el siguiente:

1. Carga del dataset con Pandas.
2. Exploración inicial de la estructura de los datos.
3. Revisión de tipos de datos, valores nulos y duplicados.
4. Transformación de la variable `quality` en la variable objetivo `label`.
5. Análisis visual de la distribución de clases.
6. Exploración de variables químicas relevantes.
7. Separación entre variables predictoras y variable objetivo.
8. División del dataset en entrenamiento y prueba.
9. Escalado de variables con `StandardScaler`.
10. Entrenamiento de un modelo KNN inicial.
11. Evaluación mediante:
   - `accuracy_score`
   - `confusion_matrix`
   - `classification_report`
12. Optimización del valor de `k` probando valores entre 1 y 20.
13. Visualización de la relación entre accuracy y `k`.
14. Comparación con modelos de referencia.
15. Guardado del modelo final.
16. Creación de una función para predecir la calidad de nuevos vinos.

## ¿Por qué es necesario escalar los datos?

KNN clasifica una observación según la distancia con sus vecinos más cercanos. Por esta razón, las variables deben estar en una escala comparable.

Por ejemplo, variables como `total sulfur dioxide` pueden tener valores mucho mayores que variables como `chlorides` o `density`. Si no se escalan los datos, las variables con mayor rango numérico pueden dominar el cálculo de distancia y afectar negativamente el rendimiento del modelo.

Para evitarlo, se utilizó `StandardScaler`.

## Resultados del Modelo

Se probaron distintos valores de `k` entre 1 y 20. El mejor valor encontrado para KNN uniforme fue:

```text
k = 19
```

Además, se evaluó una variante de KNN con ponderación por distancia, donde los vecinos más cercanos tienen más influencia que los más alejados.

Resultados principales:

| Modelo | Accuracy | Precision weighted | Recall weighted | F1 weighted | F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| KNN k=19 distance | 0.8566 | 0.8093 | 0.8566 | 0.8239 | 0.4898 |
| KNN k=19 | 0.8529 | 0.8063 | 0.8529 | 0.8168 | 0.4750 |
| RandomForest | 0.8346 | 0.7821 | 0.8346 | 0.8037 | 0.4626 |
| SVC RBF | 0.6912 | 0.8364 | 0.6912 | 0.7290 | 0.5581 |

El mejor resultado por accuracy fue obtenido por `KNN k=19 distance`, alcanzando una precisión aproximada del 85.66%.

## Función de Predicción

El proyecto incluye una función que permite introducir las características químicas de un vino y obtener una predicción de su calidad.

Ejemplo:

```python
predict_wine_quality([
    7.4, 0.7, 0.0, 1.9, 0.076,
    11.0, 34.0, 0.9978, 3.51, 0.56, 9.4
])
```

Resultado:

```text
Este vino probablemente sea de calidad media 🍷
```

## Estructura del Proyecto

```text
Dragcessa1998-Proyecto-K-vecinos-m-s-Cercanos-main/
│
├── data/
│   ├── raw/
│   │   └── winequality-red.csv
│   ├── processed/
│   │   ├── winequality_red_labeled.csv
│   │   ├── knn_k_results.csv
│   │   └── model_results.csv
│   └── interim/
│
├── models/
│   ├── wine_quality_knn.pkl
│   └── wine_quality_best_model.pkl
│
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
│
├── requirements.txt
├── README.md
└── README.es.md
```

## Archivos Principales

| Archivo | Descripción |
| --- | --- |
| `src/explore.ipynb` | Notebook con el análisis completo, visualizaciones, entrenamiento y evaluación |
| `src/app.py` | Script reproducible para entrenar el modelo y guardar resultados |
| `data/raw/winequality-red.csv` | Dataset original |
| `data/processed/winequality_red_labeled.csv` | Dataset procesado con la variable objetivo |
| `data/processed/knn_k_results.csv` | Resultados de accuracy y métricas para cada valor de `k` |
| `data/processed/model_results.csv` | Comparación entre modelos |
| `models/wine_quality_knn.pkl` | Modelo KNN final |
| `models/wine_quality_best_model.pkl` | Mejor modelo global |

## Cómo Ejecutar el Proyecto

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el script principal:

```bash
python src/app.py
```

También se puede abrir y ejecutar el notebook:

```text
src/explore.ipynb
```

## Insight del Proyecto

A partir de variables químicas como el alcohol, la acidez volátil, los sulfatos y la densidad, un modelo de Machine Learning puede aproximar la calidad de un vino tinto con una precisión cercana al 86%.

Esto demuestra cómo la inteligencia artificial puede apoyar tareas de clasificación que tradicionalmente dependerían de evaluación experta, aunque el modelo también revela una limitación importante: las clases minoritarias, como vinos de baja calidad, son más difíciles de predecir cuando hay pocos ejemplos disponibles.

## Tecnologías Utilizadas

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Pickle

## Conclusión

El modelo KNN demostró ser una alternativa efectiva para clasificar vinos tintos según su composición química. La optimización del valor de `k` permitió mejorar el rendimiento, y la variante con ponderación por distancia obtuvo el mejor resultado general.

El proyecto también evidencia la importancia del escalado de variables en algoritmos basados en distancia y la necesidad de revisar métricas más allá del accuracy cuando las clases están desbalanceadas.


## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.
