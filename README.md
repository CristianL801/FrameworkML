# Framework Modular de Machine Learning

Framework orientado a objetos en Python para ingesta automática, análisis exploratorio (EDA), preprocesamiento adaptativo sin *Data Leakage* y evaluación de modelos supervisados y no supervisados.

## Estructura del Proyecto
- `data/`: Conjuntos de datos de prueba (clasificación, regresión y no supervisado).
- `reports/`: Salida automática de gráficos en 300 DPI, matriz de covarianza (`.csv`) y modelos serializados (`.joblib`).
- `core/`: Ingesta y detección automática de target (`loader.py`), diagnóstico estadístico y multicolinealidad (`eda.py`) y preprocesamiento adaptativo (`preprocessor.py`).
- `models/`:
  - `base.py`: Clase base abstracta (`BaseModel`) con interfaz polimórfica (`fit`, `predict`, `score`, `save`).
  - `supervised/`: Modelos lineales (`linear_models.py`), árboles (`tree_models.py`), splines y aditivos (`spline_models.py`: MARS, GAM) y prototipos/vecinos (`prototype_models.py`: k-NN, LVQ1, DANN).
  - `unsupervised/`: Motor de clustering K-Means++ con optimización de $k$ y proyección PCA 2D (`kmeans_engine.py`).
- `evaluation/`: Cálculo automático de métricas (`metrics.py`) y gráficos comparativos consolidados (`visualizer.py`).
- `main.py`: Orquestador CLI del flujo completo.

## Instalación
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para archivos Parquet, comprueba que el entorno tiene instalado `pyarrow`:

```powershell
python -m pip install pyarrow
python -m pip freeze | Set-Content -Encoding utf8 requirements.txt
```

## Ejecución

Desde la raíz del repositorio, con el entorno virtual activo:

```powershell
python main.py
python main.py --file "data/gym_members_exercise_tracking(supervisado).csv"
python main.py --file "data/gym_members_exercise_tracking(supervisado).csv" --target Experience_Level
python main.py --file "data/bodyfat(no_supervisado).csv"
python main.py --file "data/bodyfat(no_supervisado).csv" --unsupervised
```

En modo interactivo, Enter activa detección automática y `-` fuerza clustering. La detección automática es heurística: reconoce nombres convencionales de columnas objetivo o una última columna categórica; no puede conocer por sí sola el significado de todos los datasets. Se recomienda indicar el target ante ambigüedad.

## Flujo

1. `DataLoader`: lee los datos y determina la ruta.
2. `EDAnalyzer`: analiza calidad, covarianza y correlaciones.
3. `AdaptivePreprocessor`: imputa, codifica y escala según el modelo. En supervisado se ajusta solo sobre entrenamiento.
4. Los modelos implementan `BaseModel`; el orquestador entrena y evalúa.
5. `ModelEvaluator` y `ModelVisualizer` exportan métricas y gráficos a `reports/`.

Para K-Means se evalúa K de 2 a 10 cuando el tamaño de muestra lo permite. El codo se identifica por distancia máxima a la recta entre extremos (incluye K=1 para la inercia); la selección final maximiza la silueta. CHI y DBI se reportan para cada K válido. Los centroides se proyectan con el mismo PCA que las observaciones.

## Reportes

- `supervised_metrics.csv`, `model_comparison.png`, `cart_feature_importance.png`.
- `kmeans_metrics_by_k.csv`, `kmeans_labels.csv`, `kmeans_elbow_silhouette.png`, `kmeans_pca_2d.png` y `kmeans_silhouette_detail.png`.
- Los archivos de nombre fijo se sobrescriben en ejecuciones posteriores; conserva una copia si necesitas comparar datasets.

## Limitaciones de esta versión

La implementación MARS existente es una aproximación por funciones bisagra con nodos fijos. La selección de hiperparámetros y la validación cruzada con preprocesamiento por fold aún no están integradas. Los modelos guardados requieren aplicar el mismo preprocesamiento utilizado durante su entrenamiento.


### Modelos adicionales
- Random Forest: clasificación y regresión, 100 árboles y profundidad máxima 5.
- Lasso: regresión con regularización L1, alpha=0.1.
- La regresión logística se identifica como LOGISTIC en clasificación y Ridge como RIDGE en regresión.
- Estos hiperparámetros son iniciales, aún no se optimizan por validación cruzada.
