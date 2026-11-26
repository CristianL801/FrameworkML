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