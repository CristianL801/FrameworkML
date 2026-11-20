# Framework Modular de Machine Learning

Framework orientado a objetos en Python para ingesta automática, análisis exploratorio (EDA), preprocesamiento adaptativo sin *Data Leakage* y evaluación de modelos supervisados y no supervisados.

## Estructura del Proyecto
- `core/`: Ingesta (`loader.py`), diagnóstico estadístico (`eda.py`) y preprocesamiento adaptativo (`preprocessor.py`).
- `models/`: Clase base abstracta (`base.py`), modelos supervisados (`CART`, `k-NN`, `Ridge/Logística`) y motor no supervisado (`KMeansEngine`).
- `evaluation/`: Cálculo automático de métricas y exportación de gráficos en 300 DPI (`metrics.py`).
- `main.py`: Orquestador del flujo completo.

## Instalación y Uso
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt