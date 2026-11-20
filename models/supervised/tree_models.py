from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.base import BaseModel


class CARTModel(BaseModel):
    """
    Modelo supervisado basado en Árboles de Decisión (CART).
    Soporta clasificación y regresión heredando el contrato de BaseModel.
    """

    def __init__(self, task="classification", max_depth=5, random_state=42, reports_dir="reports"):
        self.task = task.lower()
        self.max_depth = max_depth
        self.random_state = random_state
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        if self.task == "classification":
            self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        elif self.task == "regression":
            self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        else:
            raise ValueError("El parámetro 'task' debe ser 'classification' o 'regression'.")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CARTModel requiere una variable objetivo 'y' para entrenar.")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y=None):
        if y is None:
            raise ValueError("Se requiere 'y' para calcular el score.")
        y_pred = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, y_pred)
        return r2_score(y, y_pred)

    def save(self, filepath="reports/cart_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[CARTModel] Modelo guardado en: '{path}'")

    def plot_feature_importance(self, feature_names, filename="cart_feature_importance.png"):
        """Exporta el gráfico de importancia de variables en 300 DPI."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 variables

        plt.figure(figsize=(8, 5))
        plt.barh(range(len(indices)), importances[indices][::-1], color="#2c7bb6")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
        plt.xlabel("Importancia (Gini / Varianza)")
        plt.title(f"Top Variables más Importantes - CART ({self.task})")
        plt.tight_layout()
        plt.savefig(self.reports_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()