from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.base import BaseModel


class TreeModel(BaseModel):
    """
    Operaciones comunes de modelos basados en árboles.
    Soporta clasificación y regresión heredando el contrato de BaseModel.
    """

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TreeModel requiere una variable objetivo 'y' para entrenar.")
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

    def save(self, filepath=None):
        if filepath is None:
            filepath = self.reports_dir / f"{self.name}_model.joblib"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[{self.name.upper()}] Modelo guardado en: '{path}'")

    def plot_feature_importance(self, feature_names, filename=None):
        """Exporta el gráfico de importancia de variables en 300 DPI."""
        if filename is None:
            filename = f"{self.name}_feature_importance.png"
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 variables

        plt.figure(figsize=(8, 5))
        plt.barh(range(len(indices)), importances[indices][::-1], color="#2c7bb6")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
        plt.xlabel("Importancia (Gini / Varianza)")
        plt.title(f"Top Variables más Importantes - {self.name.upper()} ({self.task})")
        plt.tight_layout()
        plt.savefig(self.reports_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


class CARTModel(TreeModel):
    """Árbol de decisión individual."""

    name = "cart"

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


class RandomForestModel(TreeModel):
    """Bosque de árboles para clasificación y regresión."""

    name = "random_forest"

    def __init__(self, task="classification", max_depth=5, random_state=42,
                 reports_dir="reports", n_estimators=100):
        self.task = task.lower()
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.task == "classification":
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        elif self.task == "regression":
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        else:
            raise ValueError("Tarea no soportada.")
