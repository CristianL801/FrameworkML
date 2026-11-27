from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, r2_score
from models.base import BaseModel


class LinearModel(BaseModel):
    """
    Modelo lineal regularizado (L2).
    Usa LogisticRegression para clasificación y Ridge para regresión.
    """

    def __init__(self, task="classification", alpha=1.0, random_state=42):
        self.task = task.lower()
        self.alpha = alpha
        self.random_state = random_state

        if self.task == "classification":
            self.model = LogisticRegression(C=1.0 / self.alpha, max_iter=1000, random_state=self.random_state)
        elif self.task == "regression":
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        else:
            raise ValueError("El parámetro 'task' debe ser 'classification' o 'regression'.")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LinearModel requiere una variable objetivo 'y' para entrenar.")
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

    def save(self, filepath="reports/linear_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[LinearModel] Modelo guardado en: '{path}'")


class LassoModel(LinearModel):
    """Regresión lineal con regularización L1."""

    def __init__(self, alpha=0.1, random_state=42):
        self.task = "regression"
        self.alpha = alpha
        self.random_state = random_state
        self.model = Lasso(alpha=alpha, max_iter=10000)

    def save(self, filepath="reports/lasso_model.joblib"):
        super().save(filepath)