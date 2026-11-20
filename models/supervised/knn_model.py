from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from models.base import BaseModel


class KNNModel(BaseModel):
    """
    Modelo supervisado basado en k-Vecinos Más Cercanos (k-NN).
    Requiere escalado previo por su dependencia en distancias euclidianas.
    """

    def __init__(self, task="classification", n_neighbors=5, weights="distance"):
        self.task = task.lower()
        self.n_neighbors = n_neighbors
        self.weights = weights

        if self.task == "classification":
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        elif self.task == "regression":
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
        else:
            raise ValueError("El parámetro 'task' debe ser 'classification' o 'regression'.")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("KNNModel requiere una variable objetivo 'y' para entrenar.")
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

    def save(self, filepath="reports/knn_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[KNNModel] Modelo guardado en: '{path}'")