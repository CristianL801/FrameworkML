from pathlib import Path
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from models.base import BaseModel


class HingeBasisTransformer(BaseEstimator, TransformerMixin):
    """
    Expansión en funciones bisagra (hinge): max(0, x - t) y max(0, t - x)
    usando cuantiles como nodos (knots) para aproximar MARS.
    """

    def __init__(self, n_knots=4):
        self.n_knots = n_knots
        self.knots_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        quantiles = np.linspace(0.2, 0.8, self.n_knots)
        self.knots_ = np.quantile(X, quantiles, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        features = [X]
        for k in range(self.knots_.shape[0]):
            t = self.knots_[k]
            features.append(np.maximum(0.0, X - t))
            features.append(np.maximum(0.0, t - X))
        return np.hstack(features)


class MARSModel(BaseModel):
    """
    Multivariate Adaptive Regression Splines (MARS) mediante funciones bisagra
    y regularización L2 (Ridge / LogisticRegression).
    """

    def __init__(self, task="regression", n_knots=4, alpha=1.0, random_state=42):
        self.task = task.lower()
        self.n_knots = n_knots
        self.alpha = alpha
        self.random_state = random_state

        if self.task == "classification":
            estimator = LogisticRegression(C=1.0 / self.alpha, max_iter=1000, random_state=self.random_state)
        elif self.task == "regression":
            estimator = Ridge(alpha=self.alpha, random_state=self.random_state)
        else:
            raise ValueError("El parámetro 'task' debe ser 'classification' o 'regression'.")

        self.model = Pipeline([
            ("hinge", HingeBasisTransformer(n_knots=self.n_knots)),
            ("estimator", estimator),
        ])

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("MARSModel requiere 'y' para entrenar.")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y=None):
        if y is None:
            raise ValueError("Se requiere 'y' para calcular el score.")
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred) if self.task == "classification" else r2_score(y, y_pred)

    def save(self, filepath="reports/mars_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[MARSModel] Modelo guardado en: '{path}'")


class GAMModel(BaseModel):
    """
    Generalized Additive Model (GAM) usando B-splines (SplineTransformer)
    más estimador lineal regularizado para capturar relaciones no lineales suaves.
    """

    def __init__(self, task="regression", n_knots=5, degree=3, alpha=1.0, random_state=42):
        self.task = task.lower()
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        self.random_state = random_state

        if self.task == "classification":
            estimator = LogisticRegression(C=1.0 / self.alpha, max_iter=1000, random_state=self.random_state)
        elif self.task == "regression":
            estimator = Ridge(alpha=self.alpha, random_state=self.random_state)
        else:
            raise ValueError("El parámetro 'task' debe ser 'classification' o 'regression'.")

        self.model = Pipeline([
            ("splines", SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False)),
            ("estimator", estimator),
        ])

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("GAMModel requiere 'y' para entrenar.")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y=None):
        if y is None:
            raise ValueError("Se requiere 'y' para calcular el score.")
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred) if self.task == "classification" else r2_score(y, y_pred)

    def save(self, filepath="reports/gam_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[GAMModel] Modelo guardado en: '{path}'")