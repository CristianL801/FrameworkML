from pathlib import Path
import joblib
import numpy as np
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


class LVQ1Model(BaseModel):
    """
    Learning Vector Quantization (LVQ1) para clasificación.
    Ajusta prototipos por clase acercándolos si coinciden y alejándolos si fallan.
    """

    def __init__(self, prototypes_per_class=2, lr=0.1, epochs=50, random_state=42):
        self.prototypes_per_class = prototypes_per_class
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.prototypes_ = None
        self.prototype_labels_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LVQ1Model requiere 'y' para entrenar.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        classes = np.unique(y)
        protos, labels = [], []
        for c in classes:
            X_c = X[y == c]
            n_pick = min(self.prototypes_per_class, len(X_c))
            idx = rng.choice(len(X_c), size=n_pick, replace=False)
            protos.append(X_c[idx].copy())
            labels.extend([c] * n_pick)

        self.prototypes_ = np.vstack(protos)
        self.prototype_labels_ = np.array(labels)

        for epoch in range(self.epochs):
            alpha = self.lr * (1.0 - epoch / max(self.epochs, 1))
            perm = rng.permutation(len(X))
            for i in perm:
                x_i, y_i = X[i], y[i]
                dists = np.linalg.norm(self.prototypes_ - x_i, axis=1)
                bmu = np.argmin(dists)
                if self.prototype_labels_[bmu] == y_i:
                    self.prototypes_[bmu] += alpha * (x_i - self.prototypes_[bmu])
                else:
                    self.prototypes_[bmu] -= alpha * (x_i - self.prototypes_[bmu])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.prototypes_[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        return self.prototype_labels_[nearest]

    def score(self, X, y=None):
        if y is None:
            raise ValueError("Se requiere 'y' para calcular el score.")
        return accuracy_score(y, self.predict(X))

    def save(self, filepath="reports/lvq1_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[LVQ1Model] Modelo guardado en: '{path}'")


class DANNModel(BaseModel):
    """
    Discriminant Adaptive Nearest Neighbor (DANN) para clasificación.
    Adapta la métrica de distancia localmente usando matrices de dispersión
    intra-clase (W) y entre-clases (B).
    """

    def __init__(self, n_neighbors=5, neighborhood_size=50, epsilon=1.0):
        self.n_neighbors = n_neighbors
        self.neighborhood_size = neighborhood_size
        self.epsilon = epsilon
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("DANNModel requiere 'y' para entrenar.")
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def _compute_local_metric(self, x0):
        n_samples, n_features = self.X_train_.shape
        k_local = min(self.neighborhood_size, n_samples)

        euclid = np.linalg.norm(self.X_train_ - x0, axis=1)
        idx = np.argsort(euclid)[:k_local]
        X_loc = self.X_train_[idx]
        y_loc = self.y_train_[idx]

        classes = np.unique(y_loc)
        if len(classes) < 2:
            return np.eye(n_features)

        mean_global = X_loc.mean(axis=0)
        W = np.zeros((n_features, n_features))
        B = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X_loc[y_loc == c]
            pi_c = len(X_c) / k_local
            mean_c = X_c.mean(axis=0)
            diff_w = X_c - mean_c
            W += pi_c * (diff_w.T @ diff_w) / max(len(X_c), 1)
            diff_b = (mean_c - mean_global).reshape(-1, 1)
            B += pi_c * (diff_b @ diff_b.T)

        # Regularización numérica para garantizar invertibilidad de W
        W += 1e-4 * np.eye(n_features)
        eigvals, eigvecs = np.linalg.eigh(W)
        eigvals = np.maximum(eigvals, 1e-4)
        W_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        B_star = W_inv_sqrt @ B @ W_inv_sqrt
        sigma = W_inv_sqrt @ (B_star + self.epsilon * np.eye(n_features)) @ W_inv_sqrt
        return sigma

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for x0 in X:
            sigma = self._compute_local_metric(x0)
            diffs = self.X_train_ - x0
            dists = np.einsum("ij,jk,ik->i", diffs, sigma, diffs)
            nn_idx = np.argsort(dists)[: self.n_neighbors]
            vals, counts = np.unique(self.y_train_[nn_idx], return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)

    def score(self, X, y=None):
        if y is None:
            raise ValueError("Se requiere 'y' para calcular el score.")
        return accuracy_score(y, self.predict(X))

    def save(self, filepath="reports/dann_model.joblib"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[DANNModel] Modelo guardado en: '{path}'")