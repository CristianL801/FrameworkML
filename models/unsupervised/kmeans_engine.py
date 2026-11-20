from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from models.base import BaseModel


class KMeansEngine(BaseModel):
    """
    Motor No Supervisado basado en K-Means.
    Determina automáticamente el número óptimo de clusters (k),
    evalúa métricas internas y exporta gráficos de diagnóstico en 300 DPI.
    """

    def __init__(self, k_range=range(2, 11), random_state=42, reports_dir="reports"):
        self.k_range = list(k_range)
        self.random_state = random_state
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.best_k = None
        self.model = None
        self.labels_ = None
        self.metrics_ = {}
        self.inertias_ = []
        self.silhouettes_ = []

    def fit(self, X, y=None):
        """
        Busca el k óptimo en k_range usando Silhouette Score,
        entrena el modelo final y genera los gráficos de diagnóstico.
        """
        max_k = min(max(self.k_range), len(X) - 1)
        valid_ks = [k for k in self.k_range if 2 <= k <= max_k]

        self.inertias_ = []
        self.silhouettes_ = []
        best_score = -1.0

        for k in valid_ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = km.fit_predict(X)
            inertia = km.inertia_
            sil = silhouette_score(X, labels)

            self.inertias_.append(inertia)
            self.silhouettes_.append(sil)

            if sil > best_score:
                best_score = sil
                self.best_k = k
                self.model = km
                self.labels_ = labels

        # Métricas internas del modelo ganador
        self.metrics_ = {
            "best_k": self.best_k,
            "silhouette": silhouette_score(X, self.labels_),
            "calinski_harabasz": calinski_harabasz_score(X, self.labels_),
            "davies_bouldin": davies_bouldin_score(X, self.labels_),
        }

        print(f"\n[KMeansEngine] k óptimo seleccionado: {self.best_k}")
        print(f" - Silhouette Score:        {self.metrics_['silhouette']:.4f} (más cercano a 1 es mejor)")
        print(f" - Calinski-Harabasz Index: {self.metrics_['calinski_harabasz']:.2f} (más alto es mejor)")
        print(f" - Davies-Bouldin Index:    {self.metrics_['davies_bouldin']:.4f} (más bajo es mejor)")

        self._plot_elbow_silhouette(valid_ks)
        self._plot_pca_clusters(X)
        return self

    def predict(self, X):
        """Asigna el cluster más cercano a nuevas observaciones."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado. Ejecuta fit(X) primero.")
        return self.model.predict(X)

    def score(self, X, y=None):
        """Retorna el Silhouette Score como métrica principal."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado. Ejecuta fit(X) primero.")
        labels = self.model.predict(X)
        return silhouette_score(X, labels)

    def save(self, filepath="reports/kmeans_model.joblib"):
        """Guarda el modelo entrenado en disco."""
        if self.model is None:
            raise RuntimeError("No hay modelo entrenado para guardar.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[KMeansEngine] Modelo guardado en: '{path}'")

    def _plot_elbow_silhouette(self, ks):
        """Exporta gráfico doble: Método del Codo (Inercia) y Silhouette Score."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(ks, self.inertias_, marker="o", color="#2c7bb6")
        axes[0].set_title("Método del Codo (Inercia)")
        axes[0].set_xlabel("Número de Clusters (k)")
        axes[0].set_ylabel("Inercia")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(ks, self.silhouettes_, marker="o", color="#d7191c")
        axes[1].axvline(self.best_k, color="gray", linestyle="--", label=f"k óptimo = {self.best_k}")
        axes[1].set_title("Silhouette Score por k")
        axes[1].set_xlabel("Número de Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.reports_dir / "kmeans_elbow_silhouette.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_pca_clusters(self, X):
        """Proyecta los datos a 2D con PCA y grafica los clusters resultantes."""
        if X.shape[1] < 2:
            return

        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_ * 100

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels_, cmap="viridis", alpha=0.75, edgecolor="k", s=45)
        plt.colorbar(scatter, label="Cluster")
        plt.title(f"Clusters K-Means (k={self.best_k}) - Proyección PCA 2D")
        plt.xlabel(f"PC1 ({var_exp[0]:.1f}% varianza)")
        plt.ylabel(f"PC2 ({var_exp[1]:.1f}% varianza)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.reports_dir / "kmeans_pca_2d.png", dpi=300, bbox_inches="tight")
        plt.close()