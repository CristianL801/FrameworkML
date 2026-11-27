from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve


class ModelVisualizer:
    """
    Genera gráficos comparativos y diagnósticos avanzados en alta resolución (300 DPI)
    para Clasificación, Regresión y Clustering (K-Means).
    """

    def __init__(self, reports_dir="reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def plot_model_comparison(self, results_dict, metric_name="score", filename="model_comparison.png"):
        """Gráfico de barras comparativo ordenado entre modelos."""
        if not results_dict:
            return

        sorted_items = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        models = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(models, scores, color="#2c7bb6", edgecolor="black", alpha=0.85)
        plt.title(f"Comparativa de Modelos ({metric_name.upper()})")
        plt.xlabel("Modelo")
        plt.ylabel(metric_name.upper())
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, val in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        filepath = self.reports_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Gráfico comparativo guardado en: '{filepath}'")

    # =========================================================================
    # 1. CLASIFICACIÓN: Confusión normalizada, ROC multiclase, PR y Learning Curve
    # =========================================================================
    def plot_classification_diagnostics(self, estimator, X_train, y_train, X_test, y_test, model_name="best_clf"):
        """
        Recibe el estimador interno (ej. model.model) ya entrenado y genera:
        1) Matriz de confusión normalizada
        2) Curvas ROC multiclase con AUC
        3) Curvas Precision-Recall
        4) Curva de aprendizaje (Error vs. Tamaño muestral)
        """
        y_pred = estimator.predict(X_test)
        classes = np.unique(np.concatenate([y_train, y_test]))
        n_classes = len(classes)

        # 1) Matriz de confusión normalizada
        cm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Matriz de Confusión Normalizada - {model_name.upper()}")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"{model_name}_confusion_norm.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Obtener scores/probabilidades para ROC y Precision-Recall
        y_score = None
        if hasattr(estimator, "predict_proba"):
            y_score = estimator.predict_proba(X_test)
        elif hasattr(estimator, "decision_function"):
            y_score = estimator.decision_function(X_test)

        if y_score is not None:
            y_test_bin = label_binarize(y_test, classes=classes)
            if n_classes == 2 and y_test_bin.shape[1] == 1:
                y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
            if y_score.ndim == 1:
                y_score = np.vstack([1 - y_score, y_score]).T

            # 2) Curvas ROC multiclase con AUC
            plt.figure(figsize=(7, 5))
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"Clase {cls} (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.title(f"Curvas ROC Multiclase - {model_name.upper()}")
            plt.xlabel("Tasa de Falsos Positivos (FPR)")
            plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.reports_dir / f"{model_name}_roc_multiclass.png", dpi=300, bbox_inches="tight")
            plt.close()

            # 3) Curvas Precision-Recall
            plt.figure(figsize=(7, 5))
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
                plt.plot(rec, prec, lw=2, label=f"Clase {cls} (AP = {ap:.3f})")
            plt.title(f"Curvas Precision-Recall - {model_name.upper()}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.reports_dir / f"{model_name}_pr_curves.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 4) Curva de aprendizaje (Error = 1 - Accuracy vs. Tamaño muestral)
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                estimator, X_train, y_train, cv=3, scoring="accuracy",
                train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1
            )
            train_err = 1.0 - train_scores.mean(axis=1)
            val_err = 1.0 - val_scores.mean(axis=1)

            plt.figure(figsize=(7, 5))
            plt.plot(train_sizes, train_err, "o-", label="Error Entrenamiento")
            plt.plot(train_sizes, val_err, "s-", label="Error Validación (CV)")
            plt.title(f"Curva de Aprendizaje - {model_name.upper()}")
            plt.xlabel("Tamaño muestral de entrenamiento")
            plt.ylabel("Error (1 - Accuracy)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.reports_dir / f"{model_name}_learning_curve.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        print(f"[Visualizer] Diagnósticos de clasificación guardados para '{model_name}'.")

    # =========================================================================
    # 2. REGRESIÓN: Real vs Predicho con banda de confianza + Residuales
    # =========================================================================
    def plot_regression_diagnostics(self, y_true, y_pred, model_name="best_reg"):
        """Genera Real vs Predicho (con banda ±1.96σ y métricas) + distribución de residuales."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residuals = y_true - y_pred

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        std_res = np.std(residuals)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Real vs Predicho + banda de confianza 95%
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        grid = np.linspace(min_v, max_v, 100)
        axes[0].scatter(y_true, y_pred, alpha=0.7, edgecolor="k", color="#2c7bb6")
        axes[0].plot(grid, grid, "r--", lw=2, label="Predicción ideal (y = ŷ)")
        axes[0].fill_between(grid, grid - 1.96 * std_res, grid + 1.96 * std_res,
                             color="red", alpha=0.12, label="Banda de confianza 95%")
        axes[0].set_title(f"Real vs. Predicho - {model_name.upper()}")
        axes[0].set_xlabel("Valor Real")
        axes[0].set_ylabel("Valor Predicho")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)
        axes[0].text(
            0.97, 0.05,
            f"R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=axes[0].transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        # Panel 2: Distribución de residuales
        sns.histplot(residuals, kde=True, ax=axes[1], color="#fdae61", edgecolor="black")
        axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
        axes[1].set_title(f"Distribución de Residuales - {model_name.upper()}")
        axes[1].set_xlabel("Residual (y - ŷ)")
        axes[1].set_ylabel("Frecuencia")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        filepath = self.reports_dir / f"{model_name}_regression_diagnostics.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Diagnóstico de regresión guardado en: '{filepath}'")

    # =========================================================================
    # 3. K-MEANS: Diagrama de siluetas individuales por clúster
    # =========================================================================
    def plot_kmeans_silhouette_detail(self, X_scaled, labels, filename="kmeans_silhouette_detail.png"):
        """Diagrama de siluetas individuales ordenadas por clúster con línea de corte promedio."""
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            return

        sil_avg = silhouette_score(X_scaled, labels)
        sample_sil = silhouette_samples(X_scaled, labels)

        plt.figure(figsize=(8, 5))
        y_lower = 10
        cmap = plt.cm.viridis

        for i in range(n_clusters):
            ith_vals = np.sort(sample_sil[labels == i])
            size_i = ith_vals.shape[0]
            y_upper = y_lower + size_i
            color = cmap(float(i) / n_clusters)

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                              facecolor=color, edgecolor=color, alpha=0.8)
            plt.text(-0.05, y_lower + 0.5 * size_i, f"C{i}", fontsize=9, fontweight="bold")
            y_lower = y_upper + 10

        plt.axvline(x=sil_avg, color="red", linestyle="--", lw=2,
                    label=f"Silueta promedio = {sil_avg:.3f}")
        plt.title(f"Diagrama de Siluetas Individuales (k = {n_clusters})")
        plt.xlabel("Coeficiente de Silueta")
        plt.ylabel("Clúster")
        plt.yticks([])
        plt.legend(loc="best")
        plt.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        filepath = self.reports_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Diagrama de siluetas guardado en: '{filepath}'")