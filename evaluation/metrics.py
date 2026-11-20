from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class ModelEvaluator:
    """
    Evalúa modelos supervisados (clasificación y regresión)
    y exporta gráficos de diagnóstico en 300 DPI.
    """

    def __init__(self, task="classification", reports_dir="reports"):
        self.task = task.lower()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_pred, model_name="model"):
        """Calcula métricas y genera gráficos según el tipo de tarea."""
        if self.task == "classification":
            return self._evaluate_classification(y_true, y_pred, model_name)
        elif self.task == "regression":
            return self._evaluate_regression(y_true, y_pred, model_name)
        else:
            raise ValueError("El parámetro 'task' debe ser 'classification' o 'regression'.")

    def _evaluate_classification(self, y_true, y_pred, model_name):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        print(f"\n[Evaluation - {model_name} (Clasificación)]")
        for k, v in metrics.items():
            print(f" - {k.capitalize()}: {v:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        return metrics

    def _evaluate_regression(self, y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics = {
            "rmse": rmse,
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        print(f"\n[Evaluation - {model_name} (Regresión)]")
        print(f" - RMSE: {metrics['rmse']:.4f}")
        print(f" - MAE:  {metrics['mae']:.4f}")
        print(f" - R²:   {metrics['r2']:.4f}")

        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k", color="#2c7bb6")
        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
        plt.plot(lims, lims, "r--", label="Predicción Ideal")
        plt.title(f"Real vs Predicho - {model_name}")
        plt.xlabel("Valor Real")
        plt.ylabel("Predicción")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"{model_name}_real_vs_pred.png", dpi=300, bbox_inches="tight")
        plt.close()

        return metrics