import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class ModelEvaluator:
    """
    Calcula y reporta métricas de desempeño para Clasificación y Regresión
    sin generar gráficos duplicados (delegados a ModelVisualizer).
    """

    def __init__(self, task="classification", reports_dir="reports"):
        self.task = task
        self.reports_dir = reports_dir

    def evaluate(self, y_true, y_pred, model_name="model"):
        if self.task == "classification":
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
            print(f"\n[Evaluation - {model_name} (Clasificación)]")
            for k, v in metrics.items():
                print(f"- {k.capitalize()}: {v:.4f}")
            return metrics

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"\n[Evaluation - {model_name} (Regresión)]")
        print(f"- RMSE: {rmse:.4f}")
        print(f"- MAE: {mae:.4f}")
        print(f"- R2: {r2:.4f}")
        return metrics