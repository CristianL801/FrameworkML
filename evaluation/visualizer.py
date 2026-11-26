from pathlib import Path
import matplotlib.pyplot as plt


class ModelVisualizer:
    """
    Genera gráficos comparativos consolidados entre los modelos evaluados
    y los exporta en alta resolución (300 DPI).
    """

    def __init__(self, reports_dir="reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def plot_model_comparison(self, results_dict, metric_name="score", filename="model_comparison.png"):
        """
        Recibe un diccionario {nombre_modelo: valor_metrica} y exporta
        un gráfico de barras comparativo ordenado.
        """
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