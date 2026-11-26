import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from core.eda import EDAnalyzer
from core.loader import DataLoader
from core.preprocessor import AdaptivePreprocessor
from evaluation.metrics import ModelEvaluator
from evaluation.visualizer import ModelVisualizer
from models.supervised.linear_models import LinearModel
from models.supervised.prototype_models import DANNModel, KNNModel, LVQ1Model
from models.supervised.spline_models import GAMModel, MARSModel
from models.supervised.tree_models import CARTModel
from models.unsupervised.kmeans_engine import KMeansEngine


def run_pipeline(file_path: str, target_col: str = None):
    print(f"\n=== Iniciando Framework ML sobre: {file_path} ===")

    # 1. Ingesta Inteligente
    loader = DataLoader(file_path)
    df = loader.load_data()
    if df is None:
        return
    detected_target, task_type = loader.detect_target(df, target_col=target_col)

    # 2. Análisis Exploratorio Automatizado (EDA)
    eda = EDAnalyzer(df, reports_dir="reports")
    eda.run_full_eda()

    # 3. Enrutamiento: No Supervisado vs Supervisado
    if detected_target is None or task_type == "unsupervised":
        print("\n>>> Ruta seleccionada: APRENDIZAJE NO SUPERVISADO (K-Means)")
        prep = AdaptivePreprocessor(model_name="kmeans")
        X_clean = prep.fit_transform(df)

        engine = KMeansEngine(reports_dir="reports")
        engine.fit(X_clean)
        engine.save()
        return

    # Ruta Supervisada (Clasificación o Regresión)
    print(f"\n>>> Ruta seleccionada: APRENDIZAJE SUPERVISADO ({task_type.upper()})")
    X = df.drop(columns=[detected_target])
    y = df[detected_target]

    stratify = y if (task_type == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    models_to_run = {
        "cart": CARTModel(task=task_type, reports_dir="reports"),
        "knn": KNNModel(task=task_type),
        "ridge": LinearModel(task=task_type),
        "mars": MARSModel(task=task_type),
        "gam": GAMModel(task=task_type),
    }

    if task_type == "classification":
        models_to_run["lvq1"] = LVQ1Model()
        models_to_run["dann"] = DANNModel()

    evaluator = ModelEvaluator(task=task_type, reports_dir="reports")
    results_summary = {}

    for name, model in models_to_run.items():
        print(f"\n--- Entrenando modelo: {name.upper()} ---")
        prep = AdaptivePreprocessor(model_name=name)
        X_train_clean = prep.fit_transform(X_train)
        X_test_clean = prep.transform(X_test)

        model.fit(X_train_clean, y_train)
        y_pred = model.predict(X_test_clean)

        results_summary[name.upper()] = evaluator.evaluate(y_test, y_pred, model_name=name)
        model.save()

        if name == "cart":
            model.plot_feature_importance(prep.feature_names_out)

    print("\n=== Resumen Comparativo de Modelos ===")
    summary_df = pd.DataFrame(results_summary).T
    print(summary_df.round(4))

    # Gráfico comparativo consolidado
    metric_key = "f1_score" if task_type == "classification" else "r2"
    comparison_scores = {m: vals[metric_key] for m, vals in results_summary.items()}
    visualizer = ModelVisualizer(reports_dir="reports")
    visualizer.plot_model_comparison(comparison_scores, metric_name=metric_key)


def create_demo_csv(path="data/demo_iris.csv"):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    iris.frame.to_csv(out, index=False)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Modular de Machine Learning")
    parser.add_argument("--file", type=str, default=None, help="Ruta al dataset (CSV, TSV o Parquet)")
    parser.add_argument("--target", type=str, default=None, help="Nombre opcional de la columna objetivo")
    args = parser.parse_args()

    dataset_path = args.file if args.file else create_demo_csv()
    run_pipeline(dataset_path, target_col=args.target)