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
from models.supervised.linear_models import LinearModel, LassoModel
from models.supervised.prototype_models import DANNModel, KNNModel, LVQ1Model
from models.supervised.spline_models import GAMModel, MARSModel
from models.supervised.tree_models import CARTModel, RandomForestModel
from models.unsupervised.kmeans_engine import KMeansEngine


def run_pipeline(file_path: str, target_col: str = None, force_unsupervised=False):
    print(f"\n=== Iniciando Framework ML sobre: {file_path} ===")

    # 1. Ingesta Inteligente
    loader = DataLoader(file_path)
    df = loader.load_data()
    if df is None:
        return
    if force_unsupervised:
        detected_target, task_type = None, "unsupervised"
    else:
        detected_target, task_type = loader.detect_target(df, target_col=target_col)

    # 2. Análisis Exploratorio Automatizado (EDA)
    eda = EDAnalyzer(df, reports_dir="reports")
    eda.run_full_eda()

    visualizer = ModelVisualizer(reports_dir="reports")

    # 3. Enrutamiento: No Supervisado vs Supervisado
    if detected_target is None or task_type == "unsupervised":
        print("\n>>> Ruta seleccionada: APRENDIZAJE NO SUPERVISADO (K-Means)")
        prep = AdaptivePreprocessor(model_name="kmeans")
        X_clean = prep.fit_transform(df)

        engine = KMeansEngine(reports_dir="reports")
        engine.fit(X_clean)
        labels = engine.predict(X_clean)
        visualizer.plot_kmeans_silhouette_detail(X_clean, labels)
        engine.save()
        df.assign(Cluster=labels).to_csv("reports/kmeans_labels.csv", index=False)
        return

    # Ruta Supervisada (Clasificación o Regresión)
    print(f"\n>>> Ruta seleccionada: APRENDIZAJE SUPERVISADO ({task_type.upper()})")
    # Las filas sin etiqueta no permiten entrenamiento supervisado.
    df = df.dropna(subset=[detected_target])
    X = df.drop(columns=[detected_target])
    y = df[detected_target]

    stratify = y if (task_type == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    models_to_run = {
        "cart": CARTModel(task=task_type, reports_dir="reports"),
        "knn": KNNModel(task=task_type),
        "random_forest": RandomForestModel(task=task_type, reports_dir="reports"),
        "logistic" if task_type == "classification" else "ridge": LinearModel(task=task_type),
        "mars": MARSModel(task=task_type),
        "gam": GAMModel(task=task_type),
    }

    if task_type == "classification":
        models_to_run["lvq1"] = LVQ1Model()
        models_to_run["dann"] = DANNModel()
    else:
        models_to_run["lasso"] = LassoModel()

    evaluator = ModelEvaluator(task=task_type, reports_dir="reports")
    metric_key = "f1_score" if task_type == "classification" else "r2"
    results_summary = {}
    best_score = -float("inf")
    best_info = None

    for name, model in models_to_run.items():
        print(f"\n--- Entrenando modelo: {name.upper()} ---")
        prep = AdaptivePreprocessor(model_name=name)
        X_train_clean = prep.fit_transform(X_train)
        X_test_clean = prep.transform(X_test)

        model.fit(X_train_clean, y_train)
        y_pred = model.predict(X_test_clean)

        metrics = evaluator.evaluate(y_test, y_pred, model_name=name)
        results_summary[name.upper()] = metrics
        if name in {"logistic", "ridge"}:
            model.save(f"reports/{name}_model.joblib")
        else:
            model.save()
        if name in {"cart", "random_forest"}:
            model.plot_feature_importance(prep.feature_names_out)

        if metrics[metric_key] > best_score:
            best_score = metrics[metric_key]
            best_info = (name, model, X_train_clean, X_test_clean, y_pred)

    print("\n=== Resumen Comparativo de Modelos ===")
    summary_df = pd.DataFrame(results_summary).T
    print(summary_df.round(4))
    summary_df.to_csv("reports/supervised_metrics.csv")
    visualizer.plot_model_comparison(summary_df[metric_key].to_dict(), metric_name=metric_key)

    # Diagnósticos avanzados solo para el mejor modelo
    if best_info is not None:
        best_name, best_model, best_X_train, best_X_test, best_y_pred = best_info
        if task_type == "classification":
            estimator = getattr(best_model, "model", best_model)
            visualizer.plot_classification_diagnostics(
                estimator, best_X_train, y_train, best_X_test, y_test, model_name=best_name
            )
        else:
            visualizer.plot_regression_diagnostics(y_test, best_y_pred, model_name=best_name)


def create_demo_csv(path="data/demo_iris.csv"):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    iris.frame.to_csv(out, index=False)
    return str(out)


def prompt_interactive():
    print("\n" + "=" * 52)
    print("   FRAMEWORK MODULAR DE MACHINE LEARNING")
    print("=" * 52)

    data_dir = Path("data")
    available = sorted(
        [p for p in data_dir.glob("*") if p.suffix.lower() in {".csv", ".tsv", ".parquet"}]
    ) if data_dir.exists() else []

    if available:
        print("\nDatasets detectados en 'data/':")
        for idx, file_p in enumerate(available, start=1):
            print(f"  [{idx}] {file_p.name}")

    while True:
        choice = input("\nElige un número o pega la ruta de tu archivo (.csv/.tsv/.parquet): ").strip().strip('"')
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            dataset_path = str(available[int(choice) - 1])
            break
        if choice and Path(choice).is_file():
            dataset_path = choice
            break
        print("  [!] Archivo no encontrado. Intenta de nuevo.")

    target_input = input("Columna objetivo (Enter: automática, -: no supervisado): ").strip()
    target_col = target_input if target_input and target_input != "-" else None
    return dataset_path, target_col, target_input == "-"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Modular de Machine Learning")
    parser.add_argument("--file", type=str, default=None, help="Ruta al dataset (CSV, TSV o Parquet)")
    parser.add_argument("--target", type=str, default=None, help="Nombre opcional de la columna objetivo")
    parser.add_argument("--unsupervised", action="store_true", help="Forzar aprendizaje no supervisado")
    args = parser.parse_args()
    if args.unsupervised and args.target:
        parser.error("--unsupervised y --target no se pueden combinar")

    if args.file:
        run_pipeline(args.file, target_col=args.target, force_unsupervised=args.unsupervised)
    else:
        dataset_path, target_col, force_unsupervised = prompt_interactive()
        run_pipeline(dataset_path, target_col=target_col, force_unsupervised=force_unsupervised or args.unsupervised)