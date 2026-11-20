import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class AdaptivePreprocessor:
    """
    Pipeline de preprocesamiento adaptativo.
    Ajusta estadísticas solo en entrenamiento para evitar Data Leakage
    y aplica escalado condicionalmente según el tipo de modelo.
    """

    SCALE_SENSITIVE_MODELS = {
        "knn", "lvq1", "dann", "kmeans", "ridge", "lasso", "logistic", "mars", "gam"
    }

    def __init__(self, model_name="kmeans"):
        self.model_name = model_name.lower()
        self.requires_scaling = self.model_name in self.SCALE_SENSITIVE_MODELS
        self.pipeline = None
        self.zero_var_cols = []
        self.feature_names_out = []

    def _build_pipeline(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.requires_scaling:
            num_steps.append(("scaler", StandardScaler()))

        num_pipeline = Pipeline(steps=num_steps)

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_pipeline, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipeline, cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Ajusta el preprocesador estrictamente sobre X_train y lo transforma.
        """
        X_clean = X_train.copy()

        # Eliminar columnas constantes (varianza cero) detectadas en train
        num_cols = X_clean.select_dtypes(include=[np.number]).columns
        self.zero_var_cols = [col for col in num_cols if X_clean[col].nunique() <= 1]
        if self.zero_var_cols:
            X_clean = X_clean.drop(columns=self.zero_var_cols)

        self.pipeline = self._build_pipeline(X_clean)
        X_transformed = self.pipeline.fit_transform(X_clean)
        self.feature_names_out = self.pipeline.get_feature_names_out().tolist()

        estado_escala = "ACTIVO (StandardScaler)" if self.requires_scaling else "OMITIDO (Modelo inmune)"
        print(f"[Preprocessor] Modelo: '{self.model_name}' | Escalado: {estado_escala}")
        return X_transformed

    def transform(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Aplica las estadísticas ya aprendidas en train sobre X_test (sin Data Leakage).
        """
        if self.pipeline is None:
            raise RuntimeError("Debes ejecutar fit_transform(X_train) antes de transform(X_test).")

        X_clean = X_test.drop(columns=self.zero_var_cols, errors="ignore")
        return self.pipeline.transform(X_clean)