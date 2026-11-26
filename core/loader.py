import warnings
from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Clase encargada de la ingesta de datos tabulares (CSV, TSV, Parquet)
    y la detección automática de la variable objetivo.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.dataframe = None

    def load_data(self):
        """
        Carga los datos según la extensión del archivo.
        Retorna el DataFrame cargado.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"El archivo no existe: {self.filepath}")

        extension = self.filepath.suffix.lower()

        if extension not in (".csv", ".tsv", ".parquet"):
            raise ValueError(f"Formato no soportado: {extension}. Use CSV, TSV o Parquet.")

        try:
            if extension == ".csv":
                try:
                    self.dataframe = pd.read_csv(self.filepath, encoding="utf-8")
                except UnicodeDecodeError:
                    self.dataframe = pd.read_csv(self.filepath, encoding="latin-1")

            elif extension == ".tsv":
                try:
                    self.dataframe = pd.read_csv(self.filepath, sep="\t", encoding="utf-8")
                except UnicodeDecodeError:
                    self.dataframe = pd.read_csv(self.filepath, sep="\t", encoding="latin-1")

            elif extension == ".parquet":
                self.dataframe = pd.read_parquet(self.filepath)

            print(f"Datos cargados exitosamente: {self.dataframe.shape[0]} filas, {self.dataframe.shape[1]} columnas.")
            return self.dataframe

        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None

    def detect_target(self, df, target_col=None):
        """
        Analiza la columna objetivo (indicada por el usuario o la última por defecto)
        para inferir si la tarea es clasificación, regresión o no supervisada (K-Means).
        """
        # Si el usuario especificó el target, validamos que exista
        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")
            col_data = df[target_col]
            unique_ratio = col_data.nunique() / len(df)

            if col_data.dtype == "object" or col_data.dtype.name == "category" or unique_ratio < 0.05:
                print(f"[Loader] Target especificado: '{target_col}' -> Modo: CLASIFICACIÓN")
                return target_col, "classification"

            print(f"[Loader] Target especificado: '{target_col}' -> Modo: REGRESIÓN")
            return target_col, "regression"

        # Detección automática sobre la última columna
        last_col = df.columns[-1]
        col_data = df[last_col]
        unique_ratio = col_data.nunique() / len(df)

        # 1. Categórica o baja cardinalidad (< 5% únicos) -> Clasificación
        if col_data.dtype == "object" or col_data.dtype.name == "category" or unique_ratio < 0.05:
            print(f"[Loader] Target detectado: '{last_col}' -> Modo: CLASIFICACIÓN")
            return last_col, "classification"

        # 2. Si el nombre sugiere objetivo numérico -> Regresión
        col_lower = last_col.lower().strip()
        target_hints = ("target", "price", "precio", "label", "class", "sales", "score")
        if col_lower == "y" or any(hint in col_lower for hint in target_hints):
            print(f"[Loader] Target detectado: '{last_col}' -> Modo: REGRESIÓN")
            return last_col, "regression"

        # 3. Fallback: sin objetivo claro -> No Supervisado
        warnings.warn(
            f"No se detectó variable objetivo clara en '{last_col}'. "
            "Enrutando automáticamente al motor No Supervisado (K-Means)."
        )
        return None, "unsupervised"