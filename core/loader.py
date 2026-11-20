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

        try:
            if extension == ".csv":
                try:
                    self.dataframe = pd.read_csv(self.filepath, encoding="utf-8")
                except UnicodeDecodeError:
                    self.dataframe = pd.read_csv(self.filepath, encoding="latin-1")

            elif extension == ".tsv":
                self.dataframe = pd.read_csv(self.filepath, sep="\t")

            elif extension == ".parquet":
                self.dataframe = pd.read_parquet(self.filepath)

            else:
                raise ValueError(f"Formato no soportado: {extension}. Use CSV, TSV o Parquet.")

            print(f"Datos cargados exitosamente: {self.dataframe.shape[0]} filas, {self.dataframe.shape[1]} columnas.")
            return self.dataframe

        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None

    def detect_target(self, df):
        """
        Analiza la última columna para inferir automáticamente si la tarea es
        clasificación, regresión o no supervisada (K-Means).
        """
        last_col = df.columns[-1]
        col_data = df[last_col]
        unique_ratio = col_data.nunique() / len(df)

        # 1. Categórica o baja cardinalidad (< 5% únicos) -> Clasificación
        if col_data.dtype == "object" or col_data.dtype.name == "category" or unique_ratio < 0.05:
            print(f"[Loader] Target detectado: '{last_col}' -> Modo: CLASIFICACIÓN")
            return last_col, "classification"

        # 2. Si el nombre sugiere objetivo numérico -> Regresión
        target_hints = ("target", "price", "precio", "label", "class", "y", "sales", "score")
        if any(hint in last_col.lower() for hint in target_hints):
            print(f"[Loader] Target detectado: '{last_col}' -> Modo: REGRESIÓN")
            return last_col, "regression"

        # 3. Fallback: sin objetivo claro -> No Supervisado
        warnings.warn(
            f"No se detectó variable objetivo clara en '{last_col}'. "
            "Enrutando automáticamente al motor No Supervisado (K-Means)."
        )
        return None, "unsupervised"