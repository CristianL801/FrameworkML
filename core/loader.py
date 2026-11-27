
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
            raise ValueError(
                f"Formato no soportado: {extension}. "
                "Use CSV, TSV o Parquet."
            )

        try:
            if extension == ".csv":
                try:
                    self.dataframe = pd.read_csv(
                        self.filepath,
                        encoding="utf-8"
                    )
                except UnicodeDecodeError:
                    self.dataframe = pd.read_csv(
                        self.filepath,
                        encoding="latin-1"
                    )

            elif extension == ".tsv":
                try:
                    self.dataframe = pd.read_csv(
                        self.filepath,
                        sep="\t",
                        encoding="utf-8"
                    )
                except UnicodeDecodeError:
                    self.dataframe = pd.read_csv(
                        self.filepath,
                        sep="\t",
                        encoding="latin-1"
                    )

            elif extension == ".parquet":
                self.dataframe = pd.read_parquet(self.filepath)

            print(
                f"Datos cargados exitosamente: "
                f"{self.dataframe.shape[0]} filas, "
                f"{self.dataframe.shape[1]} columnas."
            )

            return self.dataframe

        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None

    def detect_target(self, df, target_col=None):
        """
        Analiza la columna objetivo indicada por el usuario o detecta
        automáticamente si la tarea es clasificación, regresión
        o no supervisada.
        """

        if df.empty or not len(df.columns):
            raise ValueError("El dataset no contiene datos suficientes.")

        def task_for(column):
            values = df[column].dropna()
            if values.empty:
                raise ValueError(f"La columna objetivo '{column}' no contiene valores.")
            if (not pd.api.types.is_numeric_dtype(values)
                    or pd.api.types.is_bool_dtype(values)
                    or (values.nunique() <= 20 and values.nunique() / len(values) < 0.05
                        and (values % 1 == 0).all())):
                return "classification"
            return "regression"

        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")
            task = task_for(target_col)
            print(f"[Loader] Target especificado: '{target_col}' -> Modo: {task.upper()}")
            return target_col, task

        # Nombres convencionales de target, sin depender del nombre del archivo.
        hints = {"target", "label", "class", "y", "price", "precio", "sales", "score",
                 "experience_level", "objetivo", "etiqueta", "clase"}
        candidates = [col for col in df.columns if str(col).strip().lower() in hints]
        last_col = df.columns[-1]
        if len(candidates) == 1:
            target_col = candidates[0]
        elif len(candidates) == 0 and df[last_col].notna().any() and task_for(last_col) == "classification":
            target_col = last_col
        else:
            warnings.warn("No se detectó una variable objetivo inequívoca. Enrutando a K-Means.")
            return None, "unsupervised"

        task = task_for(target_col)
        print(f"[Loader] Target detectado automáticamente: '{target_col}' -> Modo: {task.upper()}")
        return target_col, task
