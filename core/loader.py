import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Clase encargada de la ingesta de datos tabulares (CSV, TSV, Parquet).
    Maneja rutas relativas usando pathlib.
    """
    
    def __init__(self, filepath):
        """
        Inicializa el DataLoader con la ruta del archivo.
        """
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
            if extension == '.csv':
                # Tolera codificaciones comunes
                try:
                    self.dataframe = pd.read_csv(self.filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    self.dataframe = pd.read_csv(self.filepath, encoding='latin-1')
            
            elif extension == '.tsv':
                self.dataframe = pd.read_csv(self.filepath, sep='\t')
            
            elif extension == '.parquet':
                self.dataframe = pd.read_parquet(self.filepath)
            
            else:
                raise ValueError(f"Formato no soportado: {extension}. Use CSV, TSV o Parquet.")
            
            print(f"Datos cargados exitosamente: {self.dataframe.shape[0]} filas, {self.dataframe.shape[1]} columnas.")
            return self.dataframe
            
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None