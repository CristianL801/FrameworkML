from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Clase base abstracta para todos los modelos de Machine Learning.
    Exige la implementación de métodos estándar (interfaz) para mantener 
    un bajo acoplamiento y alto polimorfismo.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Ajusta el modelo a los datos de entrenamiento.
        y es opcional para soportar modelos no supervisados.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Genera predicciones o asigna clústeres a nuevos datos."""
        pass

    @abstractmethod
    def score(self, X, y=None):
        """Calcula una métrica de rendimiento por defecto del modelo."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Guarda el estado del modelo en el disco (ej. usando pickle o joblib)."""
        pass