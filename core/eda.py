import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


class EDAnalyzer:
    """
    Módulo automatizado para Análisis Exploratorio de Datos (EDA).
    Genera diagnósticos de calidad, matrices de covarianza/correlación
    y detecta multicolinealidad, asimetrías y outliers sin intervención manual.
    """

    def __init__(self, dataframe, reports_dir="reports"):
        """
        Inicializa el analizador con un DataFrame y asegura que el
        directorio de reportes exista usando rutas relativas.
        """
        self.df = dataframe
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Separar variables numéricas y categóricas para análisis específico
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    def diagnose_quality(self):
        """
        Diagnóstico de Calidad:
        Detección de nulos, cardinalidad, valores atípicos (IQR) y asimetrías (skewness).
        """
        print("\n--- Diagnóstico de Calidad de Datos ---")

        # 1. Valores faltantes
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print(f"Valores faltantes detectados:\n{missing}\n")
        else:
            print("No se detectaron valores faltantes.\n")

        # 2. Cardinalidad de categóricas
        if self.cat_cols:
            print("Cardinalidad de variables categóricas:")
            for col in self.cat_cols:
                print(f" - {col}: {self.df[col].nunique()} valores únicos")
            print()

        # 3. Detección de Outliers mediante IQR (Rango Intercuartílico)
        print("Detección de Outliers (IQR):")
        outliers_found = False
        for col in self.num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                outliers_found = True
                print(f" - {col}: {len(outliers)} valores atípicos detectados.")
        if not outliers_found:
            print(" - No se detectaron valores atípicos mediante IQR.")

        # 4. Detección de Asimetrías (Skewness)
        if self.num_cols:
            print("\nDetección de Asimetrías (|skew| > 1.0):")
            skew_series = self.df[self.num_cols].skew()
            high_skew = skew_series[skew_series.abs() > 1.0]
            if not high_skew.empty:
                for col, val in high_skew.items():
                    tipo = "positiva (cola derecha)" if val > 0 else "negativa (cola izquierda)"
                    print(f" - {col}: skew = {val:.2f} (asimetría {tipo})")
            else:
                print(" - No se detectaron asimetrías severas (|skew| > 1.0).")

    def analyze_correlations(self):
        """
        Calcula y expone matrices de covarianza y correlación (Pearson y Spearman),
        detecta varianza nula y multicolinealidad, y exporta mapas de calor en alta calidad.
        """
        if len(self.num_cols) < 2:
            warnings.warn("No hay suficientes variables numéricas para análisis de correlación.")
            return

        print("\n--- Análisis de Dispersión y Relaciones ---")

        df_num = self.df[self.num_cols]

        # 1. Varianza Nula
        zero_var = df_num.columns[df_num.var() == 0].tolist()
        if zero_var:
            print(f"Advertencia: Variables con varianza nula detectadas (constantes): {zero_var}")

        # 2. Matrices Matemáticas
        cov_matrix = df_num.cov()
        pearson_corr = df_num.corr(method="pearson")
        spearman_corr = df_num.corr(method="spearman")

        # Exponer y exportar Matriz de Covarianza
        cov_path = self.reports_dir / "covariance_matrix.csv"
        cov_matrix.round(4).to_csv(cov_path)
        print("\nMatriz de Covarianza (redondeada a 2 decimales):")
        print(cov_matrix.round(2))
        print(f"Matriz de covarianza exportada en: '{cov_path}'\n")

        # 3. Detección de Multicolinealidad (|r| > 0.85)
        high_corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i + 1, len(pearson_corr.columns)):
                if abs(pearson_corr.iloc[i, j]) > 0.85:
                    high_corr_pairs.append(
                        (pearson_corr.columns[i], pearson_corr.columns[j], pearson_corr.iloc[i, j])
                    )

        if high_corr_pairs:
            print("ALERTA: Alta multicolinealidad detectada (|r| > 0.85):")
            for pair in high_corr_pairs:
                print(f" - {pair[0]} y {pair[1]}: r = {pair[2]:.3f}")
        else:
            print("No se detectó multicolinealidad severa (|r| > 0.85).")

        # 4. Renderizado y guardado de Heatmaps (300 DPI)
        self._plot_heatmap(pearson_corr, "Correlación de Pearson", "pearson_heatmap.png")
        self._plot_heatmap(spearman_corr, "Correlación de Spearman", "spearman_heatmap.png")
        self._plot_heatmap(cov_matrix, "Matriz de Covarianza", "covariance_heatmap.png")
        print(f"\nMapas de calor guardados exitosamente en la carpeta: '{self.reports_dir}'")

    def _plot_heatmap(self, matrix, title, filename):
        """Método interno para renderizar mapas de calor."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
        plt.title(title, fontsize=14)
        plt.tight_layout()

        # Guardar en la carpeta reports usando rutas relativas
        filepath = self.reports_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

    def run_full_eda(self):
        """Ejecuta el pipeline completo de EDA."""
        self.diagnose_quality()
        self.analyze_correlations()
        