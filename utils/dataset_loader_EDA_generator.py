import pandas as pd
from ydata_profiling import ProfileReport  # EDA Automatico

# 1) Detecta el tipo de problema
def detect_problem_type(y: pd.Series) -> str:
    """
    Detecta automáticamente si el problema es 'regression' o 'classification'.
    Regresión si la variable objetivo es numérica y tiene más de 20 valores únicos.
    Clasificación si la variable es objet o tiene pocos valores únicos (<=20).
    """
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 20:
            return 'regression'
        else:
            return 'classification'
    else:
        return 'classification'


# 2) Carga del DATASET
def load_user_dataset(path: str) -> pd.DataFrame:  #Type hints
    df = pd.read_csv(path)
    print(f'Dataset cargado: {path} con {df.shape[0]} filas y {df.shape[1]} columnas')
    return df


# 3) EDA Automático (ydata-profiling)
def generate_eda_report(df: pd.DataFrame, output_html: str = 'reports/EDA_report.html'):
    """Genera EDA interactivo con ydata-profiling y guarda el HTML"""
    print('Generando EDA automático con ydata-profiling...')
    profile = ProfileReport(df, title='EDA Automático - ML_autonomus_agent', explorative=True)
    profile.to_file(output_html)
    print(f'EDA guardado en {output_html}')
    return output_html

# Abrir html en servidor local para visualizar todos los desplegable
# python -m http.server 8000
# http://localhost:8000/reports/EDA_report.html