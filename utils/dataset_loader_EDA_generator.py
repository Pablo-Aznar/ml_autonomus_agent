import pandas as pd
from ydata_profiling import ProfileReport  # EDA Automatico
import chardet

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
def load_user_dataset(path: str):
    """
    Carga un CSV del usuario limpiando los nombres de columnas:
    - Detecta encoding automáticamente
    - Intenta detectar separador automáticamente (para evitar 1 sola columna)
    - Elimina BOM/UTF-8
    - Convierte columnas a minúsculas
    - Quita espacios y caracteres raros
    """

    # Detectar encoding automáticamente
    with open(path, "rb") as f:
        raw = f.read()
        enc = chardet.detect(raw)["encoding"]

    # --- Intento inteligente de carga ---
    try:
        # Detectar automáticamente separador
        df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
    except Exception:
        # Si falla, intentar con coma
        try:
            df = pd.read_csv(path, encoding=enc, sep=",")
        except Exception:
            # Último fallback: separador ;
            df = pd.read_csv(path, encoding=enc, sep=";")

    # Normalizar columnas
    df.columns = (
        df.columns
        .str.encode('utf-8', 'ignore').str.decode('utf-8')
        .str.replace(r"[^\w\s]", "", regex=True)  # quitar símbolos extraños
        .str.strip()
        .str.lower()
        .str.replace(" +", "_", regex=True)
    )

    print("\n🔍 Columnas detectadas tras normalización:")
    print(df.columns.tolist())

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