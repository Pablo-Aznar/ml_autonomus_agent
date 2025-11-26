import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 4) Preprocesamiento automatico / FEATURE ENGINEERING
def build_preprocessing_pipeline(df: pd.DataFrame, target_column: str):
    """
    Detecta columnas numéricas, categóricas, de texto y datetime.
    Construye un ColumnTransformer que:
    - Imputa y escala numéricas
    - Imputa y codifica categóricas (OneHot si cardinalidad baja, Ordinal si alta)
    - Para texto se deja pasar (opcion Tfidf)
    Retorna: preprocessing (ColumnTransformer), lista_feature_names 
    """

    X = df.drop(columns=[target_column])
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    low_card_cat = [c for c in cat_cols if df[c].nunique() <= 10]
    high_card_cat = [c for c in cat_cols if df[c].nunique() > 10]

    print(f'Variables numéricas: {num_cols}')
    print(f'Variables categóricas: {cat_cols}')

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    low_card_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    high_card_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    transformers = []
    if num_cols: transformers.append(("num", numeric_transformer, num_cols))
    if low_card_cat: transformers.append(("lowcat", low_card_transformer, low_card_cat))
    if high_card_cat: transformers.append(("highcat", high_card_transformer, high_card_cat))

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessing, num_cols, low_card_cat, high_card_cat