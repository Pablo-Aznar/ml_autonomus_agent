"""
ml_autonomus_agent.py
Agente Autónomo de Machine Learning — AutoML Full

Características:
- Carga dinámica de datasets (CSV) provistos por el usuario
- EDA automático usando ydata-profiling
- Detección automática del tipo de problema (regresión vs clasificación)
- Procesamiento automático de features (detección num/categ/text/datetime)
- Imputación, encoding (OneHot / Ordinal según cardinalidad), escalado
- Entrenamiento automático de modelos (RandomForest + XGBoost)
- Selección automática del mejor modelo
- Explicabilidad con SHAP (gráfico)
- Generación de informe en texto con OpenAI y exportación a PDF con ReportLab

"""

import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import openai
from datetime import datetime

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# XGBoost
from xgboost import XGBRegressor, XGBClassifier

# EDA
from ydata_profiling import ProfileReport

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError('No se encontró OPENAI_API_KEY en el archivo .env')

# Creamos carpetas necesarias
os.makedirs('models', exist_ok=True)
os.makedirs('graphics', exist_ok=True)
os.makedirs('reports', exist_ok=True)


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

    # Detección de columnas
    X = df.drop(columns = [target_column])  
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = X.select_dtypes(include=['datetime','datetime64, datetime64[ns]']).columns.tolist()
    print(f'Columnas numéricas: {num_cols}')
    print(f'Columnas categóricas: {cat_cols}')
    if datetime_cols:
        print(f'Columnas datetime: {datetime_cols}')
    
    # Para categórica separamos por cardinalidad
    low_card_cat = [c for c in cat_cols if df[c].nunique() <= 10]
    high_card_cat = [c for c in cat_cols if df[c].nunique() > 10]

    # Numéricos: imputación con mediana + escalado
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', MinMaxScaler())])
    
    # Categóricas de baja cardinalidad: imputacion most_frequent + OneHot
    low_card_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    
    # Categóricas de alta cardinalidad: imputacion most_frequent + Ordinal
    high_card_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    
    # ColumnTrasnformer
    transformers = []
    if num_cols:
        transformers.append(('num', numeric_transformer, num_cols))  
    if low_card_cat:
        transformers.append(('lowcat', low_card_transformer, low_card_cat))
    if high_card_cat:
        transformers.append(('highcat', high_card_transformer, high_card_cat))
    
    preprocessing = ColumnTransformer(transformers=transformers, remainder='drop')  # drop:de momento no transformamos datetime y text (mejorar con feature extraction)

    return preprocessing, num_cols, low_card_cat, high_card_cat


# 5) PIPELINE DE ENTRENIENTO AUTOMATICO (AutoML Full)
def run_ml_pipeline_auto(df: pd.DataFrame, target_column: str, problem_type: str=None):
    """
    Pipeline completamente automático:
    - Detecta problema si no se proporciona
    - Construye preprocessing dinámico
    - Entrena modelos (RandomForest + XGBoost)
    - Evalúa y selecciona el mejor
    - Guarda modelo y pipeline
    Retorna: best_model_name, metrics, fitted_best_model, preprocessing, feature_names_df
    """
    if problem_type is None:
        problem_type = detect_problem_type(df[target_column])
    print(f'Tipo de problema detectado: {problem_type}')

    # Preprocessing pipeline
    preprocessing, num_cols, low_card_cat, high_card_cat = build_preprocessing_pipeline(df, target_column)

    # Separamos X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split
    X_train, X_test, y_train, y_test= train_test_split(X,
                                                       y, 
                                                       test_size=0.2,
                                                       random_state=21,
                                                       stratify=None if problem_type=='regression' else y)
    # Fit preprocessing en Train
    print('Adjutando preprocessing en X_train...')
    preprocessing.fit(X_train)

    # Transformamos para obtener arrays y nombre de features
    X_train_proc = preprocessing.transform(X_train)
    X_test_proc = preprocessing.transform(X_test)
    # ObTenemos nombre las columnas
    try:
        features_names = preprocessing.get_feature_names_out()
    except Exception:
        features_names = [f'f{i}' for i in range(X_train_proc.shape[1])]
    
    # Convertimos a Dataframe para facilitar SHAP
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=features_names)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=features_names)

        # Definir modelos
    if problem_type == "regression":
        models = {"RandomForestRegressor": RandomForestRegressor(random_state=21),
                  "XGBRegressor": XGBRegressor(random_state=21, verbosity=0)}
    else:
        models = {"RandomForestClassifier": RandomForestClassifier(random_state=21),
                  "XGBClassifier": XGBClassifier(random_state=21, use_label_encoder=False, eval_metric='logloss', verbosity=0)}

    results = {}
    fitted_models = {}

    # Entrenamos y evaluamos
    for name, model in model.items():
        print(f'Entrenamos {name}...')
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        if problem_type == 'regresion':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R2': r2}
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {'accuracy': acc, 'f1': f1}
        
        fitted_models[name] = model
        # Guardar modelo (modelo sin pipeline)
        joblib.dump(model,f'models/{name}.pkl')

    # Seleccionamos el mejor
    if problem_type == 'regression':
        best_model_name = min(results, key=lambda x: results[x]['MSE'])
    else:
        best_model_name = max(results, key=lambda x: results[name]['Accuracy'])
    
    best_metrics = results[best_model_name]
    best_model = fitted_models[best_model_name]

    print(f'El mejor model es: {best_model_name} --> {best_metrics}')

    # Guardar preprocessing por separado
    joblib.dump(preprocessing, f'models/preprocessing_pipeline.pkl')

    return best_model_name, best_metrics, best_model, preprocessing, X_train_proc_df, X_test_proc_df, y_test





