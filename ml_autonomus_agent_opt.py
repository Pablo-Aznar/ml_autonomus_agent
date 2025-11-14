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
                                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
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


# 5) PIPELINE DE ENTRENAMIENTO AUTOMATICO (AutoML Full)
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

    # Si es clasificación y y tiene valores numéricos no 0-indexados, los normalizamos
    if problem_type == 'classification':
        if y.dtype != 'O':  # solo si no es categórica tipo string
            y = y.astype(int) - y.min()

    # Split
    X_train, X_test, y_train, y_test= train_test_split(X,
                                                       y, 
                                                       test_size=0.2,
                                                       random_state=21,
                                                       stratify=None if problem_type=='regression' else y)
    # Fit preprocessing en Train
    print('Ajustando preprocessing en X_train...')
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
    for name, model in models.items():
        print(f'Entrenamos {name}...')
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        if problem_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R2': r2}
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {'Accuracy': acc, 'f1': f1}
        
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


# 6) EXPLICABILIDAD SHAP
def compute_shap(model, X_proc_df, output_path='graphics/shap_summary.png'):
    """
    Calcula valores SHAP (aplicable para tree-based) y guarda summary plot.
    Devuelve un breve resumen textual para el informe.
    """

    print('Calculando SHAP...')
    try:
        # TreeExplainer para modelos basados en arboles
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc_df)
    except Exception:
        # fallback shap.Explainer generico
        explainer = shap.Explainer(model)
        shap_values = explainer(X_proc_df)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_proc_df, show=False)  # False para compatibilidad entre versiones
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Gráfico SHAP guardado en: {output_path}')

    # generamos texto resumen sencillo
    shap_text = "Se calculó SHAP para identificar características con mayor influencia en las predicciones."
    return shap_text


# 7) GENERACIÓN DE TEXTO CON IA (OpenAI Chat)
def generate_text_report_openai(model_name, metrics, shap_text, eda_path, problem_type):
    """
    Genera un informe de texto usando la API de OpenAI
    con comentarios inline que explican cada paso
    """

    # 7.1) Creamos el prompt para dar estructura al informe
    prompt = f"""
    Eres un experto en ciencia de datos. Redacta un informe técnico claro y profesional con el siguiente contenido:

    - Tipo de problema: {problem_type}
    - Mejor modelo: {model_name}
    - Métricas: {metrics}
    - Resumen SHAP: {shap_text}
    - Ruta del EDA: {eda_path}

        
    Estructura el informe en secciones numeradas:
    1) Introducción breve
    2) Análisis exploratorio (puntos clave)
    3) Modelado (qué modelos se probaron y por qué)
    4) Métricas y evaluación
    5) Interpretabilidad y puntos de acción
    6) Conclusiones y recomendaciones prácticas

    Escribe es español, con frases claras y sin incluir código. Usa un tono técnico-profesional y conciso
    """

    # 7.2) Llamada a la API de OpenAI 
    # Usamos ChatCompletion para generar texto; el modelo puede variar según disponibilidad (gpt-4o-mini o similar)
    response = openai.ChatCompletion.create(model='gpt-4o-mini',  # client.chat.completions.create
                                            messages=[{'role':'user', 'content': prompt}],
                                            temperature=0.3,
                                            max_tokens=1000)
    
    text = response['choices'][0]['message']['content']  # response.choices[0].message.content
    print('Texto generado por IA (Open AI)')
    return text


# 8) GENERAR PDF (ReportLab)
def generate_pdf_report(text: str, shap_img_path: str, model_name: str, metrics: dict, problem_type: str, output_pdf: str='reports/ml_report.pdf'):
    """
    Genera un PDF profesional con:
    - Título y metadata
    - Texto generado por IA (limpio de Markdown)
    - Salto de página con gráfico SHAP
    Comentarios inline describen las transformaciones.
    """
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)

    # 8.1) Configuramos documento y estilos
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyTextJustify", parent=styles["BodyText"], alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6))
    elements = []

    # 8.2) Título y metadata
    title = f'Informe automático - Modelo: {model_name}'
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f'<b>Tipo de problema:<b> {problem_type}', styles['Normal']))
    elements.append(Paragraph(f"<b>Métricas:</b> {metrics}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # 8.3) Limpieza del texto generado por la IA (quitar Markdown)
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # eliminar negritas Markdown
    clean_text = re.sub(r"#+", "", clean_text)           # eliminar encabezados Markdown
    clean_text = re.sub(r"_([^_]+)_", r"\1", clean_text) # quitar cursivas
    # Separar por doble salto de línea para mantener párrafos
    paragraphs = re.split(r'\n\s*\n', clean_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 8.4) Añadir párrafos formateados
    for p in paragraphs:
        elements.append(Paragraph(p, styles["BodyTextJustify"]))
        elements.append(Spacer(1, 8))

    # 8.5) Salto de página y gráfico SHAP en página aparte
    elements.append(PageBreak())
    elements.append(Paragraph("Importancia de características (SHAP)", styles["SectionTitle"]))
    elements.append(Spacer(1, 12))
    if os.path.exists(shap_img_path):
        # Insertar imagen y ajustarla
        elements.append(Image(shap_img_path, width=6 * inch, height=4 * inch))
    else:
        elements.append(Paragraph("⚠️ No se encontró el gráfico SHAP.", styles["Normal"]))
    
    # 8.6) Guardar PDF
    doc.build(elements)
    print(f"PDF generado y guardado en: {output_pdf}")


# 9) EJECUCIÓN DEL SCRIPT (AutoML Full)
def main():
    print("\n=== ml_autonomus_agent: AutoML Full ===\n")

    # Entradas
    data_path = input('Ruta del dataset CSV (ej: data/raw/winequality-red.csv:').strip()
    target_column = input('Nombre de la variable objetivo (target):').strip()

    # 9.1) Carga del dataset
    df = load_user_dataset(data_path)

    # 9.2) EDA automático (ydata-profiling)
    eda_path = generate_eda_report(df, output_html='reports/EDA_report.html')

    # 9.3) Pipeline AutoML Full (detecta tipo problema, preprocessing automático, entrenar modelos)
    best_model_name, best_metrics, best_model, preprocessing, X_train_proc_df, X_test_proc_df, y_test = run_ml_pipeline_auto(
        df, target_column, problem_type=None)
    
    # 9.4) Explicabilidad SHAP
    shap_img = "graphics/shap_summary.png"
    shap_text = compute_shap(best_model, X_test_proc_df, shap_img)

    # 9.5) Generación de texto con IA (OpenAI)
    report_text = generate_text_report_openai(best_model_name, best_metrics, shap_text, eda_path, problem_type=detect_problem_type(df[target_column]))

    # 6) Generación del PDF final
    generate_pdf_report(report_text, shap_img, best_model_name, best_metrics, detect_problem_type(df[target_column]), output_pdf="reports/ml_report.pdf")

    print("\nProceso finalizado. Archivos generados:")
    print("- EDA HTML:", eda_path)
    print("- SHAP image:", shap_img)
    print("- PDF report: reports/ml_report.pdf")
    print("- Model saved in models/ (model .pkl and preprocessing_pipeline.pkl)")

if __name__ == '__main__':
    main()






