"""
Agente Autónomo de Machine Learning
-----------------------------------
Este script ejecuta un flujo de Machine Learning completo, automatizado de principio a fin:
1. Carga y preprocesa los datos
2. Entrena y evalúa varios modelos
3. Calcula métricas y selecciona el mejor modelo
4. Explica los resultados mediante SHAP
5. Genera un informe técnico automatizado con IA generativa
6. Exporta un PDF con el análisis y las visualizaciones

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Repositorio: [Tu Repositorio GitHub]
"""

import os
import re
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 🔧 Módulos de machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

# 📊 Librerías para generación del PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_JUSTIFY

# 🤖 Librerías para generación de texto con IA (OpenAI via LangChain)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# 0️⃣ CONFIGURACIÓN INICIAL Y DIRECTORIOS

load_dotenv()
open_api_key = os.getenv('OPENAI_API_KEY')
if not open_api_key:
    raise ValueError('No se encontró la variable OPEN_API_KEY en .env')

# Se crean carpetas necesarias para guardar resultados
os.makedirs('models', exist_ok=True)
os.makedirs('graphics', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# 1️⃣ PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN

def run_ml_pipeline(data_path, target_column, problem_type='regression'):
    """
    Entrena y evalúa modelos de Machine Learning.

    Parámetros:
        data_path (str): ruta del archivo CSV con los datos
        target_column (str): nombre de la columna objetivo
        problem_type (str): tipo de problema ("regression" o "classification")

    Retorna:
        best_model_name (str): nombre del mejor modelo
        best_metrics (dict): métricas del mejor modelo
        best_model (obj): modelo entrenado
        X (DataFrame): variables predictoras originales
    """
    # Cargamos los datos y separamos variables predictoras y objetivo
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Dividimos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # Escalado de variables numéricas
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definimos modelo según tipo de problema
    if problem_type == 'regression':
        models = {'RandomForestRegressor': RandomForestRegressor(random_state=21),
                  'XGBRegressor': XGBRegressor(random_state=21)}
    else:
        models = {'RandomForestClassifier': RandomForestClassifier(random_state=21),
                  'XGBClasifier': XGBClassifier(random_state=21)}

    # Entrenamiento y evaluación de modelos
    results = {}
    for name, model in models.items():
        print(f'Entrenando modelo: {name}')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Evaluación según el tipo de problema
        if problem_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE':mse, 'R2': r2}
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            results[name] = {'Accuracy':acc, 'f1':f1}
        
        # Guardamos modelos entrenado s   
        joblib.dump(model,f'models/{name}.pkl')

    # Seleccionamos el mejor modelo según metrica principal
    best_model_name = (min(results, key=lambda x: results[x]['MSE']) if problem_type == 'regression'
                       else max(results, key=lambda x: results[x]['Accuracy']))
    
    print(f'Mejor modelo: {best_model_name} → {results[best_model_name]}')
    return best_model_name, results[best_model_name], joblib.load(f'models/{best_model_name}.pkl'), X


# 2️⃣ INTERPRETACIÓN SHAP (EXPLICABILIDAD)
def compute_feature_importance(model, X):
    """
    Calcula la importancia de las variables mediante SHAP
    y guarda el gráfico correspondiente.

    Retorna:
        shap_path (str): ruta de la imagen guardada
    """
    print("📈 Calculando importancia de características con SHAP...")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)
    plt.title('Importancia de características (SHAP)')
    plt.tight_layout()
    shap_path = 'graphics/shap_feature_importance.png'
    plt.savefig(shap_path)
    plt.close()
    print('Gráfico de SHAP guardado en graphics/')
    return shap_path


# 3️⃣ Generación de texto con IA GENERATIVA
def generate_text_report(model_name, metrics, problem_type):
    """
    Genera un texto explicativo y técnico del modelo utilizando IA generativa (GPT-4o-mini).

    Explicación del proceso:
    ------------------------
    1️⃣ Se define un prompt en lenguaje natural que describe qué información queremos obtener.
        - El prompt incluye el tipo de problema, el modelo y las métricas.
        - Pedimos explícitamente un informe técnico estructurado con tres apartados.
    2️⃣ Se construye el objeto `PromptTemplate` de LangChain para rellenar el prompt dinámicamente.
    3️⃣ Se inicializa el modelo `ChatOpenAI` con la clave de API y temperatura moderada.
        - `temperature=0.5` controla la creatividad del texto.
    4️⃣ Se formatea el prompt con los valores actuales del modelo.
    5️⃣ Se invoca el modelo (IA generativa) y se obtiene el texto de salida.
    6️⃣ Se retorna el contenido generado para integrarlo en el PDF final.
    """

    # 🔹 Plantilla del prompt que se enviará a la IA
    template = """
    Eres un experto en ciencia de datos. Resume los resultados del modelo.

    - Tipo de problema: {problem_type}
    - Modelo utilizado: {model_name}
    - Métricas del modelo: {metrics}

    Escribe un informe técnico con:
    1. Interpretación de las métricas.
    2. Posibles mejoras al pipeline.
    3. Valor que aporta el modelo al negocio o investigación.
    """
    # 🔹 Definición del prompt dinámico con variables
    prompt = PromptTemplate(input_variables=['problem_type', 'model_name', 'metrics'], 
                            template=template)
    
    # 🔹 Inicializamos el modelo de lenguaje (OpenAI GPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    # 🔹 Formateamos el prompt con los valores específicos del experimento
    formatted_prompt = prompt.format(problem_type=problem_type, 
                                     model_name=model_name, 
                                     metrics=metrics)
    
    # 🔹 Ejecutamos la llamada al modelo de IA
    response = llm.invoke(formatted_prompt)

     # 🔹 Retornamos el texto generado por la IA
    return response.content


# 4️⃣ Generación del PDF
def generate_pdf_report(model_name, metrics, text, shap_img_path, problem_type):
    """
    Genera un informe PDF con formato profesional y contenido generado por IA.

    Explicación del proceso:
    ------------------------
    1️⃣ Se inicializa el documento PDF con `SimpleDocTemplate` y se definen estilos.
    2️⃣ Se agregan secciones estructuradas (título, métricas, análisis IA, gráfico SHAP).
    3️⃣ Se limpia el texto Markdown generado por la IA (**negritas**, ## encabezados).
    4️⃣ Se formatean los párrafos con estilo justificado.
    5️⃣ Se agrega un salto de página antes del gráfico SHAP.
    6️⃣ Finalmente, se construye y guarda el PDF.
    """
    pdf_path = 'reports/ml_report.pdf'

    # Configuración del documento PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Añadimos estilos personalizados
    styles.add(ParagraphStyle(name="BodyTextJustify", parent=styles["BodyText"], alignment=TA_JUSTIFY))  # genera párrafos con texto justificado
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6))  # títulos de secciones (“Análisis generado por IA”, “Importancia de características (SHAP)”) tengan un espaciado claro y limpio

    elements = []

    # 🔹 Título del informe
    elements.append(Paragraph(f"<b>Informe del Modelo:</b> {model_name}", styles["Title"]))
    elements.append(Spacer(1, 12))

    # 🔹 Información general
    elements.append(Paragraph(f"<b>Tipo de problema:</b> {problem_type}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Métricas:</b> {metrics}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # 🔹 Sección del análisis IA
    elements.append(Paragraph("Análisis generado por IA", styles["SectionTitle"]))

    # 🔹 Limpieza del texto generado (Markdown → texto plano)
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    clean_text = re.sub(r"##+", "", clean_text)
    paragraphs = [p.strip() for p in clean_text.split("\n") if p.strip()]

    # 🔹 Se agregan los párrafos del texto al PDF
    for p in paragraphs:
        elements.append(Paragraph(p, styles["BodyTextJustify"]))
        elements.append(Spacer(1, 8))
    
    # 🔹 Salto de página antes del gráfico SHAP
    elements.append(PageBreak())

    # 🔹 Sección del gráfico SHAP
    elements.append(Paragraph("Importancia de características (SHAP)", styles["SectionTitle"]))
    elements.append(Spacer(1, 12))
    elements.append(Image(shap_img_path, width=400, height=300))

    # 🔹 Construcción final del documento
    doc.build(elements)
    print(f"✅ Informe PDF guardado en {pdf_path}")


# 5️⃣ FLUJO PRINCIPAL DE EJECUCIÓN
def main():
    """
    Orquesta todo el flujo del agente autónomo:
    - Carga los datos
    - Ejecuta el pipeline de ML
    - Calcula SHAP
    - Genera informe con IA
    - Exporta PDF
    """
    data_path = 'data/raw/winequality-red.csv'
    target_column = 'quality'
    problem_type = 'regression'
    model_name, metrics, model, X = run_ml_pipeline(data_path, target_column, problem_type='regression')
    shap_path = compute_feature_importance(model, X)
    text = generate_text_report(model_name, metrics, problem_type)
    generate_pdf_report(model_name, metrics, text, shap_path, problem_type)
    
    print("🚀 Proceso completo finalizado con éxito.")


# 🚀 EJECUCIÓN DEL SCRIPT
if __name__ == '__main__':
    main()

