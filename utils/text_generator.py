from openai import OpenAI
import os

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
if not OpenAI.api_key:
    part1 = os.getenv("OPENAI_API_KEY_PART1", "")
    part2 = os.getenv("OPENAI_API_KEY_PART2", "")
    if not part1 or not part2:
        raise ValueError("Falta OPENAI_API_KEY o las dos partes (PART1 + PART2)")
    OpenAI.api_key = part1 + part2


def generate_text_report_openai(model_name, metrics, shap_text, eda_path, problem_type, models):
    client = OpenAI()

    prompt = f"""
    Eres un experto en ciencia de datos. Redacta un informe técnico claro y profesional con el siguiente contenido:

    - Tipo de problema: {problem_type}
    - Modelos utilizados: {models}
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un experto en ML."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
