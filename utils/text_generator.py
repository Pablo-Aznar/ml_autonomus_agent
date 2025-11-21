from openai import OpenAI
import os

# Cargar variables de entorno
from dotenv import load_dotenv
# Cargar .env solo en local
load_dotenv()

# === INTENTA USAR LA CLAVE NORMAL ===
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# === SI ESTÁ VACÍA O CORTADA (Railway bug), LA RECONSTRUIMOS CON LAS DOS PARTES ===
if not OpenAI.api_key or len(OpenAI.api_key) < 80 or "****" in OpenAI.api_key:
    part1 = os.getenv("OPENAI_API_KEY_PART1", "")
    part2 = os.getenv("OPENAI_API_KEY_PART2", "")
    if part1 and part2:
        full_key = part1 + part2
        print(f"[OpenAI] Clave truncada detectada → reconstruida ({len(full_key)} caracteres)")
        OpenAI.api_key = full_key
    else:
        raise ValueError(
            "OPENAI_API_KEY no encontrada o truncada, y no hay PART1 + PART2 para reconstruirla.\n"
            "→ Local: ponla en .env\n"
            "→ Railway: crea OPENAI_API_KEY_PART1 y OPENAI_API_KEY_PART2"
        )

# Verificación final (opcional, para debug)
if not OpenAI.api_key or not OpenAI.api_key.startswith("sk-"):
    raise ValueError("OPENAI_API_KEY inválida o vacía después de intentar arreglarla")
    

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
