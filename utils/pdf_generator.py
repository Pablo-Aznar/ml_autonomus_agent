import re
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

def generate_pdf_report(text: str, shap_img_path: str, model_name: str, metrics: dict, problem_type: str, output_pdf: str='reports/ml_report.pdf'):
    """
    Genera un PDF profesional con:
    - Título y metadata
    - Texto generado por IA (con saltos de línea correctos)
    - Gráfico SHAP en una página aparte
    - Espaciado mejorado usando Spacer()
    """

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)

    # === CONFIGURACIÓN DE DOCUMENTO ===
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="BodyTextJustify",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,
        leading=16))   # Espaciado entre líneas

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        spaceBefore=14,
        spaceAfter=10))
    
    elements = []

    # === 1. TITULO Y METADATA ===
    title = f'Informe automático - Modelo: {model_name}'
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f'<b>Tipo de problema:</b> {problem_type}', styles['Normal']))
    elements.append(Paragraph(f"<b>Métricas:</b> {metrics}", styles["Normal"]))
    elements.append(Spacer(1, 18))


    # === 2. LIMPIEZA DEL TEXTO GENERADO POR IA ===
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # quitar **negrita**
    clean_text = re.sub(r"#+", "", clean_text)           # quitar markdown headers
    clean_text = re.sub(r"_([^_]+)_", r"\1", clean_text) # quitar cursivas

    # Dividimos usando los saltos de línea del modelo
    lines = clean_text.split("\n")

    # === 3. AGREGAR TEXTO AL PDF CON SALTOS REALES ===
    for line in lines:
        if line.strip() == "":
            elements.append(Spacer(1, 14))  # salto extra para separar apartados
        else:
            elements.append(Paragraph(line.strip(), styles["BodyTextJustify"]))
            elements.append(Spacer(1, 10))  # espacio después de cada párrafo


    # === 4. SALTO DE PÁGINA PARA EL SHAP ===
    elements.append(PageBreak())
    elements.append(Paragraph("Importancia de características (SHAP)", styles["SectionTitle"]))
    elements.append(Spacer(1, 14))

    if os.path.exists(shap_img_path):
        elements.append(Image(shap_img_path, width=6 * inch, height=4 * inch))
    else:
        elements.append(Paragraph("⚠️ No se encontró el gráfico SHAP.", styles["Normal"]))


    # === 5. GENERAR PDF ===
    doc.build(elements)
    print(f"PDF generado correctamente en: {output_pdf}")
