import re
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

def generate_pdf_report(text, shap_img_path, model_name, metrics, problem_type, output_pdf="reports/ml_report.pdf"):
    """
    Genera un PDF profesional con:
    - Título y metadata
    - Texto generado por IA (con saltos de línea correctos)
    - Gráfico SHAP en una página aparte
    - Espaciado mejorado usando Spacer()
    """
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)

    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="BodyTextJustify",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,
        leading=16))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        spaceBefore=16,
        spaceAfter=12))

    elements = []
    elements.append(Paragraph(f"Informe – Modelo: {model_name}", styles["Title"]))
    elements.append(Spacer(1, 20))

    clean = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    clean = re.sub(r"_([^_]+)_", r"\1", clean)

    lines = clean.split("\n")

    for line in lines:
        if line.strip() == "":
            elements.append(Spacer(1, 18))
        else:
            elements.append(Paragraph(line.strip(), styles["BodyTextJustify"]))
            elements.append(Spacer(1, 12))

    elements.append(PageBreak())
    elements.append(Paragraph("Importancia de características (SHAP)", styles["SectionTitle"]))

    if os.path.exists(shap_img_path):
        elements.append(Image(shap_img_path, width=6*inch, height=4*inch))

    doc.build(elements)
    print("PDF generado:", output_pdf)
