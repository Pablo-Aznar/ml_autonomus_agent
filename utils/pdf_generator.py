import re
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


def generate_pdf_report(
    text: str,
    shap_img_path: str,
    model_name: str,
    metrics: dict,
    problem_type: str,
    dataset_name: str,
    target: str,
    output_pdf: str = "reports/ml_report.pdf"):
    """
    PDF profesional optimizado (basado en tu versión original):
    - Títulos en negrita (como los genera GPT)
    - Cada sección completa en una sola página
    - Si un título quedaría huérfano → salta a página nueva
    """

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    doc = SimpleDocTemplate(output_pdf, pagesize=A4, topMargin=0.8*inch, bottomMargin=0.8*inch)

    # === TUS ESTILOS ORIGINALES (sin conflictos) ===
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="BodyTextJustify",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,
        leading=16
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        spaceBefore=20,
        spaceAfter=12,
        fontName="Helvetica-Bold"
    ))

    elements = []

# === PORTADA ===
    elements.append(Paragraph("Informe automático", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"<b>Dataset:</b> {dataset_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Variable objetivo:</b> {target}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Tipo de problema:</b> {problem_type.capitalize()}", styles["Normal"]))

    elements.append(Spacer(1, 40))

    # === LIMPIEZA DEL TEXTO (MANTENEMOS LA NEGRITA) ===
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)      # ** → <b>
    clean_text = re.sub(r"_([^_]+)_", r"\1", clean_text)          # cursiva → quitada
    clean_text = re.sub(r"#+\s*", "", clean_text)                 # # Título → solo texto
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]

    # === AGRUPAR POR SECCIONES (mantener todo junto) ===
    current_section = []

    def add_section():
        """Inserta una sección completa en el PDF."""
        if not current_section:
            return

        for line in current_section:
            if "<b>" in line and line.endswith("</b>:"):
                elements.append(Paragraph(line, styles["SectionTitle"]))
            else:
                elements.append(Paragraph(line, styles["BodyTextJustify"]))
            elements.append(Spacer(1, 12))

        elements.append(Spacer(1, 20))
        current_section.clear()

    for line in lines:
        # Detectar nuevos títulos
        if ":" in line and "<b>" in line:
            add_section()
        current_section.append(line)

    add_section()  # última sección

    # ========== GRÁFICO SHAP ==========
    if os.path.exists(shap_img_path):
        try:
            elements.append(PageBreak())
            elements.append(Paragraph("<b>Importancia de características (SHAP)</b>", styles["SectionTitle"]))
            elements.append(Spacer(1, 20))

            img = Image(shap_img_path, width=5.5 * inch, height=4.2 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Error cargando imagen SHAP: {e}", styles["Normal"]))
    else:
        elements.append(Paragraph("No se encontró el gráfico SHAP.", styles["Normal"]))

    # ========== CREAR PDF ==========
    try:
        doc.build(elements)
    except Exception as e:
        raise RuntimeError(f"Error construyendo PDF: {e}")

    return output_pdf