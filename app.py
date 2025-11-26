# app.py
import uuid
import shutil
import time
import sys
import re
from io import StringIO
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Tus funciones
from utils.dataset_loader_EDA_generator import load_user_dataset, generate_eda_report, detect_problem_type
from utils.automl_pipeline import run_ml_pipeline_auto
from utils.explainability import compute_shap
from utils.text_generator import generate_text_report_openai
from utils.pdf_generator import generate_pdf_report

# Configuración de rutas
ROOT = Path(__file__).resolve().parent
UPLOAD_DIR   = ROOT / "data" / "raw"
REPORTS_DIR  = ROOT / "reports"
GRAPHICS_DIR = ROOT / "graphics"
MODELS_DIR   = ROOT / "models"

for d in (UPLOAD_DIR, REPORTS_DIR, GRAPHICS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ML Autonomous Agent - Web UI")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# Sirve archivos estáticos directamente
app.mount("/reports",  StaticFiles(directory=str(REPORTS_DIR)),  name="reports")
app.mount("/graphics", StaticFiles(directory=str(GRAPHICS_DIR)), name="graphics")

JOBS: dict[str, dict] = {}


# ====================== CAPTURA SOLO TUS PRINTS ======================
class CleanPrintCapture:
    def __init__(self):
        self.old_stdout = sys.stdout
        self.buffer = StringIO()
        sys.stdout = self.buffer

    def get_logs(self):
        return [line.strip() for line in self.buffer.getvalue().splitlines() if line.strip()]

    def stop(self):
        sys.stdout = __import__("sys")
        sys.stdout = self.old_stdout
# ====================================================================


# ====================== BARRA DE PROGRESO ======================
STEPS = [
    ("Cargando dataset...", 5),
    ("Generando EDA...", 20),
    ("Ejecutando AutoML...", 40),
    ("Calculando SHAP...", 15),
    ("Generando texto con OpenAI...", 10),
    ("Generando PDF...", 10),
]

def progress_bar(step: int, subprogress: int = 0, status: str = "") -> str:
    completed = sum(w for i, (_, w) in enumerate(STEPS) if i < step)
    current_weight = STEPS[step][1] if step < len(STEPS) else 0
    percent = min(100, completed + int(current_weight * subprogress / 100))
    filled = "█" * (percent // 4)
    empty  = "░" * (25 - len(filled))
    name   = STEPS[step][0] if step < len(STEPS) else "¡Completado!"
    return f"[{filled}{empty}] {percent:>3}% → {name} {status}".strip()
# =================================================================


def run_pipeline_job(job_id: str, csv_path: str, target_column: str):
    # Seguridad extra
    if job_id not in JOBS:
        return
    job = JOBS[job_id]
    job["status"] = "running"
    job["messages"].clear()
    job["detailed_log"] = []

    capture = CleanPrintCapture()

    try:
        # 0. Carga
        job["messages"].append(progress_bar(0, 0, "Guardando archivo..."))
        time.sleep(0.5)
        df = load_user_dataset(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        target = target_column.strip().lower()

        if target not in df.columns:
            job["status"] = "failed"
            job["messages"].append(f"ERROR: Columna '{target_column}' no encontrada")
            return

        # 1. EDA
        job["messages"][-1] = progress_bar(1, 0)
        eda_path = REPORTS_DIR / f"EDA_report_{job_id}.html"
        for i in range(1, 101, 10):
            time.sleep(0.25)
            job["messages"][-1] = progress_bar(1, i, f"Analizando {i}%")
        generate_eda_report(df, output_html=str(eda_path))

        # 2. AutoML (aquí se capturan tus prints bonitos)
        job["messages"][-1] = progress_bar(2, 0)
        (best_model_name, best_metrics, best_model, preprocessing,
         X_train, X_test, y_test, models) = run_ml_pipeline_auto(df, target)

        for i in range(1, 101, 5):
            time.sleep(0.35)
            job["messages"][-1] = progress_bar(2, i, f"Probando modelos... {i}%")
            job["detailed_log"] = capture.get_logs()

        # 3. SHAP
        job["messages"][-1] = progress_bar(3, 0)
        shap_path = GRAPHICS_DIR / f"shap_{job_id}.png"
        for i in range(1, 101, 15):
            time.sleep(0.4)
            job["messages"][-1] = progress_bar(3, i, "Calculando importancia...")
        compute_shap(best_model, X_test, output_path=str(shap_path))

        # 4. OpenAI
        job["messages"][-1] = progress_bar(4, 0)
        for i in range(1, 101, 20):
            time.sleep(0.7)
            job["messages"][-1] = progress_bar(4, i, "Redactando con IA...")
        report_text = generate_text_report_openai(
            best_model_name, best_metrics,
            "Ver SHAP en el informe.", str(eda_path),
            detect_problem_type(df[target]), models
        )

        # 5. PDF
        job["messages"][-1] = progress_bar(5, 0)
        pdf_path = REPORTS_DIR / f"ml_report_{job_id}.pdf"
        for i in range(1, 101, 12):
            time.sleep(0.3)
            job["messages"][-1] = progress_bar(5, i, "Renderizando PDF...")
        dataset_name = job.get("dataset_name", "dataset")
        target_name  = job.get("target", target_column)
        generate_pdf_report(
            report_text, str(shap_path), best_model_name,
            best_metrics, detect_problem_type(df[target]),
            dataset_name,target_name,output_pdf=str(pdf_path)
        )

        # ÉXITO
        job["status"] = "completed"
        job["messages"][-1] = progress_bar(6)
        job["messages"].append("¡PIPELINE COMPLETADO CON ÉXITO!")
        job["detailed_log"] = capture.get_logs()

        job["outputs"] = {
            "eda": str(eda_path),
            "shap": str(shap_path),
            "pdf": str(pdf_path),
        }

    except Exception as e:
        job["status"] = "failed"
        job["messages"].append(f"ERROR CRÍTICO: {e}")
        job["detailed_log"].extend(["ERROR EN EL PIPELINE"] + traceback.format_exc().splitlines()[-5:])

    finally:
        capture.stop()


# ====================== RUTAS ======================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Página principal con formulario para subir CSV y mostrar jobs recientes.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "jobs": JOBS  # Pasamos los jobs actuales a la plantilla
        }
    )
@app.get("/start", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start", response_class=HTMLResponse)
async def start_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target: str = Form(...)
):
    job_id = uuid.uuid4().hex[:10]
    
    JOBS[job_id] = {
        "status": "queued",
        "messages": [f"Job creado: {job_id}"],
        "detailed_log": [],
        "outputs": {},
        "dataset_name": file.filename,
        "target": target
    }

    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(run_pipeline_job, job_id, str(dest), target)
    return RedirectResponse(url=f"/status/{job_id}", status_code=303)


@app.get("/status/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        job = {"status": "not_found", "messages": ["Job no encontrado"], "detailed_log": []}

    # Cálculo del progreso
    if job["status"] == "completed":
        progress = 100
        current_step = "¡Completado!"
    elif job["status"] in ["failed", "not_found"]:
        progress = 0
        current_step = "Error"
    else:
        last = job["messages"][-1] if job["messages"] else ""
        m = re.search(r"(\d+)%", last)
        progress = int(m.group(1)) if m else 0
        current_step = last.split("→")[-1].strip() if "→" in last else "Procesando..."

    return templates.TemplateResponse(
        "status.html",
        {
            "request": request,
            "job_id": job_id,
            "job": job,
            "progress": progress,
            "current_step": current_step
        }
    )


# ====================== DESCARGA DEL PDF (CORREGIDA) ======================
@app.get("/download/pdf/{job_id}")
async def download_pdf(job_id: str):
    job = JOBS.get(job_id)
    if not job or "pdf" not in job.get("outputs", {}):
        return JSONResponse(status_code=404, content={"error": "PDF no disponible"})
    return FileResponse(
        job["outputs"]["pdf"],
        media_type="application/pdf",
        filename=f"ML_Report_{job_id}.pdf"
    )
# =======================================================================

# (Opcional) 
# @app.get("/download/shap/{job_id}")
# async def download_shap(job_id: str):
#     job = JOBS.get(job_id)
#     if not job or "shap" not in job.get("outputs", {}):
#         return JSONResponse(status_code=404, content={"error": "SHAP no disponible"})
#     return FileResponse(job["outputs"]["shap"], media_type="image/png")