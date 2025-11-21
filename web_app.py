# web_app.py
import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# import your utils
from utils.dataset_loader_EDA_generator import load_user_dataset, generate_eda_report, detect_problem_type
from utils.automl_pipeline import run_ml_pipeline_auto
from utils.explainability import compute_shap
from utils.text_generator import generate_text_report_openai
from utils.pdf_generator import generate_pdf_report

# Config
ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
GRAPHICS_DIR = ROOT / "graphics"
MODELS_DIR = ROOT / "models"
for d in (UPLOAD_DIR, REPORTS_DIR, GRAPHICS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# FastAPI + templates
app = FastAPI(title="ML Autonomous Agent - Web UI (MVP)")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# serve static (reports, graphics)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/graphics", StaticFiles(directory=str(GRAPHICS_DIR)), name="graphics")

# In-memory job store (MVP). For production use Redis / DB.
JOBS = {}  # job_id -> {status, messages, paths...}

def run_pipeline_job(job_id: str, csv_path: str, target_column: str):
    """
    Background worker that executes the full pipeline and stores file paths in JOBS.
    """
    JOBS[job_id]["status"] = "running"
    try:
        JOBS[job_id]["messages"].append("Cargando dataset...")
        df = load_user_dataset(csv_path)

        # Normalizar nombres de columnas para evitar errores
        df.columns = df.columns.str.strip().str.lower()
        target_normalized = target_column.strip().lower()

        # Verificar que la columna objetivo existe antes de seguir
        if target_normalized not in df.columns:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["messages"].append(
                f"ERROR: La columna objetivo '{target_column}' no está en el dataset. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )
            return

        JOBS[job_id]["messages"].append("Generando EDA...")
        eda_path = REPORTS_DIR / f"EDA_report_{job_id}.html"
        generate_eda_report(df, output_html=str(eda_path))

        JOBS[job_id]["messages"].append("Ejecutando AutoML...")
        best_model_name, best_metrics, best_model, preprocessing, X_train, X_test, y_test, models = run_ml_pipeline_auto(df, target_column)

        JOBS[job_id]["messages"].append("Calculando SHAP...")
        shap_path = GRAPHICS_DIR / f"shap_{job_id}.png"
        compute_shap(best_model, X_test, output_path=str(shap_path))

        JOBS[job_id]["messages"].append("Generando texto con OpenAI...")
        # pass models dictionary - your text_generator expects it
        report_text = generate_text_report_openai(best_model_name, best_metrics, "Ver SHAP en el informe.", str(eda_path), detect_problem_type(df[target_column]), models)

        JOBS[job_id]["messages"].append("Generando PDF...")
        pdf_path = REPORTS_DIR / f"ml_report_{job_id}.pdf"
        generate_pdf_report(report_text, str(shap_path), best_model_name, best_metrics, detect_problem_type(df[target_column]), output_pdf=str(pdf_path))

        # Save outputs in job
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["outputs"] = {
            "eda": str(eda_path),
            "shap": str(shap_path),
            "pdf": str(pdf_path),
            "model": f"models/{best_model_name}.pkl"
        }
        JOBS[job_id]["messages"].append("Completado correctamente.")
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["messages"].append(f"Error: {e}")
        import traceback
        JOBS[job_id]["messages"].append(traceback.format_exc())

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Formulario principal para subir un CSV y seleccionar la columna objetivo.
    """
    return templates.TemplateResponse("index.html", {"request": request, "jobs": JOBS})

@app.post("/start", response_class=HTMLResponse)
async def start_pipeline(request: Request, background_tasks: BackgroundTasks,
                         file: UploadFile = File(...), target: str = Form(...)):
    """
    Recibe CSV y target; guarda el archivo y lanza el pipeline en background.
    Redirige a la página de estado del job.
    """
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "queued", "messages": [], "outputs": {}}

    # save uploaded file
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id]["messages"].append(f"Dataset guardado: {dest}")
    # launch background work
    background_tasks.add_task(run_pipeline_job, job_id, str(dest), target)
    
    # Redirect to job status page
    return RedirectResponse(url=f"/status/{job_id}", status_code=303)

@app.get("/status/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return templates.TemplateResponse("status.html", {"request": request, "job_id": job_id, "job": None})
    return templates.TemplateResponse("status.html", {"request": request, "job_id": job_id, "job": job})

# download endpoints
@app.get("/download/pdf/{job_id}")
def download_pdf(job_id: str):
    job = JOBS.get(job_id)
    if not job or "outputs" not in job or "pdf" not in job["outputs"]:
        return {"error": "PDF no disponible aún."}
    return FileResponse(job["outputs"]["pdf"], media_type="application/pdf", filename=f"ml_report_{job_id}.pdf")

@app.get("/download/eda/{job_id}")
def download_eda(job_id: str):
    job = JOBS.get(job_id)
    if not job or "outputs" not in job or "eda" not in job["outputs"]:
        return {"error": "EDA no disponible aún."}
    return FileResponse(job["outputs"]["eda"], media_type="text/html", filename=f"EDA_report_{job_id}.html")

@app.get("/download/shap/{job_id}")
def download_shap(job_id: str):
    job = JOBS.get(job_id)
    if not job or "outputs" not in job or "shap" not in job["outputs"]:
        return {"error": "SHAP no disponible aún."}
    return FileResponse(job["outputs"]["shap"], media_type="image/png", filename=f"shap_{job_id}.png")


# uvicorn web_app:app --reload --port 8000