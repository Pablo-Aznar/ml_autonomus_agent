from utils.dataset_loader_EDA_generator import load_user_dataset, generate_eda_report, detect_problem_type
from utils.automl_pipeline import run_ml_pipeline_auto
from utils.explainability import compute_shap
from utils.text_generator import generate_text_report_openai
from utils.pdf_generator import generate_pdf_report

def main():
    print("\n=== AutoML – ML Autonomous Agent ===\n")

    data_path = input("Ruta del CSV: ").strip()
    target_column = input("Variable objetivo: ").strip()

    df = load_user_dataset(data_path)
    eda_path = generate_eda_report(df)

    best_model_name, best_metrics, best_model, preprocessing, X_train, X_test, y_test, models = (
        run_ml_pipeline_auto(df, target_column))

    shap_img = "graphics/shap_summary.png"
    shap_text = compute_shap(best_model, X_test, shap_img)

    report_text = generate_text_report_openai(
        best_model_name,
        best_metrics,
        shap_text,
        eda_path,
        detect_problem_type(df[target_column]),
        models)

    generate_pdf_report(
        report_text,
        shap_img,
        best_model_name,
        best_metrics,
        detect_problem_type(df[target_column]))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Usa $PORT de Railway o 8000 local
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=True)
