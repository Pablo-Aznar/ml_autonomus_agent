import shap
import matplotlib.pyplot as plt

def compute_shap(model, X_proc_df, output_path="graphics/shap_summary.png"):
    """
    Calcula valores SHAP (aplicable para tree-based) y guarda summary plot.
    Devuelve un breve resumen textual para el informe.
    """
    print("Calculando SHAP…")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc_df)
    except:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_proc_df)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_proc_df, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return "Se calcularon valores SHAP para evaluar la importancia de características."
