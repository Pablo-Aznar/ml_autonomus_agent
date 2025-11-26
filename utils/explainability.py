# utils/explainability.py 
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def compute_shap(model, X_proc_df: pd.DataFrame, output_path: str = "graphics/shap_summary.png") -> str:
    """
    SHAP 100% compatible con XGBoost 2.0+, scikit-learn 1.4+ y Python 3.11+
    """
    print("Calculando SHAP..")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # MUESTREO para velocidad (máximo 1000 filas)
    X_sample = X_proc_df.sample(n=min(1000, len(X_proc_df)), random_state=21)
    
    try:
        # OPCIÓN 1: shap.Explainer con masker explícito
        explainer = shap.Explainer(
            model, 
            masker=X_sample, 
            feature_names=X_proc_df.columns.tolist()
        )
        shap_values = explainer(X_sample)

        print("SHAP calculado con Explainer (rápido y exacto)")

    except Exception as e1:
        print(f"Explainer falló, intentando con LinearExplainer para XGBoost...")
        try:
            # OPCIÓN 2: LinearExplainer (funciona perfecto con XGBoost en clasificación)
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer(X_sample)
            print("SHAP calculado con LinearExplainer")
        except Exception as e2:
            print(f"Linear falló, fallback a KernelExplainer simplificado...")
            # OPCIÓN 3: Kernel pero solo para 100 filas
            background = shap.sample(X_proc_df, 50)
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1],  # solo clase positiva
                background,
                nsamples=100
            )
            shap_values = explainer.shap_values(X_sample.head(100))
            print("SHAP calculado con KernelExplainer")

    # Extraer valores SHAP (para clasificación binaria)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # clase positiva
    else:
        shap_values = shap_values.values if hasattr(shap_values, 'values') else shap_values

    # Gráfico
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, 
        X_sample.head(len(shap_values)), 
        show=False, 
        max_display=12,
        plot_type="dot"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    #print(f"Gráfico SHAP guardado: {output_path}")
    #print("=== FIN AUTO-ML PIPELINE ===\n")

    # Resumen textual
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    top3_features = [X_proc_df.columns[i] for i in top3_idx]

    shap_text = (
        f"Análisis SHAP completado\n"
        f"Variables más importantes:\n"
        f"1. {top3_features[0]}\n"
        f"2. {top3_features[1]}\n"
        f"3. {top3_features[2]}"
    )

    print(shap_text)
    print("=== FIN AUTO-ML PIPELINE ===\n")

    return shap_text