import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder

from utils.dataset_loader_EDA_generator import detect_problem_type
from utils.preprocessing import build_preprocessing_pipeline


def run_ml_pipeline_auto(df, target_column, problem_type=None):
    """
    AutoML pipeline:
    - Detecta el tipo de problema
    - Construye preprocessing completo
    - Codifica target si es necesario
    - Entrena RF y XGBoost
    - Evalúa y selecciona el mejor
    """

    print("\n=== INICIO AUTO-ML PIPELINE ===")

    # 1) Detectar tipo de problema
    if problem_type is None:
        problem_type = detect_problem_type(df[target_column])
    print(f"→ Tipo de problema detectado: {problem_type}")

    # 2) Construir preprocessing dinámico
    preprocessing, num_cols, low_card_cat, high_card_cat = build_preprocessing_pipeline(df, target_column)

    # 3) Separar X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --------------------------------------------------------
    # Codificación correcta del target categórico
    # --------------------------------------------------------
    label_encoder = None
    # CODIFICACIÓN OBLIGATORIA DE ETIQUETAS PARA CLASIFICACIÓN (SIEMPRE de 0 a n-1)
    if problem_type == "classification":
        print("→ Aplicando codificación forzada de etiquetas [0, 1, 2, ...] para clasificación...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, "models/label_encoder.pkl")
        print(f"   Clases originales: {label_encoder.classes_}")
        print(f"   Clases codificadas: {np.unique(y)}")

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=21,
        stratify=y if problem_type == "classification" else None
    )

    # 5) Fit + transform del preprocessing
    preprocessing.fit(X_train)
    X_train_proc = preprocessing.transform(X_train)
    X_test_proc = preprocessing.transform(X_test)

    # Convertir a DataFrame
    try:
        features = preprocessing.get_feature_names_out()
    except:
        features = [f"f{i}" for i in range(X_train_proc.shape[1])]

    X_train_df = pd.DataFrame(X_train_proc, columns=features)
    X_test_df = pd.DataFrame(X_test_proc, columns=features)

    # 6) Definir modelos
    if problem_type == "regression":
        models = {
            "RandomForestRegressor": RandomForestRegressor(random_state=21),
            "XGBRegressor": XGBRegressor(random_state=21, verbosity=0)
        }
    else:
        models = {
            "RandomForestClassifier": RandomForestClassifier(random_state=21),
            "XGBClassifier": XGBClassifier(random_state=21, eval_metric="logloss", verbosity=0)
        }

    # 7) Entrenamiento y evaluación
    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nEntrenando {name}…")

        model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)

        if problem_type == "regression":
            metrics = {
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }
        else:
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted")
            }

        results[name] = metrics
        fitted_models[name] = model

        joblib.dump(model, f"models/{name}.pkl")

    # 8) Seleccionar el mejor modelo
    best_model_name = (
        min(results, key=lambda x: results[x]["MSE"])
        if problem_type == "regression"
        else max(results, key=lambda x: results[x]["Accuracy"])
    )

    joblib.dump(preprocessing, "models/preprocessing_pipeline.pkl")

    print(f"\n→ Mejor modelo: {best_model_name}")
    print("→ Métricas del mejor modelo:", results[best_model_name])
    print("=== FIN AUTO-ML PIPELINE ===\n")

    return (best_model_name,
            results[best_model_name],
            fitted_models[best_model_name],
            preprocessing,
            X_train_df,
            X_test_df,
            y_test,
            models)