import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from data_utils.dataset_loader_EDA_generator import detect_problem_type
from data_utils.preprocessing import build_preprocessing_pipeline


# 5) PIPELINE DE ENTRENAMIENTO AUTOMATICO (AutoML Full)
def run_ml_pipeline_auto(df, target_column, problem_type=None):
    """
    Pipeline completamente automático:
    - Detecta problema si no se proporciona
    - Construye preprocessing dinámico
    - Entrena modelos (RandomForest + XGBoost)
    - Evalúa y selecciona el mejor
    - Guarda modelo y pipeline
    Retorna: best_model_name, metrics, fitted_best_model, preprocessing, feature_names_df
    """
    if problem_type is None:
        problem_type = detect_problem_type(df[target_column])

    preprocessing, num_cols, low_card_cat, high_card_cat = build_preprocessing_pipeline(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if problem_type == "classification" and y.dtype != "O":
        y = y.astype(int) - y.min()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21,
        stratify=None if problem_type == "regression" else y
    )

    preprocessing.fit(X_train)
    X_train_proc = preprocessing.transform(X_train)
    X_test_proc = preprocessing.transform(X_test)

    try:
        features = preprocessing.get_feature_names_out()
    except:
        features = [f"f{i}" for i in range(X_train_proc.shape[1])]

    X_train_df = pd.DataFrame(X_train_proc, columns=features)
    X_test_df = pd.DataFrame(X_test_proc, columns=features)

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

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"Entrenando {name}…")
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

    best_model_name = (
        min(results, key=lambda x: results[x]["MSE"])
        if problem_type == "regression"
        else max(results, key=lambda x: results[x]["Accuracy"])
    )

    joblib.dump(preprocessing, "models/preprocessing_pipeline.pkl")

    return (
        best_model_name,
        results[best_model_name],
        fitted_models[best_model_name],
        preprocessing,
        X_train_df,
        X_test_df,
        y_test,
        models
    )