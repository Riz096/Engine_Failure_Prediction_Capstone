
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib
import json

from huggingface_hub import HfApi

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# ==============================
# CONFIG
# ==============================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found")

DATASET_BASE = "hf://datasets/Rizwan9/Engine_Failure_Prediction_Capstone/splits/"
MODEL_REPO = "Rizwan9/Engine_Failure_Model"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Engine_Failure_Prod")

api = HfApi(token=HF_TOKEN)


# ==============================
# LOAD DATA FROM HF
# ==============================

Xtrain = pd.read_csv(DATASET_BASE + "Xtrain.csv")
Xtest = pd.read_csv(DATASET_BASE + "Xtest.csv")

ytrain = pd.read_csv(DATASET_BASE + "ytrain.csv").values.ravel()
ytest = pd.read_csv(DATASET_BASE + "ytest.csv").values.ravel()

print("Train-test data loaded")


# ==============================
# CLASS IMBALANCE
# ==============================

class_weight = (ytrain == 0).sum() / max((ytrain == 1).sum(), 1)


# ==============================
# PREPROCESSING
# ==============================

numeric_features = Xtrain.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)


# ==============================
# MODELS
# ==============================

models = {

    "DecisionTree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Bagging": BaggingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150,
        random_state=42
    )
}


# ==============================
# XGBOOST TUNING
# ==============================

xgb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.01, 0.1]
}

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        random_state=42,
        scale_pos_weight=class_weight,
        eval_metric="logloss"
    ))
])

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1
)

xgb_grid.fit(Xtrain, ytrain)

best_xgb = xgb_grid.best_estimator_
models["XGBoost_Tuned"] = best_xgb

print("Best XGBoost Params:", xgb_grid.best_params_)

with open("best_xgb_params.json", "w") as f:
    json.dump(xgb_grid.best_params_, f)


# ==============================
# EXPERIMENT LOOP
# ==============================

results = []
best_model = None
best_recall = 0
best_model_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # ✅ FIX: Avoid double preprocessing
        if name == "XGBoost_Tuned":
            pipeline = model
        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

        pipeline.fit(Xtrain, ytrain)

        y_pred = pipeline.predict(Xtest)

        acc = accuracy_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred)
        precision = precision_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)

        mlflow.log_param("model_type", name)
        mlflow.log_metrics({
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        })

        results.append({
            "model": name,
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        })

        if recall > best_recall:
            best_recall = recall
            best_model = pipeline
            best_model_name = name


# ==============================
# SAVE
# ==============================

joblib.dump(best_model, "best_engine_model.pkl")

with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)

mlflow.sklearn.log_model(best_model, "best_model")


# ==============================
# UPLOAD TO HF
# ==============================

api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

api.upload_file("best_engine_model.pkl", "best_engine_model.pkl", MODEL_REPO, repo_type="model")
api.upload_file("metrics.json", "metrics.json", MODEL_REPO, repo_type="model")
api.upload_file("best_xgb_params.json", "best_xgb_params.json", MODEL_REPO, repo_type="model")


print(f"\nBest Model: {best_model_name}")
print(f"Best Recall: {best_recall}")
print("Model uploaded to Hugging Face")
