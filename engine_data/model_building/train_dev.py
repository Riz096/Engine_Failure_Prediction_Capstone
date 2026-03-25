import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split


# ==============================
# MLflow Setup
# ==============================

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Engine_Failure_Dev")


# ==============================
# Load Dataset
# ==============================

DATA_PATH = "engine_data/data/engine_data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found")

df = pd.read_csv(DATA_PATH)

TARGET = "Engine Condition"

X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Dataset split completed")


# ==============================
# Preprocessing
# ==============================

numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)


# ==============================
# Models
# ==============================

models = {
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier()
}

results = []
best_model = None
best_recall = 0
best_model_name = ""


# ==============================
# Training Loop
# ==============================

for name, model in models.items():

    with mlflow.start_run(run_name=name):

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

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("Recall:", recall)

        if recall > best_recall:
            best_recall = recall
            best_model = pipeline
            best_model_name = name


# ==============================
# Save Outputs
# ==============================

joblib.dump(best_model, "best_engine_model_dev.pkl")

with open("dev_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nDevelopment model saved")
print(f"Best Model: {best_model_name}")
print(f"Best Recall: {best_recall}")
