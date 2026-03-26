import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import json
import os

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

os.makedirs("mlruns", exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Engine_Failure_Dev")


# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("engine_data/data/engine_data.csv")

# ==============================
# SAFE COLUMN STANDARDIZATION
# ==============================

df.columns = df.columns.str.strip()

column_mapping = {
    df.columns[0]: "Engine rpm",
    df.columns[1]: "Lub oil pressure",
    df.columns[2]: "Fuel pressure",
    df.columns[3]: "Coolant pressure",
    df.columns[4]: "Lub oil temp",
    df.columns[5]: "Coolant temp",
    df.columns[6]: "Engine Condition"
}

df = df.rename(columns=column_mapping)

TARGET = "Engine Condition"

if TARGET not in df.columns:
    raise ValueError("Target column not found after renaming")


# ==============================
# Train-Test Split
# ==============================

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
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}


# ==============================
# Training Loop
# ==============================

results = []
best_model = None
best_recall = 0
best_model_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(Xtrain, ytrain)

        y_pred = pipeline.predict(Xtest)

        acc = accuracy_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred, zero_division=0)
        precision = precision_score(ytest, y_pred, zero_division=0)
        f1 = f1_score(ytest, y_pred, zero_division=0)

        # ==============================
        # MLflow Logging
        # ==============================

        mlflow.log_param("model_type", name)

        mlflow.log_metrics({
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        })

        # Save metrics as artifact (unique per model)
        metrics_dict = {
            "model": name,
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        }

        file_name = f"{name}_metrics.json"

        with open(file_name, "w") as f:
            json.dump(metrics_dict, f)

        mlflow.log_artifact(file_name)

        # 🔥 Log model (IMPORTANT)
        mlflow.sklearn.log_model(pipeline, "model")

        # ==============================
        # Track Results
        # ==============================

        results.append(metrics_dict)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("Recall:", recall)

        if recall > best_recall:
            best_recall = recall
            best_model = pipeline
            best_model_name = name


# ==============================
# Save Best Model
# ==============================

joblib.dump(best_model, "best_engine_model_dev.pkl")

with open("dev_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nBest Model:", best_model_name)
print("Best Recall:", best_recall)
print("Development model saved successfully")

