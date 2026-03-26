
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ==============================
# Configuration
# ==============================

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Rizwan9/Engine_Failure_Prediction_Capstone"
DATASET_PATH = "hf://datasets/Rizwan9/Engine_Failure_Prediction_Capstone/engine_data.csv"
TARGET_COLUMN = "Engine Condition"

# ==============================
# Validate Token
# ==============================

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please set it as an environment variable.")

api = HfApi(token=HF_TOKEN)

# ==============================
# Load Dataset
# ==============================

print("Loading dataset from Hugging Face...")

try:
    df = pd.read_csv(DATASET_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# ==============================
# Data Cleaning
# ==============================

df = df.drop_duplicates()

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

# ==============================
# 🔥 Column Standardization (VERY IMPORTANT)
# ==============================

df.columns = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "Lub oil temp",
    "Coolant temp",
    "Engine Condition"
]

print("Column names standardized.")

# ==============================
# Train-Test Split
# ==============================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed.")

# ==============================
# Save Locally
# ==============================

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Local files saved successfully.")

# ==============================
# Validate Files
# ==============================

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

# ==============================
# Upload to Hugging Face
# ==============================

for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"splits/{file_path}",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print(f"{file_path} uploaded successfully.")

    except Exception as e:
        raise RuntimeError(f"Failed to upload {file_path}: {e}")

print("Train-test splits uploaded successfully to Hugging Face.")
