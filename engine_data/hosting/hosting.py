
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ==============================
# Configuration
# ==============================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found")

# ✅ Use lowercase (HF best practice)
SPACE_REPO = "rizwan9/engine-failure-prediction-app"

DEPLOYMENT_PATH = "engine_data/deployment"

api = HfApi(token=HF_TOKEN)


# ==============================
# Validate Deployment Files
# ==============================

required_files = ["app.py", "requirements.txt"]

for file in required_files:
    if not os.path.exists(f"{DEPLOYMENT_PATH}/{file}"):
        raise FileNotFoundError(f"{file} not found in deployment folder")


# ==============================
# Create Space if not exists
# ==============================

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print("Space already exists")

except RepositoryNotFoundError:
    print("Creating Hugging Face Space...")

    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        token=HF_TOKEN
    )

    print("Space created successfully")


# ==============================
# UPLOAD TO HF (FIXED)
# ==============================

print("Uploading model to Hugging Face...")

api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

# Correct usage (named arguments)
api.upload_file(
    path_or_fileobj="best_engine_model.pkl",
    path_in_repo="best_engine_model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="metrics.json",
    path_in_repo="metrics.json",
    repo_id=MODEL_REPO,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="best_xgb_params.json",
    path_in_repo="best_xgb_params.json",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Model uploaded successfully to Hugging Face")
print(f"Best Model: {best_model_name}")
print(f"Best Recall: {best_recall}")
