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
# Upload Files
# ==============================

print("Uploading deployment files...")

try:
    api.upload_folder(
        folder_path=DEPLOYMENT_PATH,
        repo_id=SPACE_REPO,
        repo_type="space",
        path_in_repo="",
        commit_message="Updated deployment files",
        ignore_patterns=["*.ipynb", "__pycache__", "mlruns"]
    )

    print("Deployment files uploaded successfully")

except Exception as e:
    raise RuntimeError(f"Deployment failed: {e}")
