
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ==============================
# Configuration
# ==============================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found")

SPACE_REPO = "Rizwan9/engine-failure-prediction-app"

api = HfApi(token=HF_TOKEN)

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
        space_sdk="docker",
        private=False,
        token=HF_TOKEN
    )

# ==============================
# Upload Files
# ==============================

print("Uploading deployment files...")

api.upload_folder(
    folder_path="engine_data/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
    commit_message="Updated deployment files",
    ignore_patterns=["*.ipynb", "__pycache__", "mlruns"]
)

print("Deployment files uploaded successfully")
