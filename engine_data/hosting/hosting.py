from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ==============================
# CONFIGURATION
# ==============================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found")

# ✅ IMPORTANT: Case-sensitive username
SPACE_REPO = "Rizwan9/engine-failure-prediction-app"

DEPLOYMENT_PATH = "engine_data/deployment"

api = HfApi(token=HF_TOKEN)

# ==============================
# VALIDATE FILES
# ==============================

required_files = ["app.py", "requirements.txt"]

for file in required_files:
    file_path = os.path.join(DEPLOYMENT_PATH, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file} not found in {DEPLOYMENT_PATH}")

print("✅ Deployment files verified")

# ==============================
# CREATE SPACE (IF NOT EXISTS)
# ==============================

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print("✅ Space already exists")

except RepositoryNotFoundError:
    print("🚀 Creating Hugging Face Space...")

    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",   # ✅ IMPORTANT FIX
        private=False,
        token=HF_TOKEN
    )

    print("✅ Space created successfully")

# ==============================
# UPLOAD FILES TO SPACE
# ==============================

print("📤 Uploading app files...")

api.upload_folder(
    folder_path=DEPLOYMENT_PATH,
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
    commit_message="Deploying Streamlit App",
    ignore_patterns=["*.ipynb", "__pycache__", "mlruns"]
)

print("🎉 Deployment completed successfully!")
