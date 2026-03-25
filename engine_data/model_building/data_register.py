from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import time

# =====================
# Configuration
# =====================

repo_id = "Rizwan9/Engine_Failure_Prediction_Capstone"
repo_type = "dataset"
data_path = "engine_data/data"

HF_TOKEN = os.getenv("HF_TOKEN")

# =====================
# Validate Token
# =====================

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please set it in environment variables.")

api = HfApi(token=HF_TOKEN)

# =====================
# Validate Data Folder
# =====================

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Make sure dataset exists.")

# =================================
# Step 1: Check if repo exists
# =================================

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=HF_TOKEN   # ✅ FIXED
    )

    print(f"Dataset repo '{repo_id}' created successfully.")
    time.sleep(3)  # ✅ FIXED

# =====================
# Step 2: Upload dataset
# =====================

api.upload_folder(
    folder_path=data_path,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Uploading dataset files"
)

print("Dataset files uploaded successfully.")
