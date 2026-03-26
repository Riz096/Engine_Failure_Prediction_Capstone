from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

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

print("Dataset folder found:", data_path)

# =================================
# Step 1: Check if repo exists
# =================================

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=HF_TOKEN  
    )

    print(f"Dataset repo '{repo_id}' created successfully.")

# =====================
# Step 2: Upload dataset
# =====================

print("Uploading dataset to Hugging Face...")

api.upload_folder(
    folder_path=data_path,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Uploading dataset files",
    ignore_patterns=["*.ipynb", "__pycache__", "*.pyc"] 
)

print("Dataset files uploaded successfully.")
