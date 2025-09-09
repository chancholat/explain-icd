import shutil
import kagglehub
from huggingface_hub import snapshot_download


#Download the latest version.
src = kagglehub.dataset_download('chanhainguyen/thesis-data-process')
dest = './data'
# Copy entire folder (like `cp -r`)
shutil.copytree(src, dest, dirs_exist_ok=True)
shutil.rmtree(src)  # Remove the temp folder created by kagglehub
print(f"[INFO] Copied dataset from {src} -> {dest}")


# tạo thư mục models nếu chưa tồn tại
import os
os.makedirs('./models')
# Tải toàn bộ repository về thư mục local
snapshot_download(repo_id="ChanBeDu/PLM-ICD-seed10", repo_type="model", local_dir="./models")
snapshot_download(repo_id="ChanBeDu/reference_model", local_dir="./models/")