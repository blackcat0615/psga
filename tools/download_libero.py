"""
Download functionalities adapted from Mandlekar et. al.: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""
import os
import time
from tqdm import tqdm

from pathlib import Path
import zipfile
import io
import urllib.request
import shutil



try:
    from huggingface_hub import snapshot_download
    import shutil
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False




DATASET_LINKS = {
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def download_from_huggingface(dataset_name, download_dir, check_overwrite=True):
    """
    Download a specific LIBERO dataset from Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset to download (e.g., 'libero_spatial')
        download_dir (str): Directory where the dataset should be downloaded
        check_overwrite (bool): If True, will check if dataset already exists
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "Hugging Face Hub is not available. Install it with 'pip install huggingface_hub'"
        )
    
    # Create the destination folder
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if dataset already exists
    dataset_dir = os.path.join(download_dir, dataset_name)
    if check_overwrite and os.path.exists(dataset_dir):
        user_response = input(
            f"Warning: dataset {dataset_name} already exists at {dataset_dir}. Overwrite? y/n\n"
        )
        if user_response.lower() not in {"yes", "y"}:
            print(f"Skipping download of {dataset_name}")
            return
        
        # Remove existing directory
        print(f"Removing existing folder: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    # Download the dataset
    print(f"Downloading {dataset_name} from Hugging Face...")
    folder_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=f"{dataset_name}/*",
        local_dir_use_symlinks=False,  # Prevents using symlinks to cached files
        force_download=True  # Forces re-downloading files
    )
    
    # Verify downloaded files
    file_count = sum([len(files) for _, _, files in os.walk(os.path.join(download_dir, dataset_name))])
    print(f"Downloaded {file_count} files for {dataset_name}")


if __name__ == "__main__":

    download_dir = "/root/gpufree-data/sim_vla/SimVLA/datasets/metas"
    datasets_to_download = [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
    ] 

    for dataset_name in datasets_to_download:
        print(f"Downloading {dataset_name}")
        
        download_from_huggingface(
            dataset_name=dataset_name,
            download_dir=download_dir,
            check_overwrite=True,
        )