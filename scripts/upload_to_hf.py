"""
Upload changed parquet files to HuggingFace dataset.

This script:
1. Reads the manifest of changed leaderboards
2. Uploads ONLY the changed parquet files
3. Uses HfApi for efficient individual file uploads

Usage:
    # With HF_TOKEN environment variable (GitHub Actions):
    python upload_to_hf.py
    
    # Interactive login (local):
    python upload_to_hf.py --login
"""

from huggingface_hub import login, HfFolder, HfApi
import pandas as pd
from pathlib import Path
import sys
import os
import json

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "deepmage121/eee_test")
PARQUET_DIR = Path("parquet_output")
MANIFEST_PATH = PARQUET_DIR / "changed_leaderboards.json"

def upload_changed_parquets():
    """
    Upload only changed parquet files from manifest.
    """
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HF_TOKEN from environment")
        HfFolder.save_token(hf_token)
    elif "--login" in sys.argv:
        print("Logging in to HuggingFace...")
        login()
    else:
        if not HfFolder.get_token():
            print("ERROR: Not logged in. Run with --login flag or set HF_TOKEN environment variable")
            sys.exit(1)
        print("Using existing HuggingFace token")
    
    api = HfApi()
    
    if not MANIFEST_PATH.exists():
        print(f"ERROR: No manifest found at {MANIFEST_PATH}")
        print("Run convert_changed_to_parquet.py first to generate the manifest")
        sys.exit(1)
    
    manifest = json.loads(MANIFEST_PATH.read_text())
    converted_leaderboards = manifest.get("converted", [])
    
    if not converted_leaderboards:
        print("\nNo changed leaderboards to upload (per manifest)")
        sys.exit(0)
    
    print(f"\nManifest found: {len(converted_leaderboards)} leaderboard(s) to upload")
    
    files_to_upload = [
        PARQUET_DIR / f"{lb}.parquet"
        for lb in converted_leaderboards
    ]
    
    files_to_upload = [f for f in files_to_upload if f.exists()]
    
    if not files_to_upload:
        print(f"ERROR: No parquet files to upload in {PARQUET_DIR}")
        sys.exit(1)
    
    print(f"\nUploading {len(files_to_upload)} parquet file(s):")
    for pf in files_to_upload:
        print(f"  - {pf.stem}")
    
    uploaded_count = 0
    error_count = 0
    
    for parquet_file in files_to_upload:
        leaderboard_name = parquet_file.stem
        
        path_in_repo = f"data/{leaderboard_name}/data-00000-of-00001.parquet"
        
        try:
            print(f"\nUploading: {leaderboard_name}")
            
            df = pd.read_parquet(parquet_file)
            print(f"   {len(df)} rows, {len(df.columns)} columns")
            
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update {leaderboard_name} leaderboard data"
            )
            
            print(f"   SUCCESS: Uploaded → {path_in_repo}")
            uploaded_count += 1
            
        except Exception as e:
            print(f"   ERROR: Error uploading {leaderboard_name}: {e}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"Upload Summary:")
    print(f"{'='*70}")
    print(f"  Successfully uploaded: {uploaded_count} file(s)")
    print(f"  Errors:                {error_count} file(s)")
    print(f"{'='*70}")
    
    if error_count > 0:
        print(f"\nWARNING: {error_count} file(s) failed to upload")
        sys.exit(1)
    
    print(f"\nSuccessfully uploaded to HuggingFace!")
    print(f"View at: https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    upload_changed_parquets()