"""
Incremental parquet conversion with HuggingFace sync.

Optimized workflow:
1. Detect changed leaderboards via git diff (instant!)
2. Download ONLY changed parquets from HF (fast!)
3. Re-convert ONLY changed leaderboards
4. Ready for upload (handled by upload_to_hf.py)

This avoids downloading and processing unchanged leaderboards.
"""

from pathlib import Path
import sys
import subprocess
import os
import json
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent.resolve().parent))

from json_to_parquet import add_to_parquet

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "deepmage121/eee_test")

def download_leaderboards(output_dir: Path, leaderboard_names: set[str]) -> set[str]:
    """Download existing leaderboard parquets from HuggingFace."""
    try:
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token) if hf_token else HfApi()
        
        files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
        
        available_leaderboards = set()
        for file in files:
            if file.startswith("data/") and file.endswith(".parquet"):
                lb_name = file[5:-8]
                available_leaderboards.add(lb_name)
        
        downloaded: set[str] = set()
        
        for lb in leaderboard_names:
            if lb in available_leaderboards:
                print(f"  Downloading {lb}")
                file_path = f"data/{lb}.parquet"
                local_path = output_dir / f"{lb}.parquet"
                
                # Download directly to the target location
                downloaded_path = api.hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    filename=file_path,
                )
                
                # Move to output directory with correct name
                import shutil
                shutil.copy(downloaded_path, local_path)
                print(f"    → {local_path}")
                downloaded.add(lb)
            else:
                print(f"  {lb} (new)")
        
        print(f"Downloaded {len(downloaded)}/{len(leaderboard_names)} parquet(s)")
        return downloaded
        
    except Exception as e:
        print(f"HF download failed: {e}")
        print("Treating all leaderboards as new (no existing data to download)")
        return set()


def detect_modified_leaderboards() -> dict[str, list[str]]:
    """Get leaderboards with changed JSONs and the specific files that changed."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD", "data/"],
            capture_output=True, text=True, check=True
        )
        
        changed_files = result.stdout.strip().split('\n')
        if not changed_files or changed_files == ['']:
            print("No changes detected in data/")
            return {}
        
        # Group changed files by leaderboard
        leaderboards_to_files: dict[str, list[str]] = {}
        for f in changed_files:
            if f.startswith('data/') and f.endswith('.json') and len(Path(f).parts) >= 2:
                leaderboard = Path(f).parts[1]
                if leaderboard not in leaderboards_to_files:
                    leaderboards_to_files[leaderboard] = []
                leaderboards_to_files[leaderboard].append(f)
        
        return leaderboards_to_files
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Git command failed: {e}")
        sys.exit(1)


def convert_changed_leaderboards():
    """
    Optimized conversion: detect changes, download only changed, process ONLY changed files.
    """
    
    data_dir = Path("data")
    output_dir = Path("parquet_output")
    output_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    leaderboards_to_files: dict[str, list[str]] = detect_modified_leaderboards()
    
    if len(leaderboards_to_files) == 0:
        print("No changes detected, nothing to upload")
        manifest = {"changed": [], "converted": []}
        (output_dir / "changed_leaderboards.json").write_text(json.dumps(manifest, indent=2))
        sys.exit(0)
    
    changed_leaderboards = set(leaderboards_to_files.keys())
    total_changed_files = sum(len(files) for files in leaderboards_to_files.values())
    
    print(f"Detected {len(changed_leaderboards)} changed leaderboard(s) with {total_changed_files} changed file(s):")
    for lb, files in leaderboards_to_files.items():
        print(f"  {lb}: {len(files)} file(s)")

    downloaded = download_leaderboards(output_dir, changed_leaderboards)
    
    converted_count = 0
    error_count = 0
    converted_leaderboards = []
    
    for leaderboard_name, changed_files in leaderboards_to_files.items():
        parquet_path = os.path.join(output_dir, f"{leaderboard_name}.parquet")
        leaderboard_dir = os.path.join(data_dir, leaderboard_name)
        
        # If no existing parquet, process entire directory (first time or recovery)
        if not Path(parquet_path).exists():
            print(f"\nConverting: {leaderboard_name} (FULL - no existing parquet)")
            try:
                add_to_parquet(str(leaderboard_dir), str(parquet_path))
                print(f"   Created {parquet_path}")
                converted_count += 1
                converted_leaderboards.append(leaderboard_name)
            except Exception as e:
                print(f"   Error: {e}")
                error_count += 1
            continue
        
        print(f"\nConverting: {leaderboard_name} ({len(changed_files)} changed file(s))")
        
        try:
            # Process only changed files for incremental update
            add_to_parquet(changed_files, str(parquet_path))
            
            print(f"   Updated {parquet_path}")
            converted_count += 1
            converted_leaderboards.append(leaderboard_name)
            
        except Exception as e:
            print(f"   Error: {e}")
            error_count += 1
    
    manifest = {
        "changed": list(changed_leaderboards),
        "converted": converted_leaderboards,
        "downloaded": list(downloaded),
        "errors": error_count
    }
    manifest_path = os.path.join(output_dir, "changed_leaderboards.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    if error_count > 0:
        sys.exit(1)
    
    if converted_count == 0:
        print("Warning: No parquet files successfully converted!")
        sys.exit(1)


if __name__ == "__main__":
    convert_changed_leaderboards()

