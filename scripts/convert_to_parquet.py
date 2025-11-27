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
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.resolve().parent))

from json_to_parquet import add_to_parquet

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "deepmage121/eee_test")

def download_leaderboards(output_dir: Path, leaderboard_names: set[str]) -> set[str]:
    """Download existing leaderboard parquets from HuggingFace."""
    try:
        dataset_dict = load_dataset(HF_DATASET_REPO)
        downloaded: set[str] = set()
        
        for lb in leaderboard_names:
            if lb in dataset_dict:
                print(f"  Downloading {lb}")
                dataset_dict[lb].to_pandas().to_parquet(output_dir / f"{lb}.parquet", index=False)
                downloaded.add(lb)
            else:
                print(f"  {lb} (new)")
        
        print(f"Downloaded {len(downloaded)}/{len(leaderboard_names)} parquet(s)")
        return downloaded
        
    except Exception as e:
        print(f"HF download failed: {e}")
        sys.exit(1)


def detect_modified_leaderboards() -> set[str]:
    """Get leaderboards with changed JSONs via git diff (HEAD~1)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD", "data/"],
            capture_output=True, text=True, check=True
        )
        
        changed_files = result.stdout.strip().split('\n')
        if not changed_files or changed_files == ['']:
            print("No changes detected in data/")
            return set()
        
        leaderboards = {
            Path(f).parts[1] 
            for f in changed_files 
            if f.startswith('data/') and f.endswith('.json') and len(Path(f).parts) >= 2
        }
        return leaderboards
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Git command failed: {e}")
        sys.exit(1)


def convert_changed_leaderboards():
    """
    Optimized conversion: detect changes, download only changed, re-convert only changed.
    """
    
    data_dir = Path("data")
    output_dir = Path("parquet_output")
    output_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    changed_leaderboards: set[str] = detect_modified_leaderboards()
    
    if len(changed_leaderboards) == 0:
        print("No changes detected, nothing to upload")
        manifest = {"changed": [], "converted": []}
        (output_dir / "changed_leaderboards.json").write_text(json.dumps(manifest, indent=2))
        sys.exit(0)
    
    print(f"Detected {len(changed_leaderboards)} changed leaderboard(s):")
    for lb in changed_leaderboards:
        print(f"  {lb}")

    downloaded = download_leaderboards(output_dir, changed_leaderboards)
    
    converted_count = 0
    error_count = 0
    converted_leaderboards = []
    
    for leaderboard_name in changed_leaderboards:
        leaderboard_dir = os.path.join(data_dir, leaderboard_name)
        
        parquet_path = os.path.join(output_dir, f"{leaderboard_name}.parquet")
        
        print(f"\nConverting: {leaderboard_name}")
        
        try:
            add_to_parquet(json_or_folder=str(leaderboard_dir), parquet_file=str(parquet_path))
            
            print(f"   Converted to {parquet_path}")
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

