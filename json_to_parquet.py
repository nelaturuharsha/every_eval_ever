

import json
from pathlib import Path
import pandas as pd


def json_to_row(json_path: Path) -> dict:
    """Convert one JSON to a single row (1 JSON = 1 row, evaluations as columns)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    required_fields = ["schema_version", "evaluation_id", "evaluation_source", "retrieved_timestamp", 
                      "source_data", "source_metadata", "model_info", "evaluation_results"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"{json_path}: Missing required field '{field}'")
    
    if "evaluation_source_name" not in data["evaluation_source"]:
        raise ValueError(f"{json_path}: Missing required field 'evaluation_source.evaluation_source_name'")
    if "evaluation_source_type" not in data["evaluation_source"]:
        raise ValueError(f"{json_path}: Missing required field 'evaluation_source.evaluation_source_type'")
    
    if "source_organization_name" not in data["source_metadata"]:
        raise ValueError(f"{json_path}: Missing required field 'source_metadata.source_organization_name'")
    if "evaluator_relationship" not in data["source_metadata"]:
        raise ValueError(f"{json_path}: Missing required field 'source_metadata.evaluator_relationship'")
    
    if "name" not in data["model_info"]:
        raise ValueError(f"{json_path}: Missing required field 'model_info.name'")
    if "id" not in data["model_info"]:
        raise ValueError(f"{json_path}: Missing required field 'model_info.id'")
    if "developer" not in data["model_info"]:
        raise ValueError(f"{json_path}: Missing required field 'model_info.developer'")
    
    leaderboard = data["evaluation_source"]["evaluation_source_name"]
    model = data["model_info"]["id"]
    uuid = json_path.stem
    developer = data["model_info"]["developer"]
    
    # Validate evaluation results
    for eval_result in data["evaluation_results"]:
        if "evaluation_name" not in eval_result:
            raise ValueError(f"{json_path}: Missing required field 'evaluation_results[].evaluation_name'")
        if "metric_config" not in eval_result:
            raise ValueError(f"{json_path}: Missing required field 'evaluation_results[].metric_config'")
        if "score_details" not in eval_result:
            raise ValueError(f"{json_path}: Missing required field 'evaluation_results[].score_details'")
        
        if "lower_is_better" not in eval_result["metric_config"]:
            raise ValueError(f"{json_path}: Missing required field 'evaluation_results[].metric_config.lower_is_better'")
        if "score" not in eval_result["score_details"]:
            raise ValueError(f"{json_path}: Missing required field 'evaluation_results[].score_details.score'")
    
    row = {
        # Folder structure (for reconstruction)
        "_leaderboard": leaderboard,
        "_developer": developer,
        "_model": model,
        "_uuid": uuid,
        
        # Required top-level fields
        "schema_version": data["schema_version"],
        "evaluation_id": data["evaluation_id"],
        "retrieved_timestamp": data["retrieved_timestamp"],
        "source_data": json.dumps(data["source_data"]),
        
        # Required nested fields
        "evaluation_source_name": data["evaluation_source"]["evaluation_source_name"],
        "evaluation_source_type": data["evaluation_source"]["evaluation_source_type"],
        
        "source_organization_name": data["source_metadata"]["source_organization_name"],
        "source_organization_url": data["source_metadata"].get("source_organization_url"),
        "source_organization_logo_url": data["source_metadata"].get("source_organization_logo_url"),
        "evaluator_relationship": data["source_metadata"]["evaluator_relationship"],
        
        "model_name": data["model_info"]["name"],
        "model_id": data["model_info"]["id"],
        "model_developer": data["model_info"]["developer"],
        "model_inference_platform": data["model_info"].get("inference_platform"),
        
        # Store full evaluation_results and additional_details as JSON
        "evaluation_results": json.dumps(data["evaluation_results"]),
        "additional_details": json.dumps(data["additional_details"]) if "additional_details" in data else None,
    }
    
    return row


def add_to_parquet(json_or_folder: str, parquet_file: str):
    """
    Add JSON(s) to Parquet file.
    Creates new file if it doesn't exist, appends and deduplicates if it does.
    
    Args:
        json_or_folder: Path to single JSON file or folder containing JSONs
        parquet_file: Output Parquet file path
    """
    input_path = Path(json_or_folder)
    
    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = list(input_path.rglob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {json_or_folder}")
    else:
        raise ValueError(f"Invalid input: {json_or_folder}")
    
    print(f"Processing {len(json_files)} JSON file(s)...")
    
    parquet_path = Path(parquet_file)
    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_file)
        existing_keys = set(
            existing_df[["_leaderboard", "_developer", "_model", "_uuid"]]
            .apply(tuple, axis=1)
        )
        print(f"Found {len(existing_df)} existing rows")
    else:
        existing_df = None
        existing_keys = set()
    
    all_rows = []
    skipped = 0
    for i, jf in enumerate(json_files, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(json_files)}")
        
        row = json_to_row(jf)
        key = (row["_leaderboard"], row["_developer"], row["_model"], row["_uuid"])
        if key not in existing_keys:
            all_rows.append(row)
            existing_keys.add(key)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped} duplicate file(s)")
    
    # Handle case where no new rows to add
    if not all_rows:
        if existing_df is not None:
            print(f"No new files to add, keeping existing {len(existing_df)} file(s)")
            return
        else:
            raise ValueError("No valid JSON files to process and no existing parquet file")
    
    new_df = pd.DataFrame(all_rows)
    
    if existing_df is not None:
        df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Added {len(new_df)} new file(s) to existing {len(existing_df)} file(s)")
    else:
        df = new_df
    
    df.to_parquet(parquet_file, index=False)
    print(f"Saved {len(df)} total file(s) to {parquet_file} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")


def parquet_to_folder(parquet_file: str, output_dir: str):
    """Reconstruct folder structure from Parquet."""
    df = pd.read_parquet(parquet_file)
    out = Path(output_dir)
    
    for _, row in df.iterrows():
        lb = row["_leaderboard"]
        dev = row["_developer"]
        model = row["_model"]
        uuid = row["_uuid"]
        
        json_data = {
            "schema_version": row["schema_version"],
            "evaluation_id": row["evaluation_id"],
            "retrieved_timestamp": row["retrieved_timestamp"],
            "source_data": json.loads(row["source_data"]),
            "evaluation_source": {
                "evaluation_source_name": row["evaluation_source_name"],
                "evaluation_source_type": row["evaluation_source_type"]
            },
            "source_metadata": {
                "source_organization_name": row["source_organization_name"],
                "evaluator_relationship": row["evaluator_relationship"]
            },
            "model_info": {
                "name": row["model_name"],
                "id": row["model_id"],
                "developer": row["model_developer"]
            },
            "evaluation_results": json.loads(row["evaluation_results"])
        }
        
        if pd.notna(row["source_organization_url"]):
            json_data["source_metadata"]["source_organization_url"] = row["source_organization_url"]
        if pd.notna(row["source_organization_logo_url"]):
            json_data["source_metadata"]["source_organization_logo_url"] = row["source_organization_logo_url"]
        
        if pd.notna(row["model_inference_platform"]):
            json_data["model_info"]["inference_platform"] = row["model_inference_platform"]
        
        if pd.notna(row["additional_details"]):
            json_data["additional_details"] = json.loads(row["additional_details"])
        
        file_path = out / lb / dev / model / f"{uuid}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    print(f"Reconstructed {len(df)} files to {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python json_to_parquet.py add <json_or_folder> <output.parquet>")
        print("  python json_to_parquet.py export <input.parquet> <output_dir>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "add":
        add_to_parquet(sys.argv[2], sys.argv[3])
    elif cmd == "export":
        parquet_to_folder(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {cmd}")
