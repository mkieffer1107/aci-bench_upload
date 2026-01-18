import os
import json
import asyncio
import signal
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset
from tqdm import tqdm

load_dotenv(find_dotenv())

from prompt import user_prompt

# Global state for graceful shutdown
shutdown_requested = False

# 14 + 12 = 26 / milliion tokens
# MODELS = ["openai/gpt-5.2", "google/gemini-3-pro-preview"]
# 3 + 0.5 + 2 = 5.5 / million tokens
MODELS = ["google/gemini-3-flash-preview", "x-ai/grok-4.1-fast", "openai/gpt-5-mini"]

SUBSETS = ["aci", "virtassist", "virtscribe"]
SPLITS = ["train", "valid", "test1", "test2", "test3"]

API_KEY = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

# JSON Schema for structured output
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript_errors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "errors": {
                    "type": "array",
                    "description": "List of detected speaker tag swap errors. Empty if no errors found.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lines": {
                                "type": "array",
                                "description": "Range of line numbers [start, end] inclusive. Use [1, -1] if entire transcript is swapped. Use [-1, -1] if no errors found.",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "reason": {
                                "type": "string",
                                "description": "Explanation of why these lines appear to have swapped speaker tags."
                            }
                        },
                        "required": ["lines", "reason"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["errors"],
            "additionalProperties": False
        }
    }
}


async def query_model(
    prompt: str,
    model: str,
    *,
    verbose: bool = False,
) -> tuple[str, list[dict]]:
    """Query a single model and return (model_name, errors)."""
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=RESPONSE_SCHEMA,
        )

        response = completion.choices[0].message.content
        parsed = json.loads(response)
        model_errors = parsed.get("errors", [])

        if verbose:
            # small, useful status line
            n = len(model_errors)
            tqdm.write(f"[done] {model} — {n} error{'s' if n != 1 else ''}")

        return (model, model_errors)

    except Exception as e:
        tqdm.write(f"Error with model {model}: {e}")
        if verbose:
            tqdm.write(f"[done] {model} — FAILED")
        return (model, [])


async def analyze_note(prompt: str, models: list[str]) -> dict[str, list[dict]]:
    """Analyze a note using all models concurrently and return a dict mapping model name to errors."""
    tasks = [query_model(prompt, model, verbose=True) for model in models]
    results = await asyncio.gather(*tasks)
    
    errors_by_model = {}
    for model, errors in results:
        if errors:
            errors_by_model[model] = errors
    
    return errors_by_model

def prepare_transcript(transcript: str) -> str:
    """Prepends line numbers to the transcript"""
    lines = transcript.split("\n")
    for i, line in enumerate(lines):
        lines[i] = f"{i+1}: {line}"
    return "\n".join(lines)

def has_actual_errors(errors_by_model: dict[str, list[dict]]) -> bool:
    """Check if at least one model found actual errors (not [-1, -1])."""
    for model, errors in errors_by_model.items():
        for error in errors:
            lines = error.get("lines", [])
            # [-1, -1] means no errors, anything else is an actual error
            if lines != [-1, -1]:
                return True
    return False

def save_errors(errors: list[dict], subset: str, output_dir: str = "errors") -> None:
    """Save errors to a JSON file named {subset}_errors.json"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subset}_errors.json")
    
    with open(output_path, "w") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    
    tqdm.write(f"Saved {len(errors)} error records to {output_path}")

def load_existing_errors(subset: str, output_dir: str = "errors") -> list[dict]:
    """Load existing errors from a previous run."""
    output_path = os.path.join(output_dir, f"{subset}_errors.json")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return json.load(f)
    return []

def load_progress(output_dir: str = "errors") -> dict[str, set[tuple[str, str]]]:
    """Load progress from existing error files. Returns dict of subset -> set of (split, encounter_id)."""
    progress = {}
    progress_path = os.path.join(output_dir, "progress.json")
    
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            data = json.load(f)
            for subset, entries in data.items():
                progress[subset] = set(tuple(e) for e in entries)
    
    return progress

def save_progress(progress: dict[str, set[tuple[str, str]]], output_dir: str = "errors") -> None:
    """Save progress to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    progress_path = os.path.join(output_dir, "progress.json")
    
    # Convert sets to lists for JSON serialization
    data = {subset: list(entries) for subset, entries in progress.items()}
    
    with open(progress_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    tqdm.write(f"Saved progress to {progress_path}")

def count_models_with_actual_errors(error_record: dict) -> int:
    """Count how many models found actual errors (not [-1, -1]) for a given record."""
    count = 0
    for model, errors in error_record.get("errors", {}).items():
        for error in errors:
            lines = error.get("lines", [])
            if lines != [-1, -1]:
                count += 1
                break  # Only count each model once
    return count

def get_models_with_actual_errors(errors_by_model: dict[str, list[dict]]) -> list[str]:
    """Return list of model names that found actual errors (not [-1, -1])."""
    models = []
    for model, errors in errors_by_model.items():
        for error in errors:
            lines = error.get("lines", [])
            if lines != [-1, -1]:
                models.append(model)
                break  # Only add each model once
    return models

def has_majority_consensus(error_record: dict, num_models: int) -> bool:
    """Check if a majority of models found actual errors."""
    models_with_errors = count_models_with_actual_errors(error_record)
    return models_with_errors > num_models / 2

def save_metadata(
    all_errors: dict[str, list[dict]], 
    totals: dict[str, dict[str, int]],
    num_models: int,
    output_dir: str = "errors"
) -> None:
    """Save metadata with error statistics."""
    metadata = {
        "summary": {
            "total_notes_with_errors": sum(len(errors) for errors in all_errors.values()),
            "total_notes_processed": sum(
                totals[subset][split] 
                for subset in totals 
                for split in totals[subset]
            ),
            "majority_consensus_errors": sum(
                1 for errors in all_errors.values() 
                for e in errors if has_majority_consensus(e, num_models)
            ),
        },
        "by_subset": {},
        "by_split": {},
    }
    
    # Compute stats by subset
    for subset, errors in all_errors.items():
        total_in_subset = sum(totals[subset].values())
        all_models = sum(1 for e in errors if len(e["models_with_errors"]) == num_models)
        majority_consensus = sum(1 for e in errors if has_majority_consensus(e, num_models))
        
        metadata["by_subset"][subset] = {
            "total_notes": total_in_subset,
            "notes_with_errors": len(errors),
            "error_percentage": round(len(errors) / total_in_subset * 100, 2) if total_in_subset > 0 else 0,
            "all_models_found_errors": all_models,
            "majority_consensus_errors": majority_consensus,
        }
    
    # Compute stats by split (across all subsets)
    for split in SPLITS:
        split_errors = [
            e for errors in all_errors.values() 
            for e in errors if e["split"] == split
        ]
        total_in_split = sum(totals[subset].get(split, 0) for subset in totals)
        all_models = sum(1 for e in split_errors if len(e["models_with_errors"]) == num_models)
        majority_consensus = sum(1 for e in split_errors if has_majority_consensus(e, num_models))
        
        metadata["by_split"][split] = {
            "total_notes": total_in_split,
            "notes_with_errors": len(split_errors),
            "error_percentage": round(len(split_errors) / total_in_split * 100, 2) if total_in_split > 0 else 0,
            "all_models_found_errors": all_models,
            "majority_consensus_errors": majority_consensus,
        }
    
    # Compute stats by subset and split
    metadata["by_subset_and_split"] = {}
    for subset, errors in all_errors.items():
        metadata["by_subset_and_split"][subset] = {}
        for split in SPLITS:
            split_errors = [e for e in errors if e["split"] == split]
            total_in_split = totals[subset].get(split, 0)
            all_models = sum(1 for e in split_errors if len(e["models_with_errors"]) == num_models)
            majority_consensus = sum(1 for e in split_errors if has_majority_consensus(e, num_models))
            
            metadata["by_subset_and_split"][subset][split] = {
                "total_notes": total_in_split,
                "notes_with_errors": len(split_errors),
                "error_percentage": round(len(split_errors) / total_in_split * 100, 2) if total_in_split > 0 else 0,
                "all_models_found_errors": all_models,
                "majority_consensus_errors": majority_consensus,
            }
    
    # Compute stats by transcript_version
    all_flat_errors = [e for errors in all_errors.values() for e in errors]
    transcript_versions = set(e["transcript_version"] for e in all_flat_errors)
    metadata["by_transcript_version"] = {}
    for version in transcript_versions:
        version_errors = [e for e in all_flat_errors if e["transcript_version"] == version]
        all_models = sum(1 for e in version_errors if len(e["models_with_errors"]) == num_models)
        majority_consensus = sum(1 for e in version_errors if has_majority_consensus(e, num_models))
        
        metadata["by_transcript_version"][version] = {
            "notes_with_errors": len(version_errors),
            "all_models_found_errors": all_models,
            "majority_consensus_errors": majority_consensus,
        }
    
    # Compute stats by subset and transcript_version
    metadata["by_subset_and_transcript_version"] = {}
    for subset, errors in all_errors.items():
        metadata["by_subset_and_transcript_version"][subset] = {}
        subset_versions = set(e["transcript_version"] for e in errors)
        for version in subset_versions:
            version_errors = [e for e in errors if e["transcript_version"] == version]
            all_models = sum(1 for e in version_errors if len(e["models_with_errors"]) == num_models)
            majority_consensus = sum(1 for e in version_errors if has_majority_consensus(e, num_models))
            
            metadata["by_subset_and_transcript_version"][subset][version] = {
                "notes_with_errors": len(version_errors),
                "all_models_found_errors": all_models,
                "majority_consensus_errors": majority_consensus,
            }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metadata.json")
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    tqdm.write(f"Saved metadata to {output_path}")

async def main():
    global shutdown_requested
    
    all_errors = {}  # subset -> list of error records
    totals = {}  # subset -> split -> count of notes processed
    progress = load_progress()  # subset -> set of (split, encounter_id)
    
    # Signal handler for graceful shutdown
    def handle_shutdown(signum, frame):
        global shutdown_requested
        print("\n\nShutdown requested. Saving current progress...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # First pass: count total notes for overall progress
    print("Loading datasets...")
    total_notes = 0
    already_processed = 0
    datasets_cache = {}
    for subset in SUBSETS:
        datasets_cache[subset] = load_dataset("mkieffer/ACI-Bench-MedARC", name=subset)
        for split in SPLITS:
            split_len = len(datasets_cache[subset][split])
            total_notes += split_len
            # Count already processed
            if subset in progress:
                for row in datasets_cache[subset][split]:
                    if (split, row["encounter_id"]) in progress[subset]:
                        already_processed += 1
    
    remaining = total_notes - already_processed
    print(f"Total notes: {total_notes}")
    print(f"Already processed: {already_processed}")
    print(f"Remaining: {remaining}")
    print(f"Subsets: {SUBSETS}")
    print(f"Splits: {SPLITS}")
    print("-" * 50)
    
    # If everything is already processed, just generate metadata from existing files
    if remaining == 0:
        print("All notes already processed. Generating metadata from existing files...")
        all_errors = {}
        totals = {}
        for subset in SUBSETS:
            all_errors[subset] = load_existing_errors(subset)
            totals[subset] = {}
            for split in SPLITS:
                totals[subset][split] = len(datasets_cache[subset][split])
        save_metadata(all_errors, totals, num_models=len(MODELS))
        print("Done.")
        return
    
    # Overall progress bar (for remaining items)
    overall_pbar = tqdm(total=total_notes, initial=already_processed, desc="Overall Progress", position=0)
    
    try:
        for subset in SUBSETS:
            if shutdown_requested:
                break
                
            dataset = datasets_cache[subset]
            # Load existing errors for this subset
            subset_errors = load_existing_errors(subset)
            totals[subset] = {}
            
            # Initialize progress set for this subset if not exists
            if subset not in progress:
                progress[subset] = set()
            
            tqdm.write(f"\nProcessing subset: {subset}")
            
            for split in SPLITS:
                if shutdown_requested:
                    break
                    
                ds = dataset[split]
                totals[subset][split] = len(ds)
                
                # if subset != "aci" or split != "test1":
                #     overall_pbar.update(len(ds))
                #     continue

                # Split progress bar
                split_pbar = tqdm(
                    ds, 
                    desc=f"  {subset}/{split}", 
                    position=1, 
                    leave=False
                )
                
                for row in split_pbar:
                    if shutdown_requested:
                        break
                        
                    encounter_id = row["encounter_id"]
                    
                    # Skip if already processed
                    if (split, encounter_id) in progress[subset]:
                        continue
                    
                    transcript = row["dialogue"]
                    transcript_version = row["transcript_version"]

                    prepared_transcript = prepare_transcript(transcript)

                    prompt = user_prompt.format(transcript=prepared_transcript)
                    errors_by_model = await analyze_note(prompt, MODELS)
                    
                    # Only save if at least one model found actual errors (not [-1, -1])
                    if errors_by_model and has_actual_errors(errors_by_model):
                        subset_errors.append({
                            "subset": subset,
                            "split": split,
                            "encounter_id": encounter_id,
                            "transcript_version": transcript_version,
                            "models_with_errors": get_models_with_actual_errors(errors_by_model),
                            "errors": errors_by_model
                        })
                    
                    # Mark as processed
                    progress[subset].add((split, encounter_id))
                    overall_pbar.update(1)
                
                split_pbar.close()
            
            all_errors[subset] = subset_errors
            save_errors(subset_errors, subset)
            save_progress(progress)
            tqdm.write(f"  Found {len(subset_errors)} transcripts with errors in {subset}")
    
    finally:
        # Always save on exit (normal or interrupted)
        overall_pbar.close()
        print("\n" + "-" * 50)
        
        # Save any remaining data
        for subset in SUBSETS:
            if subset in all_errors:
                save_errors(all_errors[subset], subset)
        save_progress(progress)
        
        if not shutdown_requested:
            save_metadata(all_errors, totals, num_models=len(MODELS))
        else:
            print("Interrupted. Progress saved. Run again to resume.")

if __name__ == "__main__":
    asyncio.run(main())

