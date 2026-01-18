import os
import json
import argparse
import csv
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, Features, Value
from huggingface_hub import HfApi, create_repo


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def normalize_text(value: object, *, ascii_normalize: bool) -> str:
    """Normalize text to remove common non-standard characters."""
    if value is None:
        text = ""
    else:
        text = str(value)

    if not ascii_normalize:
        return text

    text = unicodedata.normalize("NFKC", text)

    translation_table = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201b": "'",
            "\u2032": "'",
            "\u2035": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u201f": '"',
            "\u2033": '"',
            "\u2036": '"',
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u00a0": " ",
            "\u2007": " ",
            "\u202f": " ",
            "\u2009": " ",
            "\u200a": " ",
            "\u200b": "",
            "\ufeff": "",
            "\u2026": "...",
        }
    )
    text = text.translate(translation_table)

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def normalize_dataset_name(name: str) -> str:
    return (name or "").strip().lower()


def enumerate_dialogue(dialogue: str) -> str:
    """Prepend 1-based line numbers to each line of dialogue."""
    lines = dialogue.split("\n")
    return "\n".join(f"{i+1} {line}" for i, line in enumerate(lines))


def load_challenge_csv_items(
    filepath: str,
    *,
    ascii_normalize: bool,
    transcript_version_by_subset: Dict[str, str],
) -> List[Tuple[str, Dict[str, str]]]:
    """Load the *challenge* split CSV.

    Challenge CSVs include a `dataset` column used to route into Hub configs.
    We additionally attach `transcript_version` based on the requested mapping.
    """
    items: List[Tuple[str, Dict[str, str]]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"dataset", "encounter_id", "dialogue", "note"}
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {filepath}")
        missing = required_columns.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"CSV is missing required columns {sorted(missing)}: {filepath} (found: {reader.fieldnames})"
            )

        for row in reader:
            dataset_value = normalize_text(row.get("dataset", ""), ascii_normalize=ascii_normalize)
            subset = normalize_dataset_name(dataset_value)
            tv = transcript_version_by_subset.get(subset)
            if tv is None:
                raise ValueError(
                    f"Unknown subset/dataset '{dataset_value}' in {filepath}. "
                    f"Expected one of: {sorted(transcript_version_by_subset.keys())}"
                )

            dialogue = normalize_text(row.get("dialogue", ""), ascii_normalize=ascii_normalize)
            record = {
                "encounter_id": normalize_text(row.get("encounter_id", ""), ascii_normalize=ascii_normalize),
                "transcript_version": tv,
                "dialogue": dialogue,
                "enumerated_dialogue": enumerate_dialogue(dialogue),
                "note": normalize_text(row.get("note", ""), ascii_normalize=ascii_normalize),
            }
            items.append((subset, record))

    print(f"Loaded {len(items)} examples from {filepath}")
    return items


def parse_ablation_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Parse {split}_{subset}_{transcript_version}.csv.

    Returns (split, subset, transcript_version) or None if the name doesn't match.
    """
    base = os.path.basename(filename)
    if not base.lower().endswith(".csv"):
        return None
    stem = base[:-4]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    split = parts[0].strip().lower()
    subset = parts[1].strip().lower()
    transcript_version = "_".join(parts[2:]).strip().lower()
    if not split or not subset or not transcript_version:
        return None
    return split, subset, transcript_version


def load_ablation_csv_records(
    filepath: str,
    *,
    ascii_normalize: bool,
    transcript_version: str,
) -> List[Dict[str, str]]:
    """Load ablation CSVs.

    These files typically do NOT have `encounter_id`; they use `id` instead.
    We map `id` -> `encounter_id` to keep a consistent schema.
    """
    records: List[Dict[str, str]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {filepath}")

        fieldnames = set(reader.fieldnames)
        # `dialogue` and `note` are required.
        required_columns = {"dialogue", "note"}
        missing = required_columns.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Ablation CSV is missing required columns {sorted(missing)}: {filepath} (found: {reader.fieldnames})"
            )

        # `id` is preferred; `encounter_id` fallback if present.
        id_field = "id" if "id" in fieldnames else "encounter_id" if "encounter_id" in fieldnames else None

        for row in reader:
            enc_id = normalize_text(row.get(id_field, ""), ascii_normalize=ascii_normalize) if id_field else ""
            dialogue = normalize_text(row.get("dialogue", ""), ascii_normalize=ascii_normalize)

            records.append(
                {
                    "encounter_id": enc_id,
                    "transcript_version": normalize_text(transcript_version, ascii_normalize=ascii_normalize),
                    "dialogue": dialogue,
                    "enumerated_dialogue": enumerate_dialogue(dialogue),
                    "note": normalize_text(row.get("note", ""), ascii_normalize=ascii_normalize),
                }
            )

    print(f"Loaded {len(records)} examples from {filepath}")
    return records


def iter_ablation_csv_paths(data_dir: str) -> Iterable[str]:
    """Yield all ablation CSV paths found under data_dir (recursive)."""
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            # Skip the main challenge files, which are handled explicitly.
            if name in {"train.csv", "valid.csv", "test1.csv", "test2.csv", "test3.csv"}:
                continue
            parsed = parse_ablation_filename(name)
            if parsed is None:
                continue
            yield os.path.join(root, name)


def dedup_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """De-duplicate using the dialogue text.

    We key by (dialogue, transcript_version) so overlapping sources don't create duplicates,
    while still allowing multiple transcript variants to coexist in the same split.
    """
    # Use a plain `set()` typing here for broad Python compatibility (e.g., 3.8+).
    seen = set()
    out: List[Dict[str, str]] = []
    for r in records:
        key = (r.get("dialogue", ""), r.get("transcript_version", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_readme_with_configs(
    out_path: str,
    *,
    configs: List[str],
    splits: List[str],
) -> None:
    """
    Write a README.md that uses *manual data_files configuration*.
    We intentionally do NOT include dataset_info size fields, since those are a common
    source of "size not coherent" errors when stale.
    """
    # Make the first one the default subset in the viewer.
    default_cfg = configs[0] if configs else None

    yaml_lines: List[str] = ["---", "configs:"]
    for cfg in configs:
        yaml_lines.append(f"- config_name: {cfg}")
        if cfg == default_cfg:
            yaml_lines.append("  default: true")
        yaml_lines.append("  data_files:")
        for split in splits:
            # Exact file paths (avoid globs that could accidentally match extra artifacts)
            yaml_lines.append(f"  - split: {split}")
            yaml_lines.append(f"    path: {cfg}/{split}.parquet")
    yaml_lines.append("---")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Format ACI-Bench data and upload as data files to the HF Hub")
    parser.add_argument("--hf_username", default=os.environ.get("HF_USERNAME", "mkieffer"))
    parser.add_argument("--hf_repo_name", default=os.environ.get("HF_REPO_NAME", "ACI-Bench"))
    parser.add_argument(
        "--private",
        default=os.environ.get("PRIVATE", "false"),
        help="Whether the HF dataset repo should be private (true/false)",
    )
    parser.add_argument(
        "--ascii_normalize",
        default=os.environ.get("ASCII_NORMALIZE", "true"),
        help="Normalize text to ASCII (true/false)",
    )
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--work_dir", default=os.environ.get("WORK_DIR", "output"))
    parser.add_argument(
        "--commit_message",
        default="Upload dataset data files (manual configs)",
    )
    args = parser.parse_args()

    hf_username = args.hf_username
    hf_repo_name = args.hf_repo_name
    private = str_to_bool(args.private)
    ascii_normalize = str_to_bool(args.ascii_normalize)
    data_dir = args.data_dir
    work_dir = args.work_dir
    hf_repo_id = f"{hf_username}/{hf_repo_name}"

    os.makedirs(work_dir, exist_ok=True)

    # This folder becomes the repo root we upload.
    upload_root = os.path.join(work_dir, "hub_upload")
    os.makedirs(upload_root, exist_ok=True)

    # Desired configs and splits.
    CONFIGS = ["virtassist", "aci", "virtscribe"]
    SPLITS = ["train", "valid", "test1", "test2", "test3"]

    # transcript_version assignment for *challenge* data (per-user spec)
    CHALLENGE_TRANSCRIPT_VERSION = {
        "aci": "asr",
        "virtassist": "humantrans",
        "virtscribe": "humantrans",
    }

    split_to_file = {
        "train": "train.csv",
        "valid": "valid.csv",
        "test1": "test1.csv",
        "test2": "test2.csv",
        "test3": "test3.csv",
    }

    # Final schema (dataset column removed).
    features = Features(
        {
            "encounter_id": Value("string"),
            "transcript_version": Value("string"),
            "dialogue": Value("string"),
            "enumerated_dialogue": Value("string"),
            "note": Value("string"),
        }
    )

    # Collect all records by split/config.
    split_cfg_records: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        split: {cfg: [] for cfg in CONFIGS} for split in SPLITS
    }

    # 1) Load challenge data.
    for split, filename in split_to_file.items():
        csv_path = os.path.join(data_dir, filename)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Missing required challenge split file: {csv_path}. Make sure your CSVs are in --data_dir."
            )
        items = load_challenge_csv_items(
            csv_path,
            ascii_normalize=ascii_normalize,
            transcript_version_by_subset=CHALLENGE_TRANSCRIPT_VERSION,
        )
        for subset, record in items:
            if subset not in split_cfg_records[split]:
                # Ignore unexpected subsets rather than creating new configs.
                continue
            split_cfg_records[split][subset].append(record)

    # 2) Load ablation data (if present).
    ablation_paths = list(iter_ablation_csv_paths(data_dir))
    if ablation_paths:
        print(f"\nFound {len(ablation_paths)} ablation CSV(s) under {data_dir}")

    for path in ablation_paths:
        parsed = parse_ablation_filename(os.path.basename(path))
        if parsed is None:
            continue
        split, subset, transcript_version = parsed
        if split not in split_cfg_records:
            # Ignore unknown split prefixes.
            continue
        if subset not in split_cfg_records[split]:
            # Only keep the three requested configs.
            continue

        records = load_ablation_csv_records(
            path,
            ascii_normalize=ascii_normalize,
            transcript_version=transcript_version,
        )
        split_cfg_records[split][subset].extend(records)

    # 3) Deduplicate + write Parquet files: <config>/<split>.parquet
    total_duplicates_removed = 0
    for cfg in CONFIGS:
        cfg_dir = os.path.join(upload_root, cfg)
        os.makedirs(cfg_dir, exist_ok=True)

        for split in SPLITS:
            input_records = split_cfg_records[split][cfg]
            input_count = len(input_records)
            records = dedup_records(input_records)
            deduped_count = len(records)
            duplicates_removed = input_count - deduped_count
            total_duplicates_removed += duplicates_removed
            ds = Dataset.from_list(records, features=features)

            out_parquet = os.path.join(cfg_dir, f"{split}.parquet")
            ds.to_parquet(out_parquet)

            # Optional: dump a small JSON sample for debugging locally only
            sample_path = os.path.join(work_dir, f"sample-{cfg}-{split}.json")
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(records[:5], f, ensure_ascii=False, indent=2)

            print(
                f"Wrote {cfg}/{split}.parquet with {ds.num_rows} rows "
                f"(duplicates removed: {duplicates_removed})"
            )

    print(f"\nTotal duplicates removed across all outputs: {total_duplicates_removed}")

    # Write README with manual configs (no dataset_info sizes).
    readme_path = os.path.join(upload_root, "README.md")
    write_readme_with_configs(
        readme_path,
        configs=CONFIGS,
        splits=SPLITS,
    )

    # Create repo and upload.
    print(f"\nUploading to {hf_repo_id} (private={private})...")
    create_repo(hf_repo_id, repo_type="dataset", private=private, exist_ok=True)

    api = HfApi()
    api.upload_folder(
        folder_path=upload_root,
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )

    print(f"\nDone. Repo: https://huggingface.co/datasets/{hf_repo_id}")


if __name__ == "__main__":
    main()
