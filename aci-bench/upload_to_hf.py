import os
import json
import argparse
import csv
import re
import unicodedata
from typing import Dict, List, Tuple

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


def load_csv_items(filepath: str, *, ascii_normalize: bool) -> List[Tuple[str, Dict[str, str]]]:
    """
    Returns a list of (dataset_name, record) pairs.
    The dataset_name is used only to route into Hub configs.
    The record EXCLUDES the dataset column (because config/subset is the dataset).
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
            dataset_name = normalize_dataset_name(dataset_value)

            record = {
                "encounter_id": normalize_text(row.get("encounter_id", ""), ascii_normalize=ascii_normalize),
                "dialogue": normalize_text(row.get("dialogue", ""), ascii_normalize=ascii_normalize),
                "note": normalize_text(row.get("note", ""), ascii_normalize=ascii_normalize),
            }
            items.append((dataset_name, record))

    print(f"Loaded {len(items)} examples from {filepath}")
    return items


def filter_records(items: List[Tuple[str, Dict[str, str]]], config_name: str) -> List[Dict[str, str]]:
    want = normalize_dataset_name(config_name)
    return [rec for ds, rec in items if ds == want]


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
            "dialogue": Value("string"),
            "note": Value("string"),
        }
    )

    # Load all raw rows once.
    split_to_items: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for split, filename in split_to_file.items():
        csv_path = os.path.join(data_dir, filename)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Missing required split file: {csv_path}. Make sure your CSVs are in --data_dir."
            )
        split_to_items[split] = load_csv_items(csv_path, ascii_normalize=ascii_normalize)

    # Write Parquet files: <config>/<split>.parquet
    for cfg in CONFIGS:
        cfg_dir = os.path.join(upload_root, cfg)
        os.makedirs(cfg_dir, exist_ok=True)

        for split in SPLITS:
            records = filter_records(split_to_items[split], cfg)
            ds = Dataset.from_list(records, features=features)

            out_parquet = os.path.join(cfg_dir, f"{split}.parquet")
            ds.to_parquet(out_parquet)

            # Optional: dump a small JSON sample for debugging locally only
            sample_path = os.path.join(work_dir, f"sample-{cfg}-{split}.json")
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(records[:5], f, ensure_ascii=False, indent=2)

            print(f"Wrote {cfg}/{split}.parquet with {ds.num_rows} rows")

    # Write README with manual configs (no dataset_info sizes).
    readme_path = os.path.join(upload_root, "README.md")
    write_readme_with_configs(
        readme_path,
        repo_title=hf_repo_name,
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
