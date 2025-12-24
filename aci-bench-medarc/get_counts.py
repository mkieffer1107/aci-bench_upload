#!/usr/bin/env python3
"""
Count rows by (subset/config) × transcript_version, with split breakdowns AND
informative column/row totals.

Usage:
  pip install -U datasets pandas
  python count_aci_bench_medarc_counts.py
  python count_aci_bench_medarc_counts.py --out counts_long.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without extra deps."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    s = df.astype(object).where(pd.notnull(df), "").astype(str)

    headers = list(s.columns)
    rows = s.values.tolist()

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [fmt_row(headers), sep]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)


def add_grand_total_row(df: pd.DataFrame, label_cols: list[str], label_vals: list[str]) -> pd.DataFrame:
    """Append a final row that sums all numeric columns; set label columns to label_vals."""
    out = df.copy()
    numeric_cols = [c for c in out.columns if c not in label_cols]
    totals = out[numeric_cols].sum(numeric_only=True)
    row = {**{k: v for k, v in zip(label_cols, label_vals)}, **totals.to_dict()}
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mkieffer/ACI-Bench-MedARC")
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="Write long-form counts CSV (subset, split, transcript_version, count).")
    args = ap.parse_args()

    dataset_id: str = args.dataset

    subsets = args.subsets if args.subsets else get_dataset_config_names(dataset_id)

    rows: list[dict] = []
    for subset in subsets:
        dsd = load_dataset(dataset_id, subset)  # DatasetDict of splits for this subset
        splits = args.splits if args.splits else list(dsd.keys())

        for split in splits:
            if split not in dsd:
                continue

            ds = dsd[split]
            if "transcript_version" not in ds.column_names:
                raise KeyError(
                    f"{dataset_id}/{subset}/{split} missing 'transcript_version'. Columns: {ds.column_names}"
                )

            counts = Counter(ds["transcript_version"])
            for tv, cnt in counts.items():
                tv_str = "<missing>" if tv is None else str(tv)
                rows.append({"subset": subset, "split": split, "transcript_version": tv_str, "count": int(cnt)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows counted. Check dataset/subsets/splits.")

    # Nice ordering (if applicable)
    subset_order = ["aci", "virtassist", "virtscribe"]
    tv_order = ["asr", "asrcorr", "humantrans"]

    if set(df["subset"]).issubset(set(subset_order)):
        df["subset"] = pd.Categorical(df["subset"], categories=subset_order, ordered=True)

    # transcript versions might include unexpected values; keep known ones first
    tv_cats = [t for t in tv_order if t in set(df["transcript_version"])]
    extras = sorted(set(df["transcript_version"]) - set(tv_cats))
    df["transcript_version"] = pd.Categorical(df["transcript_version"], categories=tv_cats + extras, ordered=True)

    df = df.sort_values(["subset", "transcript_version", "split"]).reset_index(drop=True)

    canonical_splits = ["train", "valid", "test1", "test2", "test3"]

    def ordered_split_cols(cols):
        cols = list(cols)
        return [c for c in canonical_splits if c in cols] + [c for c in cols if c not in canonical_splits]

    # -------------------------
    # 1) Overall (all splits combined) — your requested table shape
    # -------------------------
    overall = (
        df.groupby(["subset", "transcript_version"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "Count"})
        .sort_values(["subset", "transcript_version"])
        .reset_index(drop=True)
    )

    # Add informative totals:
    # - per subset total (across transcript versions)
    subset_totals = overall.groupby("subset", as_index=False)["Count"].sum()
    subset_totals["transcript_version"] = "ALL"
    overall_with_totals = pd.concat([overall, subset_totals], ignore_index=True)
    overall_with_totals = overall_with_totals.sort_values(["subset", "transcript_version"]).reset_index(drop=True)
    overall_with_totals = add_grand_total_row(
        overall_with_totals, label_cols=["subset", "transcript_version"], label_vals=["ALL", "ALL"]
    )

    print("\n## Overall counts (all splits combined) + totals\n")
    print(to_markdown(overall_with_totals))

    # -------------------------
    # 2) Breakdown by split for (subset, transcript_version)
    #    + column totals row (sums each split column) + row totals col
    # -------------------------
    pivot = (
        df.pivot_table(
            index=["subset", "transcript_version"],
            columns="split",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    split_cols = ordered_split_cols([c for c in pivot.columns if c not in ("subset", "transcript_version")])
    pivot = pivot[["subset", "transcript_version", *split_cols]]
    pivot["Total"] = pivot[split_cols].sum(axis=1)

    pivot = add_grand_total_row(pivot, label_cols=["subset", "transcript_version"], label_vals=["ALL", "ALL"])

    print("\n## Breakdown by split (subset × transcript_version) + column totals\n")
    print(to_markdown(pivot))

    # -------------------------
    # 3) Extra “informative sums”: totals by subset and by transcript_version (with split columns)
    # -------------------------
    by_subset = (
        df.pivot_table(index="subset", columns="split", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    split_cols2 = ordered_split_cols([c for c in by_subset.columns if c != "subset"])
    by_subset = by_subset[["subset", *split_cols2]]
    by_subset["Total"] = by_subset[split_cols2].sum(axis=1)
    by_subset = add_grand_total_row(by_subset, label_cols=["subset"], label_vals=["ALL"])

    print("\n## Totals by subset (sums across transcript versions) + column totals\n")
    print(to_markdown(by_subset))

    by_tv = (
        df.pivot_table(index="transcript_version", columns="split", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    split_cols3 = ordered_split_cols([c for c in by_tv.columns if c != "transcript_version"])
    by_tv = by_tv[["transcript_version", *split_cols3]]
    by_tv["Total"] = by_tv[split_cols3].sum(axis=1)
    by_tv = add_grand_total_row(by_tv, label_cols=["transcript_version"], label_vals=["ALL"])

    print("\n## Totals by transcript_version (sums across subsets) + column totals\n")
    print(to_markdown(by_tv))

    # -------------------------
    # Optional CSV (long form)
    # -------------------------
    if args.out:
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"\nWrote long-form counts to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
