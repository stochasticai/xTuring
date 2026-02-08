"""Build a small English subset of SIFT-50M.

This script downloads metadata from Hugging Face, filters to English and the
specified categories, and saves the resulting dataset locally.

It is designed to be robust to minor schema changes by allowing explicit
column selection and by auto-detecting likely column names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from datasets import Dataset, load_dataset


LANG_COL_CANDIDATES = ["language", "lang", "locale", "language_code", "lang_code"]
CATEGORY_COL_CANDIDATES = [
    "category",
    "task_category",
    "instruction_category",
    "task_type",
    "type",
]


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_set = set(cols)
    for name in candidates:
        if name in cols_set:
            return name
    return None


def _infer_column(name: str, cols: Iterable[str], candidates: Iterable[str]) -> str:
    inferred = _first_existing(cols, candidates)
    if inferred is None:
        raise ValueError(
            f"Could not infer {name} column. Available columns: {sorted(cols)}"
        )
    return inferred


def _as_jsonl(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_subset(
    output_dir: Path,
    max_examples: int,
    seed: int,
    include_controllable_generation: bool,
    language: str,
    language_col: Optional[str],
    category_col: Optional[str],
    jsonl: bool,
    dataset_id: str,
    split: str,
) -> None:
    dataset = load_dataset(dataset_id, split=split)

    cols = dataset.column_names
    lang_col = language_col or _infer_column("language", cols, LANG_COL_CANDIDATES)
    cat_col = category_col or _infer_column("category", cols, CATEGORY_COL_CANDIDATES)

    categories = ["closed_ended_content_level", "open_ended"]
    if include_controllable_generation:
        categories.append("controllable_generation")

    def _lang_filter(example):
        value = example.get(lang_col)
        if value is None:
            return False
        return str(value).lower().startswith(language.lower())

    def _category_filter(example):
        value = example.get(cat_col)
        if value is None:
            return False
        return value in categories

    filtered = dataset.filter(_lang_filter, desc="Filter language")
    filtered = filtered.filter(_category_filter, desc="Filter categories")

    if max_examples > 0 and len(filtered) > max_examples:
        filtered = filtered.shuffle(seed=seed).select(range(max_examples))

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(str(output_dir))

    if jsonl:
        _as_jsonl(filtered, output_dir / "subset.jsonl")

    meta = {
        "dataset_id": dataset_id,
        "split": split,
        "language_col": lang_col,
        "category_col": cat_col,
        "language": language,
        "categories": categories,
        "max_examples": max_examples,
        "seed": seed,
        "num_rows": len(filtered),
        "columns": filtered.column_names,
    }
    with (output_dir / "subset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small English subset of SIFT-50M."
    )
    parser.add_argument(
        "--output-dir",
        default="./data/sift50m_en_small",
        help="Where to write the subset dataset.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100_000,
        help="Max number of rows to keep (0 means keep all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before sampling.",
    )
    parser.add_argument(
        "--include-controllable-generation",
        action="store_true",
        help="Include controllable_generation category.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language prefix to match (default: en).",
    )
    parser.add_argument(
        "--language-col",
        default=None,
        help="Override language column name.",
    )
    parser.add_argument(
        "--category-col",
        default=None,
        help="Override category column name.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also export subset.jsonl.",
    )
    parser.add_argument(
        "--dataset-id",
        default="amazon-agi/SIFT-50M",
        help="HF dataset id.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF split to load.",
    )

    args = parser.parse_args()

    build_subset(
        output_dir=Path(args.output_dir),
        max_examples=args.max_examples,
        seed=args.seed,
        include_controllable_generation=args.include_controllable_generation,
        language=args.language,
        language_col=args.language_col,
        category_col=args.category_col,
        jsonl=args.jsonl,
        dataset_id=args.dataset_id,
        split=args.split,
    )


if __name__ == "__main__":
    main()
