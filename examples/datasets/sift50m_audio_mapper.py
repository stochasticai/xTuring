"""Resolve SIFT-50M audio paths to local files.

Given a subset saved with `save_to_disk`, this script attaches a resolved
`audio_file` column based on `audio_path` and an optional `data_source`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk


def _parse_audio_roots(values) -> Dict[str, str]:
    roots: Dict[str, str] = {}
    for item in values or []:
        if "=" in item:
            key, path = item.split("=", 1)
            roots[key.strip()] = path.strip()
        else:
            # Allow a single default root via --audio-root /path
            roots[""] = item.strip()
    return roots


def _resolve_path(
    audio_path: Optional[str],
    data_source: Optional[str],
    roots: Dict[str, str],
    default_root: Optional[str],
) -> Optional[str]:
    if not audio_path:
        return None

    # Already absolute or explicitly rooted.
    if os.path.isabs(audio_path):
        return audio_path

    base = None
    if data_source and data_source in roots:
        base = roots[data_source]
    elif "" in roots:
        base = roots[""]
    elif default_root:
        base = default_root

    if not base:
        return None

    return str(Path(base) / audio_path)


def _as_jsonl(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_any(path: str) -> Dataset:
    data = load_from_disk(path)
    if isinstance(data, DatasetDict):
        return data["train"]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach resolved audio paths to a SIFT-50M subset."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to a dataset saved with save_to_disk.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the updated dataset.",
    )
    parser.add_argument(
        "--audio-path-col",
        default="audio_path",
        help="Column containing relative audio paths.",
    )
    parser.add_argument(
        "--data-source-col",
        default="data_source",
        help="Column containing data source (e.g., mls, cv15, vctk).",
    )
    parser.add_argument(
        "--resolved-audio-col",
        default="audio_file",
        help="Name of output column for resolved paths.",
    )
    parser.add_argument(
        "--audio-root",
        action="append",
        default=[],
        help="Root mapping like data_source=/path or just /path for default.",
    )
    parser.add_argument(
        "--default-audio-root",
        default=None,
        help="Fallback root if no mapping matches.",
    )
    parser.add_argument(
        "--verify-exists",
        action="store_true",
        help="Set missing files to null and optionally drop them.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows where resolved audio files do not exist.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also export subset.jsonl.",
    )

    args = parser.parse_args()

    roots = _parse_audio_roots(args.audio_root)
    dataset = load_any(args.input_dir)

    if args.audio_path_col not in dataset.column_names:
        raise ValueError(
            f"Missing audio path column '{args.audio_path_col}'. "
            f"Available: {dataset.column_names}"
        )

    if args.data_source_col not in dataset.column_names:
        # Not fatal; proceed with None data_source.
        data_source_col = None
    else:
        data_source_col = args.data_source_col

    def _map(batch):
        audio_paths = batch[args.audio_path_col]
        data_sources = (
            batch[data_source_col] if data_source_col else [None] * len(audio_paths)
        )

        resolved = []
        exists = []
        for ap, ds in zip(audio_paths, data_sources):
            path = _resolve_path(ap, ds, roots, args.default_audio_root)
            if path and args.verify_exists:
                ok = Path(path).exists()
            else:
                ok = True
            resolved.append(path if ok else None)
            exists.append(ok)

        return {args.resolved_audio_col: resolved, "_audio_exists": exists}

    mapped = dataset.map(_map, batched=True, desc="Resolve audio paths")

    if args.drop_missing:
        mapped = mapped.filter(
            lambda row: row["_audio_exists"] and row[args.resolved_audio_col] is not None,
            desc="Drop missing audio",
        )

    if "_audio_exists" in mapped.column_names:
        mapped = mapped.remove_columns(["_audio_exists"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapped.save_to_disk(str(output_dir))

    if args.jsonl:
        _as_jsonl(mapped, output_dir / "subset.jsonl")

    meta = {
        "input_dir": args.input_dir,
        "audio_path_col": args.audio_path_col,
        "data_source_col": args.data_source_col,
        "resolved_audio_col": args.resolved_audio_col,
        "audio_roots": roots,
        "default_audio_root": args.default_audio_root,
        "verify_exists": args.verify_exists,
        "drop_missing": args.drop_missing,
        "num_rows": len(mapped),
        "columns": mapped.column_names,
    }
    with (output_dir / "audio_map_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
