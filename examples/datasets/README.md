# Datasets

This folder includes dataset helpers and recipes used by xTuring examples.

## SIFT-50M helpers (English subsets)

### 1) Build a small English subset

Filters `amazon-agi/SIFT-50M` to English plus:
- `closed_ended_content_level`
- `open_ended`
- optional `controllable_generation`

```bash
python examples/datasets/sift50m_subset_builder.py \
  --output-dir ./data/sift50m_en_small \
  --max-examples 100000 \
  --include-controllable-generation \
  --jsonl
```

Notes:
- Use `--language-col` or `--category-col` if the dataset schema changes.
- Set `--max-examples 0` to keep all rows after filtering.

### 2) Resolve audio paths to local files

SIFT-50M includes `audio_path` and (often) `data_source`. This script adds a
resolved `audio_file` column and can drop rows with missing files.

```bash
python examples/datasets/sift50m_audio_mapper.py \
  --input-dir ./data/sift50m_en_small \
  --output-dir ./data/sift50m_en_small_mapped \
  --audio-root mls=/data/mls \
  --audio-root cv15=/data/commonvoice15 \
  --audio-root vctk=/data/vctk \
  --verify-exists \
  --drop-missing \
  --jsonl
```

If your dataset uses different columns:

```bash
python examples/datasets/sift50m_audio_mapper.py \
  --input-dir ./data/sift50m_en_small \
  --output-dir ./data/sift50m_en_small_mapped \
  --audio-path-col audio_path \
  --data-source-col data_source
```

## Outputs

Each script writes:
- a Hugging Face dataset directory (via `save_to_disk`)
- `subset.jsonl` (if `--jsonl` is set)
- a `*_meta.json` file with the filter settings and detected columns
