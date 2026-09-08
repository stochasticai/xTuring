# Qwen3-Omni (HF/PEFT template)

This folder contains a minimal training config template for
`Qwen/Qwen3-Omni-30B-A3B-Instruct` intended for A100/H100 class GPUs.

## Files
- `examples/models/qwen3_omni/peft_a100_h100.yaml`: LoRA template config.
- `examples/models/qwen3_omni/train_qwen3_omni_peft.py`: minimal HF/PEFT trainer scaffold.

## Notes
- This repo does not include a Qwen3-Omni fine-tuning engine; use HF/PEFT on your
  cloud machine.
- The model expects **multimodal message-format inputs**. Your training script
  should transform each row into the required chat format and attach audio.
- Start small: 50k-200k examples with LoRA and gradient checkpointing.

## Suggested usage
Use this YAML as a base config for your own training script. For example, load
it and pass the values into `transformers.TrainingArguments` and `peft.LoraConfig`.

## Usage
The training script supports on-the-fly multimodal preprocessing using
`Qwen3OmniMoeProcessor`. Provide the dataset columns in
`examples/models/qwen3_omni/peft_a100_h100.yaml`:
- `audio_column` (path to audio)
- `text_column` (user instruction)
- `target_column` (assistant text response)
- `messages_column` (optional, if your dataset already stores full chat messages)

## Smoke test
Quickly validate the processor with a single audio file:
```bash
python examples/models/qwen3_omni/smoke_test_processor.py \
  --audio /path/to/example.wav \
  --text \"Please summarize the audio.\"
```

You can generate a quick dummy WAV:
```bash
python examples/models/qwen3_omni/generate_dummy_wav.py \
  --output /tmp/dummy_tone.wav
```
