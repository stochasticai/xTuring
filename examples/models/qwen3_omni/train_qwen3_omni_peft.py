"""Minimal HF/PEFT trainer scaffold for Qwen3-Omni-30B-A3B-Instruct.

Supports on-the-fly multimodal preprocessing using Qwen3OmniMoeProcessor.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Qwen3OmniMoeProcessor,
    Trainer,
    TrainingArguments,
)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - best-effort dependency check
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_any(path: str) -> Dataset:
    data = load_from_disk(path)
    if isinstance(data, DatasetDict):
        return data["train"]
    return data


def pad_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels_in = [item.get("labels") for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    padded_labels = []
    attention_mask = []

    for ids, lbl in zip(input_ids, labels_in):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

        if lbl is None:
            lbl = copy.deepcopy(ids)
        padded_labels.append(lbl + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def build_conversation(
    example: Dict[str, Any],
    system_prompt: str,
    audio_col: Optional[str],
    text_col: Optional[str],
    target_col: Optional[str],
    messages_col: Optional[str],
    include_assistant: bool,
) -> List[Dict[str, Any]]:
    if messages_col and messages_col in example:
        messages = example[messages_col]
        if isinstance(messages, list):
            return messages

    content: List[Dict[str, Any]] = []
    if audio_col and audio_col in example and example[audio_col]:
        content.append({"type": "audio", "path": example[audio_col]})
    if text_col and text_col in example and example[text_col]:
        content.append({"type": "text", "text": example[text_col]})

    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]

    if include_assistant and target_col and target_col in example and example[target_col]:
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example[target_col]}],
            }
        )

    return conversation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal PEFT trainer for Qwen3-Omni-30B-A3B-Instruct."
    )
    parser.add_argument(
        "--config",
        default="examples/models/qwen3_omni/peft_a100_h100.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    model_cfg = cfg["model"]
    peft_cfg = cfg["peft"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=(
            "flash_attention_2" if model_cfg.get("use_flash_attention_2") else None
        ),
    )

    lora = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        bias=peft_cfg["bias"],
        task_type=peft_cfg["task_type"],
        target_modules=peft_cfg["target_modules"],
    )
    model = get_peft_model(model, lora)

    dataset = load_dataset_any(data_cfg["train_dataset"])
    if data_cfg.get("train_split") and isinstance(dataset, DatasetDict):
        dataset = dataset[data_cfg["train_split"]]

    if data_cfg.get("max_samples", 0) and len(dataset) > data_cfg["max_samples"]:
        dataset = dataset.shuffle(seed=train_cfg.get("seed", 42)).select(
            range(data_cfg["max_samples"])
        )

    audio_col = data_cfg.get("audio_column")
    text_col = data_cfg.get("text_column")
    target_col = data_cfg.get("target_column")
    messages_col = data_cfg.get("messages_column")
    system_prompt = data_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    use_audio_in_video = data_cfg.get("use_audio_in_video", False)
    load_audio_from_video = data_cfg.get("load_audio_from_video", False)
    video_fps = data_cfg.get("video_fps", 2.0)

    def collate_fn(batch: List[Dict[str, Any]]):
        if "input_ids" in batch[0]:
            return pad_batch(batch, processor.tokenizer.pad_token_id)

        full_conversations = [
            build_conversation(
                ex,
                system_prompt=system_prompt,
                audio_col=audio_col,
                text_col=text_col,
                target_col=target_col,
                messages_col=messages_col,
                include_assistant=True,
            )
            for ex in batch
        ]
        prompt_conversations = [
            build_conversation(
                ex,
                system_prompt=system_prompt,
                audio_col=audio_col,
                text_col=text_col,
                target_col=target_col,
                messages_col=messages_col,
                include_assistant=False,
            )
            for ex in batch
        ]

        full_inputs = processor.apply_chat_template(
            full_conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
            load_audio_from_video=load_audio_from_video,
            video_fps=video_fps,
        )
        prompt_inputs = processor.apply_chat_template(
            prompt_conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            use_audio_in_video=use_audio_in_video,
            load_audio_from_video=load_audio_from_video,
            video_fps=video_fps,
        )

        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
        labels = full_inputs["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, : int(prompt_len)] = -100
        full_inputs["labels"] = labels

        return full_inputs

    training_args = TrainingArguments(**train_cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])

    meta_path = Path(train_cfg["output_dir"]) / "training_meta.json"
    meta = {"config": cfg, "num_rows": len(dataset)}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
