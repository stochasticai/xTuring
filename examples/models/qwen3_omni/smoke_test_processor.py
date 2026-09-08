"""Smoke test for Qwen3OmniMoeProcessor on a single audio file.

This validates that the processor can build model inputs from audio + text
without running a full training loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import Qwen3OmniMoeProcessor

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Qwen3-Omni processor")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HF model name or path.",
    )
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument(
        "--text",
        default="Please summarize the audio.",
        help="Instruction text.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt text.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model, trust_remote_code=True
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": args.system}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
                {"type": "text", "text": args.text},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        [conversation],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    print("input_ids shape:", tuple(inputs["input_ids"].shape))
    print("attention_mask shape:", tuple(inputs["attention_mask"].shape))


if __name__ == "__main__":
    main()
