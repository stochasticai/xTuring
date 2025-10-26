"""Minimal example showing how to fine-tune Qwen3-0.6B with LoRA using xTuring."""
from pathlib import Path

from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

# Reuse the small Alpaca-style dataset that ships with the repo. Replace this path
# with your own instruction dataset when running real experiments.
DATASET_DIR = Path(__file__).parent.parent / "llama" / "alpaca_data"

# Location where the LoRA adapters will be stored once training finishes.
OUTPUT_DIR = Path(__file__).parent / "qwen3_lora_weights"


def main():
    instruction_dataset = InstructionDataset(str(DATASET_DIR))

    # Initialize the Qwen3 0.6B model with a LoRA adapter head.
    model = BaseModel.create("qwen3_0_6b_lora")

    # Launch fine-tuning with the default configuration (see
    # src/xturing/config/finetuning_config.yaml for the exact hyper-parameters).
    model.finetune(dataset=instruction_dataset)

    # Run a quick generation to sanity-check the adapter before saving.
    output = model.generate(texts=["Why are smaller language models becoming popular?"])
    print(f"Generated output: {output}")

    # Persist the adapter and tokenizer so the run can be resumed or deployed later.
    model.save(str(OUTPUT_DIR))
    print(f"Saved fine-tuned weights to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
