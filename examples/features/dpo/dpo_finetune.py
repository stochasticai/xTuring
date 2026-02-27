"""Minimal example showing how to align a model using Direct Preference
Optimization (DPO) with xTuring.

DPO fine-tunes a language model using pairs of preferred and dispreferred
responses so that the model learns to produce outputs that match human
preferences without requiring a separate reward model.
"""

from pathlib import Path

from xturing.datasets.preference_dataset import PreferenceDataset
from xturing.models import BaseModel

OUTPUT_DIR = Path(__file__).parent / "dpo_weights"


def main():
    # Build a small preference dataset. Each sample needs a prompt, a chosen
    # (preferred) response, and a rejected (dispreferred) response.
    preference_data = {
        "prompt": [
            "Explain quantum computing in simple terms.",
            "What is the capital of France?",
            "How do I make pasta?",
            "What causes rain?",
        ],
        "chosen": [
            "Quantum computing uses qubits that can be 0, 1, or both at once, "
            "letting it solve certain problems much faster than regular computers.",
            "The capital of France is Paris.",
            "Boil salted water, cook pasta until al dente, then drain and toss "
            "with your favorite sauce.",
            "Rain forms when water evaporates, rises, cools into clouds, and "
            "falls back as droplets when clouds become saturated.",
        ],
        "rejected": [
            "Quantum computing is basically magic computers that can do "
            "everything instantly.",
            "France doesn't have a capital, it's a collective.",
            "Just put some noodles in a microwave with ketchup.",
            "Rain happens because the sky is sad.",
        ],
    }

    dataset = PreferenceDataset(preference_data)

    # Initialise a model with a LoRA adapter. DPO works with any model
    # variant, but LoRA is recommended to keep memory usage low since DPO
    # requires a frozen reference model in addition to the policy model.
    model = BaseModel.create("qwen3_0_6b_lora")

    # Run DPO fine-tuning. The beta parameter controls how strongly the model
    # is penalised for deviating from the reference policy (higher = more
    # conservative).
    model.dpo_finetune(dataset=dataset, beta=0.1)

    # Verify the aligned model generates reasonable output.
    output = model.generate(texts=["Explain gravity in simple terms."])
    print(f"Generated output: {output}")

    # Save the fine-tuned adapter weights.
    model.save(str(OUTPUT_DIR))
    print(f"Saved DPO fine-tuned weights to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
