"""
Fine-tuning MiniMaxM2 model with xTuring

This example shows how to fine-tune the MiniMaxM2 model using LoRA
for parameter-efficient training.
"""

from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

# Load your instruction dataset
# You can use your own dataset or create one following the xTuring format
instruction_dataset = InstructionDataset("path/to/your/dataset")

# Initialize the model with LoRA for efficient fine-tuning
model = BaseModel.create("minimax_m2_lora")

# Fine-tune the model
print("Starting fine-tuning...")
model.finetune(dataset=instruction_dataset)

# Save the fine-tuned model
model.save("./minimax_m2_finetuned")

# Load and use the fine-tuned model
print("Loading fine-tuned model...")
finetuned_model = BaseModel.load("./minimax_m2_finetuned")

# Generate with the fine-tuned model
output = finetuned_model.generate(texts=["Your prompt here"])
print(output)
