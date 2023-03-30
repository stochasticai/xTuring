from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models.base import BaseModel
from xturing.ui.playground import Playground

instruction_dataset = InstructionDataset("../llama/alpaca_data")

# Initializes the model
model = BaseModel.create("gpt2_lora")

# Finetuned the model
model.finetune(dataset=instruction_dataset)

# Model path
model_path = "./gpt2_weights"

# Save the model
model.save(model_path)

# launch the playground
Playground(model_path).launch()
