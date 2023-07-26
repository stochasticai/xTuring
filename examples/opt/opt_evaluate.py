from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

instruction_dataset = InstructionDataset("../examples/llama/alpaca_data")
# Initializes the model
model = BaseModel.create("opt")
# Call the evaluate function
perplexity = model.evaluate(instruction_dataset, batch_size=5)

print(perplexity)
