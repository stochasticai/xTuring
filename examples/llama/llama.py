from turing.datasets.instruction_dataset import InstructionDataset
from turing.models.base import BaseModel

instruction_dataset = InstructionDataset("./alpaca_data")
# Initializes the model
model = BaseModel.create("llama")
# Finetuned the model
model.finetune(dataset=instruction_dataset)
# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))