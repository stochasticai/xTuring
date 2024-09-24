from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

instruction_dataset = InstructionDataset("./alpaca_data")

# Initialize the model
model = BaseModel.create("mixtral")

# Fine-tune the model
model.finetune(dataset=instruction_dataset)

# Once the model has been fine-tuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
