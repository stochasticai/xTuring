from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

instruction_dataset = InstructionDataset("../llama/alpaca_data")
# Initializes the model
model = BaseModel.create("gpt2_lora")
# Finetuned the model
model.finetune(dataset=instruction_dataset)
# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
# Save the model
model.save("./gpt2_weights")

# If you want to load the model just do BaseModel.load("./gpt2_weights")
