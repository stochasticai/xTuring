import gc

from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

instruction_dataset = InstructionDataset("./alpaca_data")
# Initializes the model
model = BaseModel.create("llama_lora_int8")
# Finetuned the model
model.finetune(dataset=instruction_dataset)

# Save the model
model.save("./llama_weights")

# Once the model has been finetuned, you can start doing inferences
del model
gc.collect()
model = BaseModel.load("./llama_weights")
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))

# If you want to load the model just do BaseModel.load("./llama_weights")
