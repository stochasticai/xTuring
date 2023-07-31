from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import GenericLoraModel

instruction_dataset = InstructionDataset("../../models/llama/alpaca_data")
# Initializes the model
model = GenericLoraModel("facebook/opt-1.3b", target_modules=["q_proj", "v_proj"])
# Finetuned the model
model.finetune(dataset=instruction_dataset)
# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
# Save the model
model.save("./generic_lora_weights")

# If you want to load the model just do BaseModel.load("./llama_weights")
