from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models.base import BaseModel

prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Input:
{{text}}

### Response:
"""

instruction_dataset = InstructionDataset(
    "./alpaca_data", promt_template=prompt_template
)
# Initializes the model
model = BaseModel.create("llama_lora")
# Finetuned the model
model.finetune(dataset=instruction_dataset)
# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
# Save the model
model.save("./llama")

# If you want to load the model just do BaseModel.load("./llama")
