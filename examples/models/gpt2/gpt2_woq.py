# from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

# Initializes the model: Quantize model with weight only algorithms and replace the linear with itrex's qbits_linear kernel
model = BaseModel.create("gpt2_int8")

# Once the model has been quantized, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print(output)