from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

instruction_dataset = InstructionDataset("../llama/alpaca_data")
# Initializes the model: Quantize model with weight only algorithms and replace the linear with itrex's qbits_linear kernel
from intel_extension_for_transformers.transformers import (
    WeightOnlyQuantConfig) 
woq_config = WeightOnlyQuantConfig(weight_dtype='int8')
model = BaseModel.create("gpt2", quantization_config=woq_config)

# there is no need to fine tuning model, as we are use weight-only quantization
# model.finetune(dataset=instruction_dataset)
# Once the model has been quantized, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
# Save the model
model.save("./gpt2_woq_weights")
