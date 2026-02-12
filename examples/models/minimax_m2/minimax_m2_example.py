"""
Example usage of MiniMaxM2 model with xTuring

This example demonstrates how to use the MiniMaxM2 model from HuggingFace
with the xTuring library for inference and fine-tuning.

Model: MiniMaxAI/MiniMax-M2
"""

from xturing.models import BaseModel

# Example 1: Create the base MiniMaxM2 model
print("Loading MiniMaxM2 model...")
model = BaseModel.create("minimax_m2")

# Generate text
output = model.generate(texts=["What is machine learning?"])
print("Base model output:")
print(output)

# Example 2: Use LoRA version for efficient fine-tuning
print("\nLoading MiniMaxM2 with LoRA...")
model_lora = BaseModel.create("minimax_m2_lora")

# You can also use INT8 quantized versions for memory efficiency
# model_int8 = BaseModel.create("minimax_m2_int8")
# model_lora_int8 = BaseModel.create("minimax_m2_lora_int8")
# model_lora_kbit = BaseModel.create("minimax_m2_lora_kbit")

print("MiniMaxM2 model loaded successfully!")
