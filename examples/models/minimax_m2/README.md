# MiniMaxM2 Model Examples

This directory contains examples for using the MiniMaxM2 model from HuggingFace with xTuring.

## Model Information

- **Model**: MiniMaxAI/MiniMax-M2
- **Source**: [HuggingFace Model Hub](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## Available Variants

The MiniMaxM2 model is available in multiple configurations:

1. **minimax_m2** - Base model
2. **minimax_m2_lora** - LoRA fine-tuning enabled
3. **minimax_m2_int8** - 8-bit quantized version
4. **minimax_m2_lora_int8** - LoRA with 8-bit quantization
5. **minimax_m2_lora_kbit** - LoRA with 4-bit quantization

## Usage Examples

### Basic Inference

```python
from xturing.models import BaseModel

# Create the model
model = BaseModel.create("minimax_m2")

# Generate text
output = model.generate(texts=["What is machine learning?"])
print(output)
```

### Fine-tuning with LoRA

```python
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

# Load dataset
dataset = InstructionDataset("path/to/your/dataset")

# Create model with LoRA
model = BaseModel.create("minimax_m2_lora")

# Fine-tune
model.finetune(dataset=dataset)

# Save
model.save("./minimax_m2_finetuned")
```

### Memory-Efficient Inference

For machines with limited GPU memory, use quantized versions:

```python
from xturing.models import BaseModel

# Use 8-bit quantization
model = BaseModel.create("minimax_m2_int8")

# Or use 4-bit quantization with LoRA
model = BaseModel.create("minimax_m2_lora_kbit")

output = model.generate(texts=["Your prompt here"])
```

## Files

- `minimax_m2_example.py` - Basic usage example
- `minimax_m2_finetune.py` - Fine-tuning example
- `README.md` - This file

## Configuration

The model uses the following default settings:

### Generation Config
- `max_new_tokens`: 512
- `temperature`: 0.1
- `penalty_alpha`: 0.6 (for contrastive search)
- `top_k`: 4

### Fine-tuning Config
- `learning_rate`: 2e-4 (LoRA variants)
- `num_train_epochs`: 3
- `max_length`: 2048
- `batch_size`: Varies by variant

These can be customized through the configuration files or when creating the model.

## Requirements

Make sure you have xTuring installed with all dependencies:

```bash
pip install xturing
```

## Notes

- The model requires `trust_remote_code=True` to load properly
- LoRA variants are recommended for fine-tuning as they are more parameter-efficient
- Quantized versions (int8, kbit) require less memory but may have slightly reduced accuracy
