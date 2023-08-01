# Navigating through examples
Here, is a brief about how to navigate through examples quick and efficiently, and get your hands dirty with `xTuring`. 

## Directory structure
```
examples/
    | datasets
    | features/
        | dataset_generation/
        | evaluation/
        | generic/
        | int4_finetuning/
    | models/
    | playground_ui/
```

### datsets/
This directory consists of multiple ways to generate your custom dataset from a given set of examples. 

### features/
This directory consists of files with exapmles highlighting speific major features of the library, which can be replicated to any LLM you want.  
For example, in `dataset_generation/`, you will find an example on how to generate your custom dataset from a .jsonl file. In `evaluation/`, you will find a specific exapmle on how to evaluate your finetuned model, which can then be extended to any LLM and any dataset. 

### models/
This directory consists of examples specific to each model mentioned. 

### playground_ui/
This directory consists of an example which demonstrates how you can play around with your LLM through a web interface.

## Models
Below is a list of all the supported models via `BaseModel` class of `xTuring` and their corresponding keys to load them.

|  Model |  Key |
| -- | -- |
|Bloom | bloom|
|Cerebras | cerebras|
|DistilGPT-2 | distilgpt2|
|Falcon-7B | falcon|
|Galactica | galactica|
|GPT-J | gptj|
|GPT-2 | gpt2|
|LlaMA | llama|
|LlaMA2 | llama2|
|OPT-1.3B | opt|

The above mentioned are the base variants of the LLMs. Below are the templates to get their `LoRA`, `INT8`, `INT8 + LoRA` and `INT4 + LoRA` versions.

| Version | Template |
| -- | -- |
| LoRA|  <model_key>_lora|
| INT8|  <model_key>_int8|
| INT8 + LoRA|  <model_key>_lora_int8|

** In order to load any model's __`INT4+LoRA`__ version, you will need to make use of `GenericLoraKbitModel` class from `xturing.models`. Below is how to use it:
```python
model = GenericLoraKbitModel('<model_path>')
```
The `model_path` can be replaced with you local directory or any HuggingFace library model like `facebook/opt-1.3b`.
