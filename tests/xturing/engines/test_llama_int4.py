from xturing.models import LlamaLoraInt4
from xturing.datasets import TextDataset

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}


model = LlamaLoraInt4("/home/romanageev/peft/llama7b-4bit-128g.pt")
dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
# finetuning_config = model.finetuning_config()
# finetuning_config.num_train_epochs = 1
# model.finetune(dataset=dataset)
generation_config = model.generation_config()
generation_config.do_sample = False
generation_config.max_new_tokens = 20
generation_config.top_k = 50
generation_config.top_p = 1.0
result = model.generate(dataset=dataset)
print(result)