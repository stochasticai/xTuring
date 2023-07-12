import tempfile
from pathlib import Path

from xturing.datasets import TextDataset
from xturing.engines import GPT2LoraEngine
from xturing.models import GPT2, BaseModel

EXAMPLE_BASE_MODEL = "I want to be a part of the community"
EXAMPLE_LORA_MODEL = "I want to be a part of the community"

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}

model = BaseModel.create("distilgpt2")


def test_text_gpt2():
    # Greedy search. Parameters are set to default config of HF
    generation_config = model.generation_config()
    generation_config.do_sample = False
    generation_config.max_new_tokens = None
    generation_config.top_k = 50
    generation_config.top_p = 1.0

    assert model.generate(texts="I want to") != ""


def test_text_dataset_gpt2():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    result = model.generate(dataset=dataset)
    assert len(result) == 2


def test_text_dataset_gpt2_lora():
    # Greedy search. Parameters are set to default config of HF
    other_model = BaseModel.create("distilgpt2_lora")
    generation_config = other_model.generation_config()
    generation_config.do_sample = False
    generation_config.max_new_tokens = None
    generation_config.top_k = 50
    generation_config.top_p = 1.0
    assert other_model.generate(texts="I want to") != ""


def test_text_dataset_gpt2_lora():
    # Greedy search. Parameters are set to default config of HF
    other_model = BaseModel.create("gpt2_lora_int8")
    generation_config = other_model.generation_config()
    generation_config.do_sample = False
    generation_config.max_new_tokens = None
    generation_config.top_k = 50
    generation_config.top_p = 1.0
    assert other_model.generate(texts="I want to") != ""


def test_train_gpt2():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    model = BaseModel.create("distilgpt2")
    finetuning_config = model.finetuning_config()
    finetuning_config.num_train_epochs = 1
    model.finetune(dataset=dataset)
    generation_config = model.generation_config()
    generation_config.do_sample = False
    generation_config.max_new_tokens = None
    generation_config.top_k = 50
    generation_config.top_p = 1.0
    result = model.generate(dataset=dataset)
    assert len(result) == 2


def test_train_gpt2_lora():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    model = BaseModel.create("distilgpt2_lora")
    finetuning_config = model.finetuning_config()
    finetuning_config.num_train_epochs = 1
    model.finetune(dataset=dataset)
    generation_config = model.generation_config()
    generation_config.do_sample = False
    generation_config.max_new_tokens = None
    generation_config.top_k = 50
    generation_config.top_p = 1.0
    result = model.generate(dataset=dataset)
    assert len(result) == 2


def test_saving_loading_model():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing"
    model = BaseModel.create("distilgpt2")
    model.save(str(saving_path))

    model2 = BaseModel.load(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])


def test_saving_loading_model_lora():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing_lora"
    model = BaseModel.create("distilgpt2_lora")
    model.save(str(saving_path))

    model2 = BaseModel.load(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])
