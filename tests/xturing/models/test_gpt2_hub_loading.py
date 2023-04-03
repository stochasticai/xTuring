from xturing import GPT2, BaseModel
from xturing.models import DistilGPT2


def test_gpt_2_hub_loading():
    model = GPT2.load("x/distilgpt2_lora_finetuned_alpaca")
    model.generate(texts=["Why LLM models are becoming so important?"])
