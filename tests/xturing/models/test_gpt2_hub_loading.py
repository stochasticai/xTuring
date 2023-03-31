from xturing import GPT2
from xturing.models import DistilGPT2


def test_gpt_2_hub_loading():
    model = GPT2.load("x/gpt2")
    model.generate(texts=["Why LLM models are becoming so important?"])
