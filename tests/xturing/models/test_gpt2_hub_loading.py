import os

import pytest

from xturing import GPT2

RUN_HUB_LOADING_TEST = os.getenv("XTURING_RUN_HUB_LOADING_TEST") == "1"


@pytest.mark.skipif(
    not RUN_HUB_LOADING_TEST,
    reason="Requires network access and large external hub download",
)
def test_gpt_2_hub_loading():
    model = GPT2.load("x/distilgpt2_lora_finetuned_alpaca")
    model.generate(texts=["Why LLM models are becoming so important?"])
