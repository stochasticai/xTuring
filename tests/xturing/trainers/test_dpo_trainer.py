import importlib.machinery
import sys
import types
from pathlib import Path

import torch


def _make_module(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def _install_stub_modules():
    if "cohere" not in sys.modules:
        cohere_module = _make_module("cohere")

        class _CohereError(Exception):
            pass

        class _Client:
            def __init__(self, *_args, **_kwargs):
                self.generations = [types.SimpleNamespace(text="")]

            def generate(self, **_):
                return types.SimpleNamespace(generations=self.generations)

        cohere_module.CohereError = _CohereError
        cohere_module.Client = _Client
        sys.modules["cohere"] = cohere_module

    if "openai" not in sys.modules:
        openai_module = _make_module("openai")

        class _Completion:
            @staticmethod
            def create(n=1, **_):
                return {"choices": [types.SimpleNamespace(text="")] * n}

        class _ChatCompletion:
            @staticmethod
            def create(**_):
                return {"choices": [{"message": {"content": ""}}]}

        openai_module.api_key = None
        openai_module.organization = None
        openai_module.Completion = _Completion
        openai_module.ChatCompletion = _ChatCompletion
        openai_module.error = types.SimpleNamespace(OpenAIError=Exception)
        sys.modules["openai"] = openai_module

    if "anthropic" not in sys.modules:
        anthropic_module = _make_module("anthropic")

        class _Messages:
            def create(self, **_):
                content_block = types.SimpleNamespace(type="text", text="")
                return types.SimpleNamespace(
                    content=[content_block], stop_reason="stop"
                )

        class _Anthropic:
            def __init__(self, *_args, **_kwargs):
                self.messages = _Messages()

        anthropic_module.Anthropic = _Anthropic
        anthropic_module.APIError = Exception
        anthropic_module.APIConnectionError = Exception
        anthropic_module.RateLimitError = Exception
        sys.modules["anthropic"] = anthropic_module

    if "xturing" not in sys.modules:
        xturing_module = _make_module("xturing")
        xturing_module.__path__ = [
            str(Path(__file__).resolve().parents[3] / "src" / "xturing")
        ]
        sys.modules["xturing"] = xturing_module

    if "deepspeed" not in sys.modules:
        deepspeed_module = _make_module("deepspeed")
        ops_module = _make_module("deepspeed.ops")
        adam_module = _make_module("deepspeed.ops.adam")

        class _DeepSpeedCPUAdam:
            def __init__(self, *_, **__):
                pass

        adam_module.DeepSpeedCPUAdam = _DeepSpeedCPUAdam
        sys.modules["deepspeed"] = deepspeed_module
        sys.modules["deepspeed.ops"] = ops_module
        sys.modules["deepspeed.ops.adam"] = adam_module


_install_stub_modules()

from xturing.datasets.preference_dataset import PreferenceDataset
from xturing.engines.base import BaseEngine
from xturing.engines.qwen_engine import Qwen3LoraEngine
from xturing.models import BaseModel
from xturing.preprocessors.base import BasePreprocessor
from xturing.trainers.base import BaseTrainer
from xturing.trainers.dpo_trainer import DPOTrainer, compute_logprobs, dpo_loss


def test_dpo_trainer_registered():
    assert "dpo_trainer" in BaseTrainer.registry
    assert BaseTrainer.registry["dpo_trainer"] is DPOTrainer


def test_preference_dataset_registered():
    from xturing.datasets.base import BaseDataset

    assert "preference_dataset" in BaseDataset.registry


def test_preference_collator_registered():
    assert "preference_dataset" in BasePreprocessor.registry


def test_preference_dataset_from_dict():
    data = {
        "prompt": ["What is AI?", "Explain gravity."],
        "chosen": ["AI is a field of computer science.", "Gravity is a force."],
        "rejected": ["AI is magic.", "Gravity is fake."],
    }
    dataset = PreferenceDataset(data)
    assert len(dataset) == 2
    assert dataset[0]["prompt"] == "What is AI?"
    assert dataset[0]["chosen"] == "AI is a field of computer science."
    assert dataset[0]["rejected"] == "AI is magic."
    assert dataset.config_name == "preference_dataset"


def test_preference_dataset_validates_columns():
    import pytest

    # Missing 'rejected' column
    data = {
        "prompt": ["What is AI?"],
        "chosen": ["AI is a field."],
    }
    with pytest.raises(Exception):
        PreferenceDataset(data)


def test_compute_logprobs():
    """Verify compute_logprobs returns one scalar per sample."""
    batch_size, seq_len, vocab_size = 2, 8, 32

    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def __call__(self, input_ids, attention_mask):
            return DummyOutput(torch.randn(batch_size, seq_len, vocab_size))

    model = DummyModel()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    # Mask first 4 tokens as prompt
    labels[:, :4] = -100

    logprobs = compute_logprobs(model, input_ids, attention_mask, labels)
    assert logprobs.shape == (batch_size,)
    # Log-probs should be negative
    assert (logprobs <= 0).all()


def test_dpo_loss_basic():
    """Verify DPO loss is a positive scalar and rewards have correct sign."""
    batch_size = 4
    # Chosen should have higher log-probs under the policy
    policy_chosen = torch.tensor([-1.0] * batch_size)
    policy_rejected = torch.tensor([-3.0] * batch_size)
    ref_chosen = torch.tensor([-2.0] * batch_size)
    ref_rejected = torch.tensor([-2.0] * batch_size)

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    # Chosen rewards should be higher than rejected rewards
    assert (chosen_rewards > rejected_rewards).all()


def test_dpo_finetune_end_to_end(monkeypatch):
    """End-to-end test: create model, create preference dataset, call dpo_finetune."""

    class DummyTokenizer:
        eos_token_id = 0
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, _):
            return {"input_ids": [0], "attention_mask": [1]}

        def pad(self, samples, padding=True, max_length=None, return_tensors=None):
            batch_size = len(samples)
            return {
                "input_ids": torch.tensor([[0] for _ in range(batch_size)]),
                "attention_mask": torch.tensor([[1] for _ in range(batch_size)]),
            }

    class DummyModel:
        def to(self, *_):
            return self

        def eval(self):
            return self

        def train(self):
            return self

        def parameters(self):
            return [torch.tensor([1.0], requires_grad=True)]

        def named_parameters(self):
            yield "dummy", torch.tensor([1.0], requires_grad=True)

    class DummyEngine:
        def __init__(self, *_, **__):
            self.model = DummyModel()
            self.tokenizer = DummyTokenizer()

        def save(self, *_):
            return None

    class DummyCollator:
        def __init__(self, *_, **__):
            pass

        def __call__(self, batches):
            batch_size = len(batches)
            return {
                "chosen_input_ids": torch.tensor([[0] for _ in range(batch_size)]),
                "chosen_attention_mask": torch.tensor([[1] for _ in range(batch_size)]),
                "chosen_labels": torch.tensor([[0] for _ in range(batch_size)]),
                "rejected_input_ids": torch.tensor([[0] for _ in range(batch_size)]),
                "rejected_attention_mask": torch.tensor(
                    [[1] for _ in range(batch_size)]
                ),
                "rejected_labels": torch.tensor([[0] for _ in range(batch_size)]),
            }

    trainers = []

    class MockDPOTrainer:
        def __init__(
            self,
            engine,
            dataset,
            collate_fn,
            num_epochs,
            batch_size,
            learning_rate,
            optimizer_name,
            beta=0.1,
            logger=True,
        ):
            self.engine = engine
            self.dataset = dataset
            self.collate_fn = collate_fn
            self.beta = beta
            self.fit_called = False
            trainers.append(self)

        def fit(self):
            self.fit_called = True
            batch = self.collate_fn([self.dataset[0]])
            assert "chosen_input_ids" in batch
            assert "rejected_input_ids" in batch

    monkeypatch.setitem(BaseEngine.registry, Qwen3LoraEngine.config_name, DummyEngine)
    monkeypatch.setitem(BasePreprocessor.registry, "preference_dataset", DummyCollator)
    monkeypatch.setitem(BaseTrainer.registry, DPOTrainer.config_name, MockDPOTrainer)

    dataset = PreferenceDataset(
        {
            "prompt": [
                "What is machine learning?",
                "Explain photosynthesis.",
            ],
            "chosen": [
                "Machine learning is a subset of AI focused on learning from data.",
                "Photosynthesis is the process by which plants convert sunlight.",
            ],
            "rejected": [
                "Machine learning is when computers become sentient.",
                "Photosynthesis is when plants eat dirt.",
            ],
        }
    )

    model = BaseModel.create("qwen3_0_6b_lora")
    model.dpo_finetune(dataset=dataset, beta=0.1)

    assert trainers and trainers[0].fit_called
    assert trainers[0].beta == 0.1
