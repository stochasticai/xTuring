import importlib.machinery
import sys
import types
from pathlib import Path


def _make_module(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def _install_stub_modules():
    if "ai21" not in sys.modules:
        ai21_module = _make_module("ai21")

        class _Completion:
            @staticmethod
            def execute(**_):
                return {"prompt": {"text": ""}}

        ai21_module.api_key = None
        ai21_module.Completion = _Completion
        sys.modules["ai21"] = ai21_module

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

from xturing.config.read_config import read_yaml
from xturing.engines.base import BaseEngine
from xturing.engines.qwen_engine import (
    Qwen3Engine,
    Qwen3Int8Engine,
    Qwen3LoraEngine,
    Qwen3LoraInt8Engine,
    Qwen3LoraKbitEngine,
)
from xturing.models import BaseModel
from xturing.models.qwen import (
    Qwen3,
    Qwen3Int8,
    Qwen3Lora,
    Qwen3LoraInt8,
    Qwen3LoraKbit,
)
from xturing.preprocessors.base import BasePreprocessor
from xturing.trainers.base import BaseTrainer
from xturing.trainers.lightning_trainer import LightningTrainer


def test_qwen3_model_registry_entries_present():
    model_names = [
        "qwen3_0_6b",
        "qwen3_0_6b_lora",
        "qwen3_0_6b_int8",
        "qwen3_0_6b_lora_int8",
        "qwen3_0_6b_lora_kbit",
    ]

    for model_name in model_names:
        assert model_name in BaseModel.registry
        assert BaseModel.registry[model_name] is not None


def test_qwen3_model_class_config_names():
    assert Qwen3.config_name == "qwen3_0_6b"
    assert Qwen3Lora.config_name == "qwen3_0_6b_lora"
    assert Qwen3Int8.config_name == "qwen3_0_6b_int8"
    assert Qwen3LoraInt8.config_name == "qwen3_0_6b_lora_int8"
    assert Qwen3LoraKbit.config_name == "qwen3_0_6b_lora_kbit"


def test_qwen3_engine_class_config_names():
    assert Qwen3Engine.config_name == "qwen3_0_6b_engine"
    assert Qwen3LoraEngine.config_name == "qwen3_0_6b_lora_engine"
    assert Qwen3Int8Engine.config_name == "qwen3_0_6b_int8_engine"
    assert Qwen3LoraInt8Engine.config_name == "qwen3_0_6b_lora_int8_engine"
    assert Qwen3LoraKbitEngine.config_name == "qwen3_0_6b_lora_kbit_engine"


def test_qwen3_config_entries_exist():
    config_dir = (
        Path(__file__).parent.parent.parent.parent / "src" / "xturing" / "config"
    )

    generation_config = read_yaml(str(config_dir / "generation_config.yaml"))
    assert "qwen3_0_6b" in generation_config
    assert "qwen3_0_6b_lora" in generation_config
    assert "qwen3_0_6b_int8" in generation_config
    assert "qwen3_0_6b_lora_int8" in generation_config
    assert "qwen3_0_6b_lora_kbit" in generation_config

    finetuning_config = read_yaml(str(config_dir / "finetuning_config.yaml"))
    assert "qwen3_0_6b" in finetuning_config
    assert "qwen3_0_6b_lora" in finetuning_config
    assert "qwen3_0_6b_int8" in finetuning_config
    assert "qwen3_0_6b_lora_int8" in finetuning_config
    assert "qwen3_0_6b_lora_kbit" in finetuning_config


def test_qwen3_lora_instruction_sft(monkeypatch):
    class DummyInstructionDataset:
        config_name = "instruction_dataset"

        def __init__(self, payload):
            self.payload = payload
            self._meta = type("Meta", (), {})()

        @property
        def meta(self):
            return self._meta

        def __len__(self):
            return len(self.payload["instruction"])

        def __getitem__(self, idx):
            return {key: values[idx] for key, values in self.payload.items()}

    class DummyTokenizer:
        eos_token_id = 0
        pad_token_id = 0
        pad_token = "<pad>"

        def __call__(self, _):
            return {"input_ids": [0], "attention_mask": [1]}

        def pad(self, samples, padding=True, max_length=None, return_tensors=None):
            batch_size = len(samples)
            return {
                "input_ids": [[0] for _ in range(batch_size)],
                "attention_mask": [[1] for _ in range(batch_size)],
            }

    class DummyModel:
        def to(self, *_):
            return self

        def eval(self):
            return self

        def train(self):
            return self

    class DummyEngine:
        def __init__(self, *_, **__):
            self.model = DummyModel()
            self.tokenizer = DummyTokenizer()

        def save(self, *_):
            return None

    class DummyCollator:
        def __init__(self, *_, **__):
            self.calls = 0

        def __call__(self, batches):
            self.calls += 1
            batch_size = len(batches)
            return {
                "input_ids": [[0] for _ in range(batch_size)],
                "targets": [[0] for _ in range(batch_size)],
            }

    trainers = []

    class DummyTrainer:
        def __init__(
            self,
            engine,
            dataset,
            collate_fn,
            num_epochs,
            batch_size,
            learning_rate,
            optimizer_name,
            use_lora=False,
            use_deepspeed=False,
            logger=True,
        ):
            self.engine = engine
            self.dataset = dataset
            self.collate_fn = collate_fn
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.optimizer_name = optimizer_name
            self.use_lora = use_lora
            self.use_deepspeed = use_deepspeed
            self.logger = logger
            self.fit_called = False
            trainers.append(self)

        def fit(self):
            self.fit_called = True
            batch = self.collate_fn([self.dataset[0]])
            assert "input_ids" in batch
            assert len(batch["input_ids"]) == 1

    monkeypatch.setitem(BaseEngine.registry, Qwen3LoraEngine.config_name, DummyEngine)
    monkeypatch.setitem(
        BasePreprocessor.registry, DummyInstructionDataset.config_name, DummyCollator
    )
    monkeypatch.setitem(
        BaseTrainer.registry, LightningTrainer.config_name, DummyTrainer
    )

    dataset = DummyInstructionDataset(
        {
            "instruction": [
                "Rewrite the sentence in simple terms.",
                "Translate to English.",
            ],
            "text": [
                "Quantum entanglement exhibits spooky action.",
                "Bonjour, comment ca va?",
            ],
            "target": ["Particles can stay linked.", "Hello, how are you?"],
        }
    )

    model = BaseModel.create("qwen3_0_6b_lora")
    model.finetune(dataset=dataset)

    assert trainers and trainers[0].fit_called
