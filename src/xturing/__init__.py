from importlib import import_module

_LAZY_EXPORTS = {
    "BaseDataset": ("xturing.datasets", "BaseDataset"),
    "InstructionDataset": ("xturing.datasets", "InstructionDataset"),
    "TextDataset": ("xturing.datasets", "TextDataset"),
    "BaseEngine": ("xturing.engines", "BaseEngine"),
    "GPT2Engine": ("xturing.engines", "GPT2Engine"),
    "GPT2LoraEngine": ("xturing.engines", "GPT2LoraEngine"),
    "GPTJEngine": ("xturing.engines", "GPTJEngine"),
    "GPTJLoraEngine": ("xturing.engines", "GPTJLoraEngine"),
    "LLamaEngine": ("xturing.engines", "LLamaEngine"),
    "LlamaLoraEngine": ("xturing.engines", "LlamaLoraEngine"),
    "BaseModel": ("xturing.models", "BaseModel"),
    "GPT2": ("xturing.models", "GPT2"),
    "GPT2Lora": ("xturing.models", "GPT2Lora"),
    "GPTJLora": ("xturing.models", "GPTJLora"),
    "Llama": ("xturing.models", "Llama"),
    "LlamaLora": ("xturing.models", "LlamaLora"),
    "BaseTrainer": ("xturing.trainers", "BaseTrainer"),
    "LightningTrainer": ("xturing.trainers", "LightningTrainer"),
}

__all__ = list(_LAZY_EXPORTS)


def _configure_external_loggers():
    try:
        from xturing.utils.external_loggers import configure_external_loggers

        configure_external_loggers()
    except Exception:
        # Keep package import light and resilient when optional stacks are unavailable.
        pass


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'xturing' has no attribute '{name}'")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


_configure_external_loggers()
