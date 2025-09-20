from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


def setup_harmony_format(tokenizer):
    """Set up tokenizer for OpenAI's harmony response format."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable chat template for harmony response format
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        # Basic chat template for harmony format if not already set
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"


# GPT-OSS 120B Engines
class GPTOSS120BEngine(CausalEngine):
    config_name: str = "gpt_oss_120b_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-120b"
        super().__init__(
            model_name=model_name, weights_path=weights_path, trust_remote_code=True
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS120BLoraEngine(CausalLoraEngine):
    config_name: str = "gpt_oss_120b_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-120b"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS120BInt8Engine(CausalEngine):
    config_name: str = "gpt_oss_120b_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-120b"
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS120BLoraInt8Engine(CausalLoraEngine):
    config_name: str = "gpt_oss_120b_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-120b"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS120BLoraKbitEngine(CausalLoraEngine):
    config_name: str = "gpt_oss_120b_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-120b"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_4bit=True,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


# GPT-OSS 20B Engines
class GPTOSS20BEngine(CausalEngine):
    config_name: str = "gpt_oss_20b_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-20b"
        super().__init__(
            model_name=model_name, weights_path=weights_path, trust_remote_code=True
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS20BLoraEngine(CausalLoraEngine):
    config_name: str = "gpt_oss_20b_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-20b"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS20BInt8Engine(CausalEngine):
    config_name: str = "gpt_oss_20b_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-20b"
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS20BLoraInt8Engine(CausalLoraEngine):
    config_name: str = "gpt_oss_20b_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-20b"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)


class GPTOSS20BLoraKbitEngine(CausalLoraEngine):
    config_name: str = "gpt_oss_20b_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "openai/gpt-oss-20b"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_4bit=True,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        setup_harmony_format(self.tokenizer)
