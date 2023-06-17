import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.utils import PeftConfig, PeftType, transpose
from peft.tuners.lora import (
    LoraConfig,
    mark_only_lora_as_trainable,
    Linear,
    MergedLinear,
    Linear8bitLt,
    MergedLinear8bitLt,
    LoraLayer,
    is_bnb_available,
)


def is_gptq_available():
    return importlib.util.find_spec("gptqllama") is not None

if is_gptq_available():
    from gptqllama.quant import QuantLinear

class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        is_gtq_quantized = getattr(self.model, "gptq", False)  # Step 1: Check if the model is GTQ quantized

        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif is_gptq_available() and isinstance(target, QuantLinear):
                    kwargs.update(
                        {
                            "bits": target.bits,
                            "groupsize": target.groupsize,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = LinearqbitLt(target.infeatures, target.outfeatures, bias=bias, **kwargs)
                        new_module.scales = target.scales
                        new_module.qzeros = target.qzeros
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinearqbitLt(target.infeatures, target.outfeatures, bias=bias, **kwargs)
                        new_module.scales = target.scales
                        new_module.qzeros = target.qzeros
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if is_gptq_available() and isinstance(old_module, QuantLinear):
            new_module.qweight = old_module.qweight
            if old_module.bias is not None:
                new_module.bias = old_module.bias
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.qweight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.qweight.device)
        else:
            new_module.weight = old_module.weight
            if old_module.bias is not None:
                new_module.bias = old_module.bias
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.weight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

if is_gptq_available():
    class LinearqbitLt(QuantLinear, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            
            QuantLinear.__init__(
                self,
                kwargs.get('bits', 4),
                kwargs.get('groupsize', 128),
                in_features,
                out_features,
            )

            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.qweight.requires_grad = False
                self.scales.requires_grad = False
                self.qzeros.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def forward(self, x: torch.Tensor):
            custom_layer_output = super().forward(x).clone()
            
            dtype = custom_layer_output.dtype
            x = x.float()
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(dtype) * self.scaling
            result = custom_layer_output + lora_output
            return result

    class MergedLinearqbitLt(QuantLinear, LoraLayer):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            raise NotImplementedError("MergedLinearqbitLt is not implemented yet")
            QuantLinear.__init__(
                self,
                kwargs.get('bits', 4),
                kwargs.get('groupsize', 128),
                in_features,
                out_features,
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            if out_features % len(enable_lora) != 0:
                raise ValueError("The length of enable_lora must divide out_features")
            self.enable_lora = enable_lora
            # Actual trainable parameters
            if r > 0 and any(enable_lora):
                self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
                self.lora_B = nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                )
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.qweight.requires_grad = False
                # Compute the indices
                self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[enable_lora, :] = True
                self.lora_ind = self.lora_ind.view(-1)
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def zero_pad(self, x):
            result = x.new_zeros((*x.shape[:-1], self.out_features))
            result = result.view(-1, self.out_features)
            result[:, self.lora_ind] = x.reshape(
                -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
            )
            return result.view((*x.shape[:-1], self.out_features))

        def forward(self, x: torch.Tensor):
            result = super().forward(x)#.detach()
            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                    result += output
                else:
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B) * self.scaling
                    result += output
            return result