from .quant import make_quant, autotune_warmup
import torch
import torch.nn as nn
import transformers
from transformers import LlamaConfig, LlamaForCausalLM


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def load_quant(model, checkpoint, wbits, groupsize=128, warmup_autotune=True):
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    del layers
    
    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    if warmup_autotune:
        autotune_warmup(model)
    model.seqlen = 2048
    print('Done.')
    return model

def prepare_model_for_intq_training(
    model_name,
    intq_checkpoint,
    wbits,
    groupsize,
):
    quant_model = load_quant(model_name, intq_checkpoint, wbits, groupsize)
    quant_model.gptq = True
    return quant_model
