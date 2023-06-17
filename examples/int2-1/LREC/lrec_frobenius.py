"""
Frobenius norm per-layer optimization. Not part of the paper.
"""

import time
import json
import random
import copy
import shutil

import torch
import torch.nn as nn
import os
from gptq import *
from modelutils import *
from torch.nn import Module


from peft import (LoraConfig, get_peft_model,  # noqa: E402
                  get_peft_model_state_dict, prepare_model_for_int8_training,
                  set_peft_model_state_dict)
from peft.utils import prepare_model_for_intq_training
from peft.tuners.lora import LinearqbitLt
from peft.utils.quant import QuantLinear
from peft.tuners.lora import LoraLayer

class LoraAB(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        lin: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super(LoraAB, self).__init__()
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        in_features = lin.in_features
        out_features = lin.out_features
        self.old_linear = lin.cpu()
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_A.weight = torch.nn.Parameter(torch.nn.init.kaiming_uniform(self.lora_A.weight, a=math.sqrt(5)))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return lora_output

def make_quant(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model

def llama_sequential_and_pack(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    with torch.no_grad():
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    lora_As, lora_Bs, initial_losses, losses = {}, {}, {}, {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for n, module in model.named_modules():
            if layer == module:
                halfname = n
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], halfname + '.' + name)
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                gptq[name].fullname = halfname + '.' + name
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data, lora_target_modules=args.lora_target_modules, path=args.compressed_save_path)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            with torch.no_grad():
                for j in range(args.nsamples):
                    outs[j] = layer.cuda()(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
                scale,zero,g_idx = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, lora_target_modules=args.lora_target_modules, path=args.compressed_save_path)
                fullname = gptq[name].fullname
                oldlinear = gptq[name].layer
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu(),fullname,oldlinear.cpu())

                gptq[name].free()
        # Pack the current layer
        layer_quantizers = {n: quantizers[n] for n in quantizers if n.startswith(f'model.layers.{i}')}
        peft_layer = get_peft_layer(layer, i, args)

        DEBUG = i in ()
        A, B, init_loss, loss = get_lora_As_Bs(peft_layer,
                                layer_quantizers,
                                args,
                                layer_idx=i,
                                DEBUG=DEBUG)
        layers[i] = layer.cpu()
        lora_As.update(A)
        lora_Bs.update(B)
        initial_losses.update(init_loss)
        losses.update(loss)
        with open(os.path.join(f'initial_losses_bits{args.wbits}_r{args.lora_r}.json'), 'w') as f:
            json.dump(initial_losses, f, indent=4)
        with open(os.path.join(f'losses_bits{args.wbits}_r{args.lora_r}.json'), 'w') as f:
            json.dump(losses, f, indent=4)
        # Delete all saved cache at args.compressed_save_path
        # By deleting all the contents of the folder without deleting the folder itself
        if args.compressed_save_path is not None:
            for f in get_filenames(args.compressed_save_path):
                os.remove(os.path.join(args.compressed_save_path, f))
        # layer = peft_layer.old_linear
        replace_loralayers_with_linears(layer)
        print(f'Packing layer {i+1}/{len(layers)}...')
        llama_pack_layer(layer, layer_quantizers, args.wbits, args.groupsize)
        print(f'Packed layer {i+1}/{len(layers)}.')
        print()
        del layer
        del gptq
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model, quantizers, lora_As, lora_Bs

def get_filenames(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def obj(x, Qerr):
    """
    Returns the Frobenius norm ||x-Qerr||
    """
    return torch.norm(Qerr - x, p='fro')

def train_qlin(loralayer, lin, module_name, paths, base_path, lr, num_epochs=10, DEBUG=False):
    """
    Train the qlinear layer to minimize the error ||Qerr - Qlin||_F
    """
    print(f'Optimizing {module_name}...')
    # Dropout????
    loralayer.train()
    loralayer = loralayer.cuda()
    lin = lin.eval()
    lin = lin.cuda()
    lin.requires_grad_(False)
    optimizer = torch.optim.Adam(loralayer.parameters(), lr=lr)
    inp_paths = [p for p in paths if 'inp' in p]
    out_paths = [p for p in paths if 'out' in p]
    inp_paths = sorted(inp_paths, key=lambda x: int(x.split('_')[-1].strip('.pt')))
    out_paths = sorted(out_paths, key=lambda x: int(x.split('_')[-1].strip('.pt')))
    data = []
    for i in range(len(inp_paths)):
        data.append({
            'act': torch.load(
                os.path.join(base_path, inp_paths[i])
            ),
            'out': torch.load(
                os.path.join(base_path, out_paths[i])
            )
        })
    if DEBUG:
        # Save all the data to a different folder to analyze later
        # Also save lin and loralayer
        if not os.path.exists('debug'):
            os.mkdir('debug')
        if not os.path.exists(os.path.join('debug', module_name)):
            os.mkdir(os.path.join('debug', module_name))
        for i, elem in enumerate(data):
            torch.save(elem['act'], os.path.join('debug', module_name, f'act{i}.pt'))
            torch.save(elem['out'], os.path.join('debug', module_name, f'out{i}.pt'))
        torch.save(lin, os.path.join('debug', module_name, 'lin.pt'))
        torch.save(loralayer, os.path.join('debug', module_name, 'loralayer.pt'))
    initial_loss = evaluate_model(data, lin, loralayer, obj)
    print(f'Initial loss: {initial_loss}')

    for epoch in range(num_epochs):
        random.shuffle(data)
        epoch_loss = 0
        for i, elem in enumerate(data):
            optimizer.zero_grad()
            inp = elem['act'].cuda()
            qerr = elem['out'] - lin(inp)
            peft_qout = loralayer(inp)
            loss = obj(peft_qout, qerr)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}: {epoch_loss / len(data)}')
    del lin
    del data
    torch.cuda.empty_cache()
    loralayer = loralayer.cpu().eval()
    return loralayer.lora_A, loralayer.lora_B, initial_loss, loss.item()

def evaluate_model(data, lin, loralayer, objective_fn):
    total_loss = 0
    num_samples = len(data)

    with torch.no_grad():
        for elem in data:
            inp = elem['act'].cuda()
            qerr = elem['out'] - lin(inp)
            peft_qout = loralayer(inp)
            loss = objective_fn(peft_qout, qerr)
            total_loss += loss.item()

    avg_loss = total_loss / num_samples
    return avg_loss

def get_lora_As_Bs(peft_layer, layer_quantizers, args, layer_idx, DEBUG=False):
    layer_quantizers = copy.deepcopy(layer_quantizers)
    trainable_modules = [
        find_layers(peft_layer, [LoraAB])
    ]
    lora_As = {}
    lora_Bs = {}
    initial_losses = {}
    final_losses = {}
    print(f'Optimizing LORA for layer {layer_idx}...')
    print(f'Layer {layer_idx} has {len(trainable_modules)} trainable modules.')

    # Match the layer_quantizers to the trainable layers
    layer_quantizers = {'.'.join(k.split('.')[-2:]):v for k,v in layer_quantizers.items()}
    for i, layer in enumerate(trainable_modules):
        for name, module in layer.items():
            if name in layer_quantizers:
                paths = get_filenames(args.compressed_save_path)
                paths = [
                    p for p in paths if layer_quantizers[name][4] in p
                ]
                paths = [p for p in paths if 'inp' in p or 'out' in p]
                paths = sorted(paths, key=lambda x: int(x.split('_')[-1].strip('.pt')))
                lora_A, lora_B, initial_loss, loss = train_qlin(
                    module,
                    layer_quantizers[name][5],
                    layer_quantizers[name][4],
                    paths,
                    lr=args.lora_lr,
                    base_path=args.compressed_save_path,
                    num_epochs=args.lora_epochs,
                    DEBUG=DEBUG
                )
                lora_As[layer_quantizers[name][4]] = lora_A
                lora_Bs[layer_quantizers[name][4]] = lora_B
                initial_losses[layer_quantizers[name][4]] = initial_loss
                final_losses[layer_quantizers[name][4]] = loss
    return lora_As, lora_Bs, initial_losses, final_losses

def replace_qlinear_layers_with_linearqbitlt(module: Module, bits: int, groupsize: int, r: float, lora_alpha: float, lora_dropout: float, target_modules: list):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, QuantLinear):
            if name in target_modules:
                in_features, out_features = sub_module.infeatures, sub_module.outfeatures
                bias = sub_module.bias is not None
                new_module = LinearqbitLt(in_features, out_features, bits=bits, groupsize=groupsize, bias=bias, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                new_module.scales = sub_module.scales
                new_module.qzeros = sub_module.qzeros
                new_module.g_idx = sub_module.g_idx
                if bias:
                    new_module.bias = sub_module.bias
                _replace_module(module, name, new_module, sub_module)
        else:
            replace_qlinear_layers_with_linearqbitlt(sub_module, bits, groupsize, r, lora_alpha, lora_dropout, target_modules)


def replace_linear_layers_with_loralayers(module: Module, r: float, lora_alpha: float, lora_dropout: float, target_modules: list):
    for name, sub_module in module.named_children():
        if name in target_modules:
            in_features, out_features = sub_module.in_features, sub_module.out_features
            bias = sub_module.bias is not None
            new_module = LoraAB(sub_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            if bias:
                new_module.bias = sub_module.bias
            new_module = new_module.to(sub_module.weight.device)
            setattr(module, name, new_module)

        else:
            replace_linear_layers_with_loralayers(sub_module, r, lora_alpha, lora_dropout, target_modules)


def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    if isinstance(old_module, QuantLinear):
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

def _get_submodules(self, model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def get_peft_layer(peft_layer, i, args):
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # peft_layer = model.model.layers[i]

    replace_linear_layers_with_loralayers(peft_layer, config.r, config.lora_alpha, config.lora_dropout, config.target_modules)
    return peft_layer

def replace_loralayers_with_linears(module: Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, LoraAB):
            new_module = sub_module.old_linear
            setattr(module, name, new_module)
        else:
            replace_loralayers_with_linears(sub_module)

def llama_pack_layer(model, layer_quantizers, wbits, groupsize):
    layers = find_layers(model)
    layer_quantizers = {'.'.join(k.split('.')[-2:]):v for k,v in layer_quantizers.items()}
    layers = {n: layers[n] for n in layer_quantizers}
    make_quant(model, layer_quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])

    for name in qlayers:
        quantizer, scale, zero, g_idx, _, _ = layer_quantizers[name]
        qlayers[name].pack(layers[name].cpu(), scale, zero, g_idx)


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load',
        default='decapoda-research/llama-7b-hf'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.',
        default='c4'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=10,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--wbits', type=int, default=2, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.',
    )
    parser.add_argument(
        '--groupsize', type=int, default=128,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic',
        default=True,
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.',
        default=True,
    )
    parser.add_argument(
        '--lora_r', type=int, default=64,
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=128,
    )
    parser.add_argument(
        '--lora_dropout', type=float, default=0.05,
    )
    parser.add_argument(
        '--lora_target_modules', type=list,
        default=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    )
    parser.add_argument(
        '--lora_lr', type=float, default=0.001,
    )
    parser.add_argument(
        '--compressed_save_path', type=str, default='llamacompressed_prequant',
    )
    parser.add_argument(
        '--lora_epochs', type=int, default=15,
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.',
        default=False,
    )

    args = parser.parse_args()
    if os.path.exists(args.compressed_save_path):
        shutil.rmtree(args.compressed_save_path)
    os.makedirs(args.compressed_save_path, exist_ok=True)

    model = get_llama(args.model)
    model.eval()
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, lora_As, lora_Bs = llama_sequential_and_pack(model, dataloader, dev, args)
    torch.save(lora_As, os.path.join(args.compressed_save_path, 'lora_As.pt'))
    torch.save(lora_Bs, os.path.join(args.compressed_save_path, 'lora_Bs.pt'))
