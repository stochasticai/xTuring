import argparse
import os
import random

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

from xturing.engines.quant_utils.qerdataloading import (
    create_random_trainloader,
    create_random_valenc,
    load_c4_datasets,
)

HF_CACHE_DIR = ...  ###ANONYMIZED###


def cache_distillation_outputs(
    base_model, seqlen, n_samples, train_cache_dir, val_cache_dir
):
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(base_model, use_fast=False)
    fp_model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, cache_dir=HF_CACHE_DIR, device_map="auto"
    )
    fp_model.eval()

    traindata, valdata = load_c4_datasets()
    trainloader = create_random_trainloader(traindata, tokenizer, seqlen, n_samples)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    valenc = create_random_valenc(valdata, tokenizer, seqlen)

    for idx, (inp, tar) in enumerate(trainloader):
        inp, tar = inp.squeeze(1).to("cuda"), tar.squeeze(1).to("cuda")
        with torch.no_grad():
            target = fp_model(input_ids=inp, labels=tar).logits  # .log_softmax(dim=-1)
        torch.save(target, os.path.join(train_cache_dir, f"target_{idx}.pt"))

    for idx, (inp, tar) in enumerate(valenc):
        inp, tar = inp.squeeze(1).to("cuda"), tar.squeeze(1).to("cuda")
        if idx < 8:
            torch.save(inp, os.path.join(val_cache_dir, f"input_{idx}.pt"))
            torch.save(tar, os.path.join(val_cache_dir, f"label_{idx}.pt"))
        with torch.no_grad():
            target = fp_model(input_ids=inp, labels=tar).logits  # .log_softmax(dim=-1)
        torch.save(target, os.path.join(val_cache_dir, f"target_{idx}.pt"))


if __name__ == "__main__":
    base_model = "decapoda-research/llama-7b-hf"
    seqlen = 2048
    n_samples = 10000
    train_cache_dir = ...  ###ANONYMIZED###
    val_cache_dir = ...  ###ANONYMIZED###
    seed = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=base_model)
    parser.add_argument("--seqlen", type=int, default=seqlen)
    parser.add_argument("--n_samples", type=int, default=n_samples)
    parser.add_argument("--train_cache_dir", type=str, default=train_cache_dir)
    parser.add_argument("--val_cache_dir", type=str, default=val_cache_dir)
    parser.add_argument("--seed", type=int, default=seed)
    args = parser.parse_args()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    cache_distillation_outputs(
        args.base_model,
        args.seqlen,
        args.n_samples,
        args.train_cache_dir,
        args.val_cache_dir,
    )
