import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

# Set all seeds to 0
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)


def get_c4(base_model, seqlen, n_samples, batch_size, seed=0):
    traindata, valdata = load_c4_datasets()

    tokenizer = LlamaTokenizer.from_pretrained(base_model, use_fast=False)

    trainloader = create_random_trainloader(
        traindata, tokenizer, seqlen, n_samples, seed
    )
    valenc = create_random_valenc(valdata, tokenizer, seqlen, seed)

    trainloader = DataLoader(
        trainloader, batch_size=batch_size, shuffle=False, drop_last=True
    )
    valenc = DataLoader(valenc, batch_size=batch_size, shuffle=False, drop_last=True)

    return trainloader, valenc


def load_c4_datasets():
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        use_auth_token=False,
    )
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        use_auth_token=False,
    )
    return traindata, valdata


def create_random_trainloader(traindata, tokenizer, seqlen, n_samples, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    trainloader = []
    pbar = tqdm(total=n_samples, desc="Creating trainloader")
    while len(trainloader) < n_samples:
        i = random.randint(0, len(traindata) - 1)
        text = traindata[i]["text"]
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] >= seqlen + 1:
            start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            end = start + seqlen
            inp = enc.input_ids[:, start:end]
            tar = torch.cat([torch.tensor([-100]), inp.squeeze()[:-1]]).unsqueeze(0)
            trainloader.append((inp, tar))
            pbar.update(1)
    return trainloader


def create_random_valenc(valdata, tokenizer, seqlen, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    valenc = []
    pbar = tqdm(total=256, desc="Creating valenc")
    while len(valenc) < 256:
        i = random.randint(0, len(valdata) - 1)
        text = valdata[i]["text"]
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] >= seqlen + 1:
            start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            end = start + seqlen
            inp = enc.input_ids[:, start:end]
            tar = torch.cat([torch.tensor([-100]), inp.squeeze()[:-1]]).unsqueeze(0)
            valenc.append((inp, tar))
            pbar.update(1)
    return valenc
