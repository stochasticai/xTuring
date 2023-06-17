import argparse

import torch
from datasets import load_dataset
from gptq import *
from modelutils import *
from quant import *
from torch.nn import KLDivLoss
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq,
                          LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
                          Trainer, TrainingArguments, modeling_utils)

from peft import PeftModel
from peft.utils.load_utils import load_quant


def kl_divergence(p_logits, q_logits):

    loss_fct = KLDivLoss(reduction="batchmean")
    p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q_probs = torch.nn.functional.softmax(q_logits, dim=-1)
    return loss_fct(p_log_probs, q_probs)


def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len):
    if data_point["input"]:
        full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        full_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""

    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len)

    return tokenized_full_prompt


def main(args):
    name = args.peft_pretrained.split("/")[-1]
    device = torch.device("cuda:0")

    model = load_quant(
        args.base_model, args.intq_checkpoint, args.wbits, args.groupsize, device
    )
    model = model.to(device)

    model = PeftModel.from_pretrained(model, args.peft_pretrained)
    if args.double_lora:
        model = PeftModel.from_pretrained(model, args.double_lora)
    if args.reference_model:
		# Change this to your own reference model format if needed
        base_model = AutoModelForCausalLM.from_pretrained(args.reference_model)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    if args.mode in ["alpaca_perplexity", "kl_divergence"]:
        dataset = load_dataset(args.dataset)
        train_val = dataset["train"].train_test_split(test_size=2000, shuffle=False, seed=42)
        dataset = train_val["test"]
        encodings = dataset.map(lambda dp: generate_and_tokenize_prompt(dp, tokenizer, args.cutoff_len))
        encodings.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    elif args.mode == "perplexity":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\\n\\n".join(test[args.col]), return_tensors="pt")

    if args.mode == "kl_divergence":
        base_model.eval().half().to(device)

    max_length = model.seqlen
    stride = args.stride
    seq_len = encodings.input_ids.size(1)

    results = []
    prev_end_loc = 0
    pbar = tqdm(range(0, seq_len, stride))
	# Implementation inspired by https://huggingface.co/docs/transformers/perplexity
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            if args.mode == "kl_divergence":
                base_logits = base_model.engine.model(input_ids, labels=target_ids).logits
                result = kl_divergence(outputs.logits, base_logits)
                pbar.set_description(f'KL: {result.item():.4f}')
            else:
                neg_log_likelihood = outputs.loss
                result = torch.exp(neg_log_likelihood)
                pbar.set_description(f'NLL: {neg_log_likelihood.item()}, PPL: {result.item()}')
            results.append(result)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    result = torch.stack(results).mean()
    print(result)
    with open('tmp/ppl/' + name + ' ' + args.dataset.split('/')[-1] + '.txt', 'w') as f:
        f.write(str(result.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["perplexity", "alpaca_perplexity", "kl_divergence"], required=True, help="Choose the mode for the script")
    parser.add_argument("--base_model", default="decapoda-research/llama-7b-hf", type=str, help="Base model")
    parser.add_argument("--intq_checkpoint", default="llama7b-2bit-128g.pt", type=str, help="Checkpoint for the model")
    parser.add_argument("--wbits", default=3, type=int, help="Weight bits")
    parser.add_argument("--groupsize", default=128, type=int, help="Group size")
    parser.add_argument("--peft_pretrained", default="tmp/models/llama13b-2bit-128g.pt-qer-r32-tm['q_proj', 'v_proj', 'k_proj', 'o_proj', 'up_proj', 'gate_proj']-ce120.0-kl0.5-lr1e-05-bs4-wd1e-05-dstltrain_cache-13b-2-400", type=str, help="Path to pretrained Peft model")
    parser.add_argument("--reference_model", default=None, type=str, help="Path to reference model")
    parser.add_argument("--double_lora", default=None, type=str, help="Path to second Peft model, Optional, leave empty if not used")
    parser.add_argument("--stride", default=512, type=int, help="Stride length")
    parser.add_argument("--dataset", default="alpaca", type=str, help="Dataset")
    parser.add_argument("--cutoff_len", default=512, type=int, help="Cutoff length")
    parser.add_argument("--col", default="sentence", type=str, help="Column name for the dataset")
    args = parser.parse_args()
    main(args)
