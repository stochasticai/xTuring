import argparse
import torch
from tqdm import tqdm

from datasets import load_dataset
from gptq import *
from modelutils import *
from quant import *
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, modeling_utils
from peft import PeftModel
from peft.utils.load_utils import load_quant

from pathlib import Path
import datasets
import pickle 
import os

def main(args):
    name = args.peft_pretrained.split('/')[-1]
    device = torch.device('cuda:0')

    if args.wbits == 16:
        print('Loading FP16 model ...')
        model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    elif args.wbits in [2, 3, 4]:
        print('Loading INT' + str(args.wbits) + ' model ...')
        checkpoint_dir = args.intq_checkpoint[:-12] + str(args.wbits) + args.intq_checkpoint[-11:]
        model = load_quant(args.base_model, checkpoint_dir, args.wbits, args.groupsize, device)
        if args.peft_pretrained != '':
            model = PeftModel.from_pretrained(model, args.peft_pretrained)
    else:
        raise NotImplementedError('wbits=' + str(args.wbits) + ' is not supported!')
    model = model.to(device)


    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    if os.path.isfile(args.encodings_pickle):
        with open(args.encodings_pickle, 'rb') as handle:
            encodings = pickle.load(handle)
        print('Finished loading encodings!')

    else:
        test = load_dataset("c4", "en", split="validation", streaming=True)
        total_text = ''
        for idx, t in enumerate(iter(test)):
            total_text += t['text'] + "\n\n"
            if idx % 30000 == 0:
                print(idx)
        encodings = tokenizer(total_text, return_tensors="pt")
        with open(args.encodings_pickle, 'wb') as handle:
            print('dumping pickle...')
            pickle.dump(encodings, handle)

    max_length = 2048
    stride = args.stride
    seq_len = encodings.input_ids.size(1)

    print(max_length, stride, encodings.input_ids.size(0), seq_len)

    #'''
    b_flag = False
    b_cout = 0
    nlls, i_list, t_list = [], [], []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if end_loc == seq_len:
            if len(i_list) != 0:
                with torch.no_grad():
                    input_batch = torch.stack(i_list)
                    target_batch = torch.stack(t_list)
                    outputs = model(input_batch, labels=target_batch)
                    neg_log_likelihood = outputs.loss
                nlls.append(neg_log_likelihood)
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)

        else:
            i_list.append(input_ids[0])
            t_list.append(target_ids[0])
            b_cout += 1

            if args.batch_size == b_cout:
                with torch.no_grad():
                    input_batch = torch.stack(i_list)
                    target_batch = torch.stack(t_list)
                    outputs = model(input_batch, labels=target_batch)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss
                
                nlls.append(neg_log_likelihood)
                i_list, t_list = [], []
                b_cout = 0

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl)
    #with open('tmp/ppl/' + name + '.txt', 'w') as f:/
    #    f.write(str(ppl.item()))
    #'''

if __name__ == '__main__':
    DATASET_CACHE_PATH = ... ###ANONYMIZED###/tmp/dataset_cache
    datasets.config.HF_DATASETS_CACHE = Path(DATASET_CACHE_PATH)

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', default='decapoda-research/llama-7b-hf', type=str, help='Base model')
    parser.add_argument('--intq_checkpoint', default='llama7b-2bit-128g.pt', type=str, help='Checkpoint for the model')
    parser.add_argument('--wbits', default=4, type=int, help='Weight bits')
    parser.add_argument('--groupsize', default=128, type=int, help='Group size')
    parser.add_argument('--peft_pretrained', default='', type=str, help='Path to pretrained Peft model')
    parser.add_argument('--stride', default=2048, type=int, help='Stride length')
    parser.add_argument('--encodings_pickle', default='###ANONYMIZED###/tmp/encodings.pickle', type=str, help='Stride length')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')

    args = parser.parse_args()

    main(args)