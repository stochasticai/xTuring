import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from torch.nn import KLDivLoss
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

from xturing.engines.lora_engine.lora import LoraConfig, LoraModel, load_quant
from xturing.engines.quant_utils.qerdataloading import get_c4


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="decapoda-research/llama-7b-hf",
        help="The base model.",
    )
    parser.add_argument(
        "--intq_checkpoint",
        type=str,
        default="llama7b-2bit-128g.pt",
        help="The intq checkpoint.",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=2,
        help="The number of bits for weight quantization.",
    )
    parser.add_argument("--groupsize", type=int, default=128, help="The group size.")
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="The Lora alpha value."
    )
    parser.add_argument("--lora_r", type=int, default=32, help="The Lora r value.")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="The Lora dropout rate."
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        help="List of target modules for Lora.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=2048, help="The number of samples."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="The learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size.")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="The number of epochs."
    )
    parser.add_argument("--kl_weight", type=float, default=1.0, help="The KL weight.")
    parser.add_argument("--ce_weight", type=float, default=200.0, help="The CE weight.")
    parser.add_argument(
        "--trainable_kl_weight",
        action="store_true",
        help="Whether to learn the KL weight.",
        default=False,
    )
    parser.add_argument(
        "--trainable_ce_weight",
        action="store_true",
        help="Whether to learn the CE weight.",
        default=False,
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="The weight decay."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="The frequency (period) of saving checkpoints in epochs.",
    )
    parser.add_argument(
        "--intra_save_freq",
        type=int,
        default=200,
        help="The period (in num_batches) of saving checkpoints within an epoch.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--seqlen", type=int, default=2048, help="The sequence length.")
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="Use cached distillation outputs.",
    )
    parser.add_argument(
        "--train_cache_dir",
        type=str,
        default="###ANONYMIZED###/train_cache/",
        help="Training cache directory.",
    )
    parser.add_argument(
        "--val_cache_dir",
        type=str,
        default="###ANONYMIZED###/val_cache/",
        help="Validation cache directory.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="The directory for saving and loading checkpoints.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="The directory for saving the final model.",
    )

    return parser.parse_args()


def get_lora_model(model, config):
    return LoraModel(config, model)


def prepare_models(args):
    if not args.cache:
        fp_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16
        ).to("cuda")
    model = load_quant(
        args.base_model,
        args.intq_checkpoint,
        args.wbits,
        args.groupsize,
    ).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        peft_type="CAUSAL_LM",
    )
    model = get_lora_model(model, config)
    # model = get_peft_model(model, config)
    return model, fp_model if not args.cache else model


def reduce_loss(pointwise_loss, reduction="batchmean"):
    if reduction == "batchmean":
        if pointwise_loss.dtype == torch.float16:
            # If the loss sum is larger than 65536, it will overflow during the mean computation.
            # If that's the case, we'll compute the iterative mean across the batch dimension
            # new average = old average * (n-1)/n + (new_loss)/n).
            if torch.isinf(pointwise_loss.sum()):
                loss = torch.tensor(
                    [0], dtype=torch.float16, device=pointwise_loss.device
                )
                for i in range(pointwise_loss.size(0)):
                    sample_loss = pointwise_loss[i].sum()
                    # Divide first to avoid overflow.
                    loss = (loss / (i + 1)) * i + sample_loss / (i + 1)
                return loss
        return pointwise_loss.sum() / pointwise_loss.size(0)
    elif reduction == "none":
        return pointwise_loss
    else:
        raise NotImplementedError(f"Unknown reduction {reduction}")


def train_model(args, model, fp_model, trainloader, valenc):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    kl_lossfn = KLDivLoss(reduction="none", log_target=True)
    base_model_name = (
        args.intq_checkpoint
        + "-qer-r"
        + str(args.lora_r)
        + "-tm"
        + str(args.lora_target_modules)
        + "-ce"
        + str(args.ce_weight)
        + "-kl"
        + str(args.kl_weight)
        + "-lr"
        + str(args.lr)
        + "-bs"
        + str(args.batch_size)
        + "-wd"
        + str(args.weight_decay)
        + "-dstl"
        + str(args.train_cache_dir.split("/")[-2])
    )
    wandb.init(project="qer", name=base_model_name, config=args)
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(args.ckpt_dir, f"{base_model_name}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            m_class = model.__class__
            model = m_class.from_pretrained(model.base_model, ckpt_path).to("cuda")
            print(f"Loaded checkpoint from {ckpt_path}")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
    else:
        start_epoch = 0

    if args.trainable_kl_weight:
        kl_weight = torch.nn.Parameter(torch.tensor(args.kl_weight))
        optimizer.add_param_group({"params": kl_weight})
    else:
        kl_weight = args.kl_weight
    if args.trainable_ce_weight:
        ce_weight = torch.nn.Parameter(torch.tensor(args.ce_weight))
        optimizer.add_param_group({"params": ce_weight})
    else:
        ce_weight = args.ce_weight

    def get_cached_targets(batch_idx, batch_size, cache_dir):
        with ThreadPoolExecutor() as executor:
            targets = list(
                executor.map(
                    lambda idx: torch.load(os.path.join(cache_dir, f"target_{idx}.pt")),
                    range(batch_idx * batch_size, (batch_idx + 1) * batch_size),
                )
            )
        return torch.cat(
            targets, dim=0
        )  # Concatenate targets along the batch dimension

    def custom_forward(inp, labels):
        model_out = model(input_ids=inp, labels=labels)
        return model_out.logits

    def eval_model(model, valenc, train_flag=False, max_samples=256):
        val_pbar = tqdm(enumerate(valenc), total=len(valenc))
        model.eval()
        if not args.cache:
            fp_model.eval()
        with torch.no_grad():
            samples = 0
            total_loss = 0
            total_kl_loss = 0
            total_ce_loss = 0
            for batch_idx, (inp, labels) in val_pbar:
                if samples > max_samples:
                    break

                inp, labels = inp.squeeze(1).to("cuda"), labels.squeeze(1).to("cuda")

                model_out = model(input_ids=inp, labels=labels).logits

                model_out_logsoftmax = model_out.add(1e-7).log_softmax(dim=-1)

                if args.cache:
                    pre_target = get_cached_targets(
                        batch_idx,
                        args.batch_size,
                        args.val_cache_dir if not train_flag else args.train_cache_dir,
                    )
                    target = pre_target.add(1e-7).log_softmax(dim=-1)
                else:
                    target = fp_model(input_ids=inp, labels=labels).logits.log_softmax(
                        dim=-1
                    )

                kl_loss_pointwise = kl_lossfn(model_out_logsoftmax, target)
                kl_loss = reduce_loss(kl_loss_pointwise, reduction="batchmean")
                ce_target = inp[:, 1:].contiguous().view(-1)
                ce_logits = (
                    model_out[:, :-1, :].contiguous().view(-1, model_out.size(-1))
                )
                ce_loss = F.cross_entropy(ce_logits, ce_target)
                total_loss += kl_weight * kl_loss + ce_weight * ce_loss
                total_kl_loss += kl_loss.item()
                total_ce_loss += ce_loss.item()
                val_pbar.set_description(
                    f"KL loss: {kl_loss.item():.3f} | CE loss: {ce_loss.item()}"
                )
                samples += args.batch_size
            denominator = samples / args.batch_size
            val_loss = total_loss / denominator
            total_kl_loss /= denominator
            total_ce_loss /= denominator
            if train_flag:
                print(f"Training loss: {val_loss}")
                print(f"Training KL loss: {total_kl_loss}")
                print(f"Training CE loss: {total_ce_loss}")
            else:
                print(f"Validation loss: {val_loss}")
                print(f"Validation KL loss: {total_kl_loss}")
                print(f"Validation CE loss: {total_ce_loss}")
            return val_loss, total_kl_loss, total_ce_loss

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        val_loss, val_kl_loss, val_ce_loss = eval_model(model, valenc)
        train_loss, train_kl_loss, train_ce_loss = eval_model(
            model, trainloader, train_flag=True
        )
        wandb.log(
            {
                "Epoch": epoch,
                "Train loss": train_loss,
                "Validation loss": val_loss,
                "Train KL loss": train_kl_loss,
                "Validation KL loss": val_kl_loss,
                "Train CE loss": train_ce_loss,
                "Validation CE loss": val_ce_loss,
            },
            step=epoch * len(trainloader),
        )
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        print(f"Epoch {epoch} validation loss: {val_loss}")
        model.train()
        fp_model.eval() if not args.cache else None
        for batch_idx, (inp, labels) in pbar:
            inp, labels = inp.squeeze(1).to("cuda"), labels.squeeze(1).to("cuda")

            model_out = checkpoint(custom_forward, inp, labels, use_reentrant=False)
            model_out_logsoftmax = model_out.add(1e-7).log_softmax(dim=-1)
            if args.cache:
                target = (
                    get_cached_targets(batch_idx, args.batch_size, args.train_cache_dir)
                    .add(1e-7)
                    .log_softmax(dim=-1)
                )
            else:
                with torch.no_grad():
                    target = fp_model(input_ids=inp, labels=labels).logits.log_softmax(
                        dim=-1
                    )

            ce_target = inp[:, 1:].contiguous().view(-1)
            ce_logits = model_out[:, :-1, :].contiguous().view(-1, model_out.size(-1))
            ce_loss = F.cross_entropy(ce_logits, ce_target)
            kl_loss_pointwise = kl_lossfn(model_out_logsoftmax, target)
            kl_loss = reduce_loss(kl_loss_pointwise, reduction="batchmean")
            loss = kl_weight * kl_loss + ce_weight * ce_loss
            # Check if loss is nan
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss.backward()
            # Double check if loss is nan
            if torch.isnan(loss):
                continue
            optimizer.step()
            description = f"Epoch {epoch} batch {batch_idx} ce loss: {ce_loss.item():.3f} | kl loss: {kl_loss.item():.2f} | total loss: {loss.item():.3f}"
            if args.trainable_ce_weight:
                description += f" | ce weight: {ce_weight}"
            if args.trainable_kl_weight:
                description += f" | kl weight: {kl_weight}"
            pbar.set_description(description)
            wandb.log(
                {
                    "Epoch": epoch,
                    "Batch": batch_idx,
                    "CE loss": ce_loss.item(),
                    "KL loss": kl_loss.item(),
                    "Total loss": loss.item(),
                },
                step=epoch * len(trainloader) + batch_idx,
            )
            optimizer.zero_grad()

            if batch_idx % args.intra_save_freq == 0 and batch_idx != 0:
                with torch.no_grad():
                    ckpt = {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        ckpt,
                        f"{args.save_dir}/tmp/ckpts/"
                        + base_model_name
                        + f"-{epoch}-{batch_idx}.pt",
                    )
                    model.save_pretrained(
                        f"{args.save_dir}/tmp/models/{base_model_name}-{epoch}-{batch_idx}"
                    )

        if epoch % args.save_freq == 0:
            ckpt = {
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, f"{args.save_dir}/tmp/ckpts/" + base_model_name + ".pt")
            model.save_pretrained(
                f"{args.save_dir}/tmp/models/{base_model_name}-{epoch}"
            )
            with open(f"{args.save_dir}/tmp/logs/{base_model_name}.txt", "a") as f:
                f.write(f"Epoch {epoch - 1} val loss: {val_loss}\n")
                f.write(f"Epoch {epoch - 1} train loss: {train_loss}\n")

    model.save_pretrained(base_model_name + ".pt")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    wandb.login()
    if args.cache:
        model = prepare_models(args)[0]
    else:
        model, fp_model = prepare_models(args)
    trainloader, valenc = get_c4(
        args.base_model, args.seqlen, args.n_samples, args.batch_size, args.seed
    )
    if args.cache:
        train_model(args, model, None, trainloader, valenc)  # Pass None for fp_model
    else:
        train_model(args, model, fp_model, trainloader, valenc)


if __name__ == "__main__":
    main()
