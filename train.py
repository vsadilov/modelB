import argparse
import importlib.util
import os
from pathlib import Path

import torch

from data import load_dataset_texts
from model import GPTLanguageModel
from tokenizers import build_vocab
from utils import (
    get_device,
    get_model_state_dict,
    load_model_state_dict,
    load_vocab_chars,
    move_optimizer_state,
    save_vocab_chars,
)


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_config_value(config, name, default=None):
    return getattr(config, name, default)


def list_configs(root):
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(p.name for p in root_path.glob("*.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/v5.py")
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    if args.list_experiments:
        configs = list_configs("config")
        if configs:
            print("Configs:")
            for name in configs:
                print(f"  {name}")
        else:
            print("No configs found.")
        return

    config_path = Path(args.config)
    config = load_config(config_path)

    name = get_config_value(config, "name", config_path.stem)
    dataset_path = get_config_value(config, "dataset_path", "data/tinyshakespeare.txt")
    train_dataset_path = get_config_value(config, "train_dataset_path", None)
    val_dataset_path = get_config_value(config, "val_dataset_path", None)

    batch_size = get_config_value(config, "batch_size", 64)
    block_size = get_config_value(config, "block_size", 256)
    max_iters = get_config_value(config, "max_iters", 10)
    eval_interval = get_config_value(config, "eval_interval", 300)
    learning_rate = get_config_value(config, "learning_rate", 3e-4)
    eval_iters = get_config_value(config, "eval_iters", 200)
    n_embd = get_config_value(config, "n_embd", 256)
    n_head = get_config_value(config, "n_head", 6)
    n_layer = get_config_value(config, "n_layer", 8)
    dropout = get_config_value(config, "dropout", 0.2)

    save_path = get_config_value(config, "save_path", f"models/{name}.pth")

    device = get_device()
    print(f"Device being used: {device}")

    torch.manual_seed(1337)

    # Tokenizer settings are saved in checkpoints to keep vocab versions aligned.
    tokenizer_config = {
        "tokenizer_type": get_config_value(config, "tokenizer_type", "char"),
        "vocab_chars": None,
        "sp_model_path": get_config_value(config, "sp_model_path", None),
        "sp_vocab_size": get_config_value(config, "sp_vocab_size", 8000),
        "sp_model_type": get_config_value(config, "sp_model_type", "bpe"),
        "sp_character_coverage": get_config_value(config, "sp_character_coverage", 1.0),
        "sp_train_if_missing": get_config_value(config, "sp_train_if_missing", False),
    }

    checkpoint = None
    try:
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        checkpoint_tokenizer = checkpoint.get("tokenizer")
        if checkpoint_tokenizer:
            if checkpoint_tokenizer != tokenizer_config:
                print("Tokenizer config differs from checkpoint; using checkpoint version.")
            tokenizer_config = checkpoint_tokenizer
    except FileNotFoundError:
        checkpoint = None

    train_text, val_text, text = load_dataset_texts(
        dataset_path=dataset_path,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
    )

    if tokenizer_config["tokenizer_type"] == "char" and tokenizer_config["vocab_chars"] is None:
        tokenizer_config["vocab_chars"] = load_vocab_chars(save_path) or sorted(list(set(text)))
    save_vocab_chars(save_path, tokenizer_config.get("vocab_chars"))

    vocab_size, encode, decode = build_vocab(
        text,
        tokenizer_type=tokenizer_config["tokenizer_type"],
        vocab_chars=tokenizer_config["vocab_chars"],
        sp_model_path=tokenizer_config["sp_model_path"],
        sp_vocab_size=tokenizer_config["sp_vocab_size"],
        sp_model_type=tokenizer_config["sp_model_type"],
        sp_character_coverage=tokenizer_config["sp_character_coverage"],
        sp_train_if_missing=tokenizer_config["sp_train_if_missing"],
    )

    if val_text is not None:
        train_data = torch.tensor(encode(train_text), dtype=torch.long)
        val_data = torch.tensor(encode(val_text), dtype=torch.long)
    else:
        # Default split for backward compatibility.
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

    def get_batch(split):
        data_split = train_data if split == "train" else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i : i + block_size] for i in ix])
        y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                if torch.is_tensor(loss):
                    loss_value = loss.mean().item()
                else:
                    loss_value = float(loss)
                losses[k] = loss_value
            out[split] = losses.mean()
        model.train()
        return out

    model = GPTLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    start_iter = 0
    if checkpoint is not None:
        load_model_state_dict(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(optimizer, device)
        start_iter = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint: {save_path}", flush=True)
    else:
        print(f"No checkpoint found at {save_path}, starting fresh.", flush=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #   Training loop execution
    for iter in range(start_iter, max_iters):
        
        # Get batch
        xb, yb = get_batch("train")

        # Forward pass
        _, loss = model(xb, yb)

        # Backward pass and optimization step
        if torch.is_tensor(loss):
            loss = loss.mean()

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Evaluate and print loss.
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}",
                flush=True,
            )

        # Save checkpoint.
        if iter % (eval_interval * 3) == 0:
            torch.save(
                {
                    "model_state_dict": get_model_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": iter,
                    "loss": loss.item() if "loss" in locals() else None,
                    "tokenizer": tokenizer_config,
                },
                save_path,
            )
            print(f"Checkpoint saved at iteration {iter}", flush=True)

    torch.save(
        {
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": max_iters - 1,
            "loss": loss.item() if "loss" in locals() else None,
            "tokenizer": tokenizer_config,
        },
        save_path,
    )
    print(f"Training completed. Final model saved at {save_path}.", flush=True)


if __name__ == "__main__":
    main()
