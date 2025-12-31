import argparse
import importlib.util
from pathlib import Path

import torch

from data import load_dataset_texts
from model import GPTLanguageModel
from tokenizers import build_vocab
from utils import get_device, load_model_state_dict, load_vocab_chars


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

    block_size = get_config_value(config, "block_size", 256)
    n_embd = get_config_value(config, "n_embd", 256)
    n_head = get_config_value(config, "n_head", 6)
    n_layer = get_config_value(config, "n_layer", 8)
    dropout = get_config_value(config, "dropout", 0.2)

    save_path = get_config_value(config, "save_path", f"models/{name}.pth")
    max_new_tokens = get_config_value(config, "max_new_tokens", 100)

    device = get_device()
    print(f"Device being used: {device}")

    _, _, text = load_dataset_texts(
        dataset_path=dataset_path,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
    )

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

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    checkpoint_tokenizer = checkpoint.get("tokenizer")
    if checkpoint_tokenizer:
        if checkpoint_tokenizer != tokenizer_config:
            print("Tokenizer config differs from checkpoint; using checkpoint version.")
        tokenizer_config = checkpoint_tokenizer

    if tokenizer_config["tokenizer_type"] == "char" and tokenizer_config["vocab_chars"] is None:
        tokenizer_config["vocab_chars"] = load_vocab_chars(save_path) or sorted(list(set(text)))

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

    model = GPTLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout)
    model = model.to(device)

    load_model_state_dict(model, checkpoint['model_state_dict'])

    model.eval()
    with torch.inference_mode():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        output = model.generate(context, max_new_tokens=max_new_tokens, block_size=block_size)
        print(decode(output[0].tolist()))


if __name__ == "__main__":
    main()
