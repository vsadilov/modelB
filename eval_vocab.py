import argparse
import importlib.util
from pathlib import Path

from data import load_dataset_texts
from tokenizers import build_vocab


def load_config(path):
    """Load a Python config file as a module."""
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_config_value(config, name, default=None):
    return getattr(config, name, default)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer vocabulary vs dataset.")
    parser.add_argument("--config", default="config/v5.py")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    dataset_path = get_config_value(config, "dataset_path", "data/tinyshakespeare.txt")
    train_dataset_path = get_config_value(config, "train_dataset_path", None)
    val_dataset_path = get_config_value(config, "val_dataset_path", None)
    tokenizer_type = get_config_value(config, "tokenizer_type", "char")
    sp_model_path = get_config_value(config, "sp_model_path", None)
    sp_vocab_size = get_config_value(config, "sp_vocab_size", 8000)
    sp_model_type = get_config_value(config, "sp_model_type", "bpe")
    sp_character_coverage = get_config_value(config, "sp_character_coverage", 1.0)
    sp_train_if_missing = get_config_value(config, "sp_train_if_missing", False)
    n_embd = get_config_value(config, "n_embd", None)

    _, _, text = load_dataset_texts(
        dataset_path=dataset_path,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
    )

    vocab_chars = None
    if tokenizer_type == "char":
        vocab_chars = sorted(list(set(text)))

    vocab_size, encode, _ = build_vocab(
        text,
        tokenizer_type=tokenizer_type,
        vocab_chars=vocab_chars,
        sp_model_path=sp_model_path,
        sp_vocab_size=sp_vocab_size,
        sp_model_type=sp_model_type,
        sp_character_coverage=sp_character_coverage,
        sp_train_if_missing=sp_train_if_missing,
    )

    total_chars = len(text)
    words = text.split()
    total_words = len(words)
    all_tokens = encode(text)
    total_tokens = len(all_tokens)

    line_counts = []
    for line in text.splitlines():
        line_counts.append(len(encode(line)))

    avg_tokens_per_word = total_tokens / total_words if total_words else 0.0
    avg_tokens_per_char = total_tokens / total_chars if total_chars else 0.0
    avg_tokens_per_line = sum(line_counts) / len(line_counts) if line_counts else 0.0
    max_tokens_per_line = max(line_counts) if line_counts else 0

    unk_rate = None
    if tokenizer_type == "sentencepiece":
        try:
            import sentencepiece as spm
        except ImportError:
            spm = None
        if spm is not None and sp_model_path:
            sp = spm.SentencePieceProcessor()
            sp.load(str(sp_model_path))
            unk_id = sp.unk_id()
            if unk_id >= 0:
                # Unknown token rate is a quick coverage proxy.
                unk_count = sum(1 for tid in all_tokens if tid == unk_id)
                unk_rate = unk_count / total_tokens if total_tokens else 0.0

    print("Tokenizer evaluation")
    if train_dataset_path and val_dataset_path:
        print(f"  train_dataset_path: {train_dataset_path}")
        print(f"  val_dataset_path: {val_dataset_path}")
    else:
        print(f"  dataset_path: {dataset_path}")
    print(f"  tokenizer_type: {tokenizer_type}")
    if tokenizer_type == "sentencepiece":
        print(f"  sp_model_path: {sp_model_path}")
        print(f"  sp_vocab_size: {sp_vocab_size}")
        print(f"  sp_model_type: {sp_model_type}")
        print(f"  sp_character_coverage: {sp_character_coverage}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  total_chars: {total_chars}")
    print(f"  total_words: {total_words}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  avg_tokens_per_word: {avg_tokens_per_word:.4f}")
    print(f"  avg_tokens_per_char: {avg_tokens_per_char:.4f}")
    print(f"  avg_tokens_per_line: {avg_tokens_per_line:.2f}")
    print(f"  max_tokens_per_line: {max_tokens_per_line}")
    if unk_rate is not None:
        print(f"  unk_rate: {unk_rate:.6f}")

    if n_embd is not None:
        embed_params = vocab_size * n_embd
        lm_head_params = vocab_size * n_embd
        print(f"  embedding_params: {embed_params}")
        print(f"  lm_head_params: {lm_head_params}")
        print(f"  vocab_params_total: {embed_params + lm_head_params}")


if __name__ == "__main__":
    main()
