from pathlib import Path


def build_vocab(
    text,
    tokenizer_type="char",
    vocab_chars=None,
    sp_model_path=None,
    sp_vocab_size=8000,
    sp_model_type="bpe",
    sp_character_coverage=1.0,
    sp_train_if_missing=False,
):
    """Return vocab size and encode/decode callables for the selected tokenizer."""
    if tokenizer_type == "char":
        chars = vocab_chars if vocab_chars is not None else sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(s):
            return [stoi[c] for c in s]

        def decode(l):
            return ''.join([itos[i] for i in l])

        return vocab_size, encode, decode

    if tokenizer_type != "sentencepiece":
        raise ValueError(f'Unknown tokenizer_type: {tokenizer_type}')

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            'sentencepiece is required for tokenizer_type="sentencepiece". '
            "Install it with: pip install sentencepiece"
        ) from exc

    if not sp_model_path:
        raise ValueError("sp_model_path is required for sentencepiece tokenization.")

    sp_model_path = Path(sp_model_path)
    if not sp_model_path.exists():
        if not sp_train_if_missing:
            raise FileNotFoundError(
                f"SentencePiece model not found at {sp_model_path}. "
                "Set sp_train_if_missing=True to train it."
            )
        sp_model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_text_path = sp_model_path.with_suffix(".txt")
        temp_text_path.write_text(text, encoding="utf-8")
        model_prefix = sp_model_path.with_suffix("").as_posix()
        spm.SentencePieceTrainer.Train(
            input=str(temp_text_path),
            model_prefix=model_prefix,
            vocab_size=sp_vocab_size,
            model_type=sp_model_type,
            character_coverage=sp_character_coverage,
        )
        temp_text_path.unlink(missing_ok=True)
        if not sp_model_path.exists():
            raise FileNotFoundError(
                f"Expected SentencePiece model at {sp_model_path} after training."
            )

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    vocab_size = sp.get_piece_size()

    def encode(s):
        return sp.encode(s, out_type=int)

    def decode(l):
        return sp.decode(l)

    return vocab_size, encode, decode
