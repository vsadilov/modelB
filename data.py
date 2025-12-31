from pathlib import Path


def load_dataset_texts(
    dataset_path="data/tinyshakespeare.txt",
    train_dataset_path=None,
    val_dataset_path=None,
):
    """Load dataset text for training/validation with optional explicit split files."""
    if train_dataset_path and val_dataset_path:
        train_text = Path(train_dataset_path).read_text(encoding="utf-8")
        val_text = Path(val_dataset_path).read_text(encoding="utf-8")
        return train_text, val_text, train_text + val_text

    text = Path(dataset_path).read_text(encoding="utf-8")
    return text, None, text
