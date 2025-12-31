import argparse
import random
import re
from pathlib import Path


ACT_RE = re.compile(r"^ACT\b", re.IGNORECASE)
SCENE_RE = re.compile(r"^SCENE\b", re.IGNORECASE)
SCENE_NUM_RE = re.compile(r"^Scene\b", re.IGNORECASE)
DIVIDER_RE = re.compile(r"^=+$")


def _split_into_blocks(text):
    lines = text.splitlines(keepends=True)
    blocks = []
    current = []
    for line in lines:
        if ACT_RE.match(line.strip()) or SCENE_RE.match(line.strip()):
            if current:
                blocks.append("".join(current))
                current = []
        current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks


def _filter_structure_lines(text):
    filtered = []
    blank_streak = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if ACT_RE.match(stripped) or SCENE_RE.match(stripped) or SCENE_NUM_RE.match(stripped):
            continue
        if DIVIDER_RE.match(stripped):
            continue
        if stripped == "":
            blank_streak += 1
            if blank_streak > 1:
                continue
        else:
            blank_streak = 0
        filtered.append(line)
    return "".join(filtered)


def _split_fallback(text, rng, val_ratio):
    """Fallback split when no scene/act markers are available."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return "", ""
    val_count = max(1, int(len(lines) * val_ratio))
    start = rng.randrange(0, max(1, len(lines) - val_count + 1))
    val_lines = lines[start : start + val_count]
    train_lines = lines[:start] + lines[start + val_count :]
    return _filter_structure_lines("".join(train_lines)), _filter_structure_lines("".join(val_lines))


def split_text(text, rng, val_ratio):
    """Split by scene/act blocks, targeting val_ratio of characters."""
    blocks = _split_into_blocks(text)
    if len(blocks) <= 1:
        return _split_fallback(text, rng, val_ratio)

    total_chars = sum(len(b) for b in blocks)
    target_chars = max(1, int(total_chars * val_ratio))
    indices = list(range(len(blocks)))
    rng.shuffle(indices)

    val_indices = set()
    val_chars = 0
    for idx in indices:
        if val_chars >= target_chars:
            break
        val_indices.add(idx)
        val_chars += len(blocks[idx])

    train_parts = []
    val_parts = []
    for idx, block in enumerate(blocks):
        if idx in val_indices:
            val_parts.append(_filter_structure_lines(block))
        else:
            train_parts.append(_filter_structure_lines(block))
    return "".join(train_parts), "".join(val_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Split each file into train/val using scene/act boundaries."
    )
    parser.add_argument("--input-dir", default="data/clean/full_list_clean")
    parser.add_argument("--train-out", default="data/train.txt")
    parser.add_argument("--val-out", default="data/val.txt")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {input_dir} matching {args.pattern}")

    rng = random.Random(args.seed)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    train_chunks = []
    val_chunks = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        train_text, val_text = split_text(text, rng, args.val_ratio)
        if train_text:
            train_chunks.append(train_text.rstrip() + "\n")
        if val_text:
            val_chunks.append(val_text.rstrip() + "\n")

    train_text = "\n".join(train_chunks).rstrip() + "\n"
    val_text = "\n".join(val_chunks).rstrip() + "\n"

    # Remove exact line overlap across splits for a stricter evaluation set.
    train_lines = {line.strip() for line in train_text.splitlines() if line.strip()}
    filtered_val_lines = []
    for line in val_text.splitlines():
        if line.strip() and line.strip() in train_lines:
            continue
        filtered_val_lines.append(line)
    val_text = "\n".join(filtered_val_lines).rstrip() + "\n"

    train_out.write_text(train_text, encoding="utf-8")
    val_out.write_text(val_text, encoding="utf-8")

    print(f"Wrote {train_out} and {val_out}")


if __name__ == "__main__":
    main()
