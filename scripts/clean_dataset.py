import argparse
import re
from pathlib import Path


CREATED_RE = re.compile(r"^Created on\b", re.IGNORECASE)
CHARS_RE = re.compile(r"^Characters in the Play\b", re.IGNORECASE)
ACT_RE = re.compile(r"^ACT\b", re.IGNORECASE)


def _find_after_created_block(lines):
    for idx, line in enumerate(lines):
        if CREATED_RE.match(line.strip()):
            # Drop through the next blank line after "Created on ..."
            for j in range(idx + 1, len(lines)):
                if lines[j].strip() == "":
                    return j + 1
            return idx + 1
    return 0


def _find_after_characters_block(lines):
    for idx, line in enumerate(lines):
        if CHARS_RE.match(line.strip()):
            # Prefer skipping to the first ACT line if present.
            for j in range(idx + 1, len(lines)):
                if ACT_RE.match(lines[j].strip()):
                    return j
            # Otherwise skip to the next blank line.
            for j in range(idx + 1, len(lines)):
                if lines[j].strip() == "":
                    return j + 1
            return idx + 1
    return None


def clean_text(text):
    """Remove header/character blocks from Folger Shakespeare texts."""
    lines = text.splitlines()

    start_idx = _find_after_characters_block(lines)
    if start_idx is None:
        start_idx = _find_after_created_block(lines)

    cleaned = lines[start_idx:]

    # Trim leading blank lines for cleanliness.
    while cleaned and cleaned[0].strip() == "":
        cleaned = cleaned[1:]

    return "\n".join(cleaned).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Clean Folger Shakespeare text headers.")
    parser.add_argument("--input-dir", default="data/raw/full_list")
    parser.add_argument("--output-dir", default="data/clean/full_list_clean")
    parser.add_argument("--pattern", default="*.txt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {input_dir} matching {args.pattern}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        cleaned = clean_text(text)
        out_path = output_dir / path.name
        out_path.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned {len(files)} files into {output_dir}")


if __name__ == "__main__":
    main()
