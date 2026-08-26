"""Read-only evaluation-row helpers without census_lib's model side effects."""

from pathlib import Path

import torch


BQ = Path(__file__).resolve().parent.parent / "bilinear_quotient"


def fineweb_rows(n=120, skip=0):
    """Match census_lib.fineweb_rows without importing or loading bilin18."""
    import tiktoken
    from datasets import load_dataset

    encoding = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    reference = torch.load(BQ / "bilin18_eval_tokens_large.pt", map_location="cpu")
    seen = {tuple(reference[row, :32].tolist()) for row in range(reference.shape[0])}
    output = []
    skipped = 0
    for example in dataset:
        if skipped < skip:
            skipped += 1
            continue
        tokens = encoding.encode_ordinary(example["text"])
        for start in range(0, len(tokens) - 513, 513):
            row = tokens[start:start + 513]
            if tuple(row[:32]) in seen:
                continue
            output.append(row)
            if len(output) >= n:
                break
        if len(output) >= n:
            break
    return torch.tensor(output, dtype=torch.long)

