"""Natural-text pipeline for the deeper-circuits program (PLAN.md).

Corpus is selected by the TL_CORPUS env var (default "owt"):

  owt   OpenWebText (Skylion007/openwebtext, streamed), byte-level BPE V=5120
        ("reduced 5k vocab") -> data_owt/, runs_owt/. Decision 2026-07-08 (Logan):
        TinyStories' statistics are too learnable to force induction; OWT's
        diversity/burstiness is the standard induction-eliciting distribution.
  tiny  TinyStories, V=1024 -> data_text/, runs_lm/ (the session-1 corpus, kept for
        the completed analyses).

Layout per corpus: <data>/tokenizer.json, train.bin, val.bin (uint16 memmaps, one doc
per <|eot|>=0). The frozen val stream is the shared datapoint set: every model is scored
on per-token CE over the same non-overlapping N_CTX+1 windows (lm_eval.py).

Build (one-off):   python text_data.py build      (~30-60 min for owt: streams ~570k docs)
"""

import os
import sys
from pathlib import Path

import numpy as np

CORPUS = os.environ.get("TL_CORPUS", "owt")
CFG = {
    "owt": dict(vocab=5120, hf="Skylion007/openwebtext", data="data_owt", runs="runs_owt",
                val_docs=15_000, train_docs=560_000, tok_sample_docs=60_000),
    "tiny": dict(vocab=1024, hf="roneneldan/TinyStories", data="data_text", runs="runs_lm",
                 val_docs=None, train_docs=None, tok_sample_docs=None),
}[CORPUS]

VOCAB = CFG["vocab"]
EOT = 0
N_CTX = 256
DATA = Path(CFG["data"])
RUNS = Path(CFG["runs"])


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(DATA / "tokenizer.json"))


def _new_bpe(sample_iter):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|eot|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(sample_iter, trainer)
    tok.save(str(DATA / "tokenizer.json"))
    return tok


def build_owt():
    """Single pass over the streamed dataset: buffer the first tok_sample_docs docs to
    train the BPE, then tokenize val (first val_docs) + train (next train_docs) to bins."""
    from datasets import load_dataset

    DATA.mkdir(exist_ok=True)
    stream = load_dataset(CFG["hf"], split="train", streaming=True)
    it = (d["text"] for d in stream)

    buffer = []
    for t in it:
        buffer.append(t)
        if len(buffer) >= CFG["tok_sample_docs"]:
            break
    print(f"buffered {len(buffer)} docs for tokenizer", flush=True)
    tok = _new_bpe(buffer) if not (DATA / "tokenizer.json").exists() else load_tokenizer()
    print("tokenizer ready", flush=True)

    def encode_to(path, texts_iter, n_docs):
        total_tok, done = 0, 0
        with open(path, "wb") as f:
            batch = []
            for t in texts_iter:
                batch.append(t)
                done += 1
                if len(batch) == 10_000 or done == n_docs:
                    for e in tok.encode_batch(batch):
                        arr = np.array(e.ids + [EOT], dtype=np.uint16)
                        total_tok += arr.size
                        arr.tofile(f)
                    batch = []
                    print(f"  {path.name}: {done}/{n_docs} docs, {total_tok/1e6:.0f}M tokens",
                          flush=True)
                if done == n_docs:
                    break
        print(f"{path.name}: {total_tok} tokens", flush=True)

    from itertools import chain
    full = chain(buffer, it)                       # buffered docs first, then the rest
    encode_to(DATA / "val.bin", full, CFG["val_docs"])
    encode_to(DATA / "train.bin", full, CFG["train_docs"])


def tokens(split: str) -> np.ndarray:
    return np.memmap(DATA / f"{split}.bin", dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, batch: int, gen, device="cuda"):
    """Random contiguous (batch, N_CTX+1) windows; x = [:, :-1], targets = [:, 1:]."""
    import torch
    ix = torch.randint(len(data) - N_CTX - 1, (batch,), generator=gen)
    buf = np.stack([data[i:i + N_CTX + 1] for i in ix.tolist()]).astype(np.int64)
    return torch.from_numpy(buf).to(device)


def val_windows():
    """The frozen eval windows: non-overlapping (N_CTX+1)-token chunks of val.bin."""
    data = tokens("val")
    n_win = (len(data) - 1) // N_CTX
    return data, n_win


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        if CORPUS == "owt":
            build_owt()
        else:
            sys.exit("tiny corpus already built (data_text/); TL_CORPUS=owt for the new one")
    else:
        print(__doc__)
