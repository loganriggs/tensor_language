"""Cyclic token sequences: sample L distinct tokens, tile them to fill the context.

The next token is fully determined once one full cycle has been seen, so the
loss/accuracy mask keeps only target positions >= L (the second cycle onward).
"""

import torch

N_VOCAB = 100
N_CTX = 96
TRAIN_LENGTHS = (5, 20)          # inclusive range sampled uniformly at train time
EVAL_LENGTHS = (5, 10, 15, 20, 25, 30)   # 25 and 30 are never seen in training


def sample_cycles(n_seq: int, lengths: torch.Tensor, n_ctx: int = N_CTX,
                  n_vocab: int = N_VOCAB, generator: torch.Generator | None = None):
    """Sequences (n_seq, n_ctx) tiling a random cycle of `lengths[i]` distinct tokens."""
    perm = torch.rand(n_seq, n_vocab, generator=generator).argsort(dim=1)
    phase = torch.arange(n_ctx)[None, :] % lengths[:, None]
    return perm.gather(1, phase)


def train_batch(n_seq: int, generator: torch.Generator | None = None):
    """Tokens plus a mask selecting the determined targets (positions >= L)."""
    lengths = torch.randint(TRAIN_LENGTHS[0], TRAIN_LENGTHS[1] + 1, (n_seq,), generator=generator)
    tokens = sample_cycles(n_seq, lengths, generator=generator)
    return tokens, target_mask(lengths)


def target_mask(lengths: torch.Tensor, n_ctx: int = N_CTX):
    """Mask over target positions 1..n_ctx-1: True where the target index >= L."""
    return torch.arange(1, n_ctx)[None, :] >= lengths[:, None]


def eval_sets(n_seq: int = 512, seed: int = 1234):
    """Fixed held-out cycles for each eval length."""
    generator = torch.Generator().manual_seed(seed)
    sets = {}
    for length in EVAL_LENGTHS:
        lengths = torch.full((n_seq,), length)
        sets[length] = (sample_cycles(n_seq, lengths, generator=generator), target_mask(lengths))
    return sets
