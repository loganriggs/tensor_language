"""Exploration probes on GPT-2 (fast): does the IN-CONTEXT walk distribution move the map,
the way the training distribution moved the toys?

Variants on ring-12 (adjacency for the organization measure is always the undirected ring):
  uniform    : 50% backtrack steps           (baseline, reversible)
  biased71   : forward 7:1                   (12.5% backtracks, low entropy, reversible)
  directed   : always forward                (0 backtracks, zero entropy, irreversible)
  dk2        : +1 or +2 uniformly, never back (1 bit entropy, irreversible; adjacency +1/+2 symmetrized)
Controls on grid45:
  numbers    : node labels are numerals not words (semantic-content probe)
  shuffled   : tokens of a real walk shuffled in time (destroys adjacency info -> org should die)

Usage: python llm_variants.py [model]   (default gpt2) -> runs_llm/<tag>-variants/variants.json
"""

import json
import sys
from pathlib import Path

import torch

from llm_reps import WINDOW, N_STEPS, build_graph, single_token_words

CONTEXTS = (64, 128, 256, 400)
N_WALKS = 96


def ring_walk(n, n_walks, p_forward, k2, generator):
    nodes = torch.empty(n_walks, N_STEPS, dtype=torch.long)
    nodes[:, 0] = torch.randint(n, (n_walks,), generator=generator)
    for t in range(1, N_STEPS):
        if k2:
            step = torch.randint(1, 3, (n_walks,), generator=generator)
        else:
            step = torch.where(torch.rand(n_walks, generator=generator) < p_forward, 1, -1)
        nodes[:, t] = (nodes[:, t - 1] + step) % n
    return nodes


def grid_walk(n_walks, generator):
    nbrs = build_graph("grid45")
    from llm_reps import sample_walks
    return sample_walks(nbrs, n_walks, generator)


def adjacency_ring(n, k2=False):
    A = torch.zeros(n, n)
    for v in range(n):
        A[v, (v + 1) % n] = A[(v + 1) % n, v] = 1
        if k2:
            A[v, (v + 2) % n] = A[(v + 2) % n, v] = 1
    return A


def adjacency_grid():
    nbrs = build_graph("grid45")
    A = torch.zeros(20, 20)
    for v, x in enumerate(nbrs):
        A[v, x] = 1.0
    return A


@torch.no_grad()
def measure(model, ids, walks, A, off):
    n = A.size(0)
    L = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    sums = {t: torch.zeros(L, n, D, device="cuda") for t in CONTEXTS}
    counts = {t: torch.zeros(n, device="cuda") for t in CONTEXTS}
    for b0 in range(0, ids.size(0), 24):
        batch = ids[b0:b0 + 24].cuda()
        bn = walks[b0:b0 + 24].cuda()
        hs = model(batch, output_hidden_states=True).hidden_states
        for t in CONTEXTS:
            lo = max(0, t - WINDOW)
            wn = bn[:, lo:t].reshape(-1)
            counts[t] += torch.bincount(wn, minlength=n).float()
            for l in range(L):
                sums[t][l].index_add_(0, wn, hs[l][:, lo + off:t + off].reshape(-1, D).float())
    offd = ~torch.eye(n, dtype=torch.bool)
    org = torch.zeros(L, len(CONTEXTS))
    for ti, t in enumerate(CONTEXTS):
        H = (sums[t] / counts[t].clamp(min=1).unsqueeze(-1)).cpu()
        for l in range(L):
            Hc = H[l] - H[l].mean(0)
            org[l, ti] = torch.corrcoef(torch.stack([(Hc @ Hc.T)[offd], A[offd]]))[0, 1]
    return org


def main(model_name="gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tag = model_name.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    pool = single_token_words(tokenizer)
    bos = tokenizer.bos_token_id
    off = 1 if bos is not None else 0
    gen = torch.Generator().manual_seed(0)

    def to_ids(walks, node_ids):
        ids = node_ids[walks]
        if off:
            ids = torch.cat([torch.full((ids.size(0), 1), bos, dtype=torch.long), ids], 1)
        return ids

    results = {}
    # ring-12 walk-distribution battery (same fixed word labeling for all variants)
    n = 12
    perm = torch.randperm(len(pool), generator=gen)[:n]
    ring_words = torch.tensor([pool[i][1] for i in perm])
    A1, A2 = adjacency_ring(n), adjacency_ring(n, k2=True)
    for name, (pf, k2, A) in {
        "ring12_uniform": (0.5, False, A1),
        "ring12_biased71": (0.875, False, A1),
        "ring12_directed": (1.0, False, A1),
        "ring12_dk2": (None, True, A2),
    }.items():
        walks = ring_walk(n, N_WALKS, pf, k2, gen)
        org = measure(model, to_ids(walks, ring_words), walks, A, off)
        results[name] = org.tolist()
        best = org[:, -1].argmax().item()
        print(f"{name:18s} last {org[-1, -1]:+.2f}  best L{best} {org[best, -1]:+.2f}")

    # grid45: numerals instead of words
    numerals = []
    for i in range(1, 60):
        t = tokenizer.encode(f" {i}", add_special_tokens=False)
        if len(t) == 1:
            numerals.append(t[0])
    # random numeral->node assignment: grid neighbors must NOT be numerically consecutive
    # (row-major assignment would let GPT-2's number priors fake organization)
    num_ids = torch.tensor(numerals)[torch.randperm(len(numerals), generator=gen)[:20]]
    Ag = adjacency_grid()
    walks = grid_walk(N_WALKS, gen)
    org = measure(model, to_ids(walks, num_ids), walks, Ag, off)
    results["grid45_numbers"] = org.tolist()
    best = org[:, -1].argmax().item()
    print(f"{'grid45_numbers':18s} last {org[-1, -1]:+.2f}  best L{best} {org[best, -1]:+.2f}")

    # grid45 shuffled control: same tokens, time-shuffled (kills transition info)
    perm_g = torch.randperm(len(pool), generator=gen)[:20]
    grid_words = torch.tensor([pool[i][1] for i in perm_g])
    walks = grid_walk(N_WALKS, gen)
    shuf = torch.stack([w[torch.randperm(N_STEPS, generator=gen)] for w in walks])
    org = measure(model, to_ids(shuf, grid_words), shuf, Ag, off)
    results["grid45_shuffled"] = org.tolist()
    best = org[:, -1].argmax().item()
    print(f"{'grid45_shuffled':18s} last {org[-1, -1]:+.2f}  best L{best} {org[best, -1]:+.2f}")

    out = Path("runs_llm") / f"{tag}-variants"
    out.mkdir(parents=True, exist_ok=True)
    (out / "variants.json").write_text(json.dumps({"contexts": CONTEXTS, "org": results}))
    print(f"saved {out}/variants.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpt2")
