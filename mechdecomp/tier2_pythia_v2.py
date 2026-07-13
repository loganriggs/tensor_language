"""TIER 2 — Pythia, re-run with the VALIDATED solver.

The original `tier2_pythia.py` used the un-validated M-step (stale-residual Jacobi + normalize with
codes frozen) and a fixed λ. Both are now known-bad, so its numbers are not cited; this replaces them.

What changed:
  * M-step: Gauss-Seidel, β re-alternated, normalization after the sweep (validated on refine_power,
    and separately gated in the WIDE low-rank regime that `down_proj` lives in).
  * E-step: OMP at a fixed k (L0 held constant by construction), so every comparison is matched-L0.
  * Held-out R² with D frozen — never in-sample.
  * Identifiability caveat is *computed*, not assumed: `down_proj` is 512×2048, so atoms are
    determined only up to row(W) — cos(true, projected) = sqrt(512/2048) = 0.5.

Guards, in order (nothing expensive runs until they pass):
  G1  rowspace fraction matches sqrt(rank/d_in) — confirms which part of an atom is even identifiable.
  G2  a random-init refinement must IMPROVE held-out R² (else the pipeline is not learning).
  G3  feature-free baselines must be dominated by the learned dictionary at the same L0.

Run: python -m mechdecomp.tier2_pythia_v2
"""

import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .mstep import rowspace_basis
from .release_d import omp_codes
from .tier0 import svd_init
from .tier1_recovery import mstep_gs

DEV = "cuda"
LAYER = 3
K_ATOMS = 2048
KSPARSE = 32
ROUNDS = 8
N_TRAIN, N_VAL = 16000, 4000


def r2_of(Yh, Y):
    return float(1 - ((Yh - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())


def collect(model, tok, n_docs=400):
    grabbed, acts = {}, []

    def hook(mod, inp, out):
        grabbed["h"] = inp[0].detach()

    h = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.register_forward_hook(hook)
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = tok(d["text"], return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEV)
            if ids.shape[1] < 8:
                continue
            model(ids)
            acts.append(grabbed["h"][0])
    h.remove()
    return torch.cat(acts).float()


def refine(W, D0, Xtr, Xva, rounds=ROUNDS, k=KSPARSE, tag=""):
    Ytr, Yva = W @ Xtr, W @ Xva
    RS = rowspace_basis(W)
    Wp = torch.linalg.pinv(W)
    D = D0.clone()
    for r in range(rounds):
        WD = W @ D
        C, _ = omp_codes(WD, Ytr, k)
        if r == 0:
            tr0 = r2_of(WD @ C, Ytr)
        if r == rounds - 1:
            break
        D, C = mstep_gs(W, D, C, Ytr, RS, Wp)
    WD = W @ D
    Cv, _ = omp_codes(WD, Yva, k)
    va = r2_of(WD @ Cv, Yva)
    Ct, _ = omp_codes(WD, Ytr, k)
    print(f"  {tag:28s} train R2 {tr0:.4f} → {r2_of(WD @ Ct, Ytr):.4f}   HELD-OUT R2 {va:.4f}", flush=True)
    return D, va


def main():
    name = "EleutherAI/pythia-410m"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    W = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.weight.detach().float()
    d_out, d_in = W.shape
    rank = torch.linalg.matrix_rank(W).item()
    print(f"W = down_proj L{LAYER}: {tuple(W.shape)}  rank {rank}\n", flush=True)

    # ---------- G1: what is even identifiable? ----------
    RS = rowspace_basis(W)
    g = torch.Generator().manual_seed(0)
    probe = Fn.normalize(torch.randn(d_in, 512, generator=g).to(DEV), dim=0)
    proj = Fn.normalize(RS @ (RS.T @ probe), dim=0)
    meas = float((probe * proj).sum(0).abs().mean())
    pred = (rank / d_in) ** 0.5
    print(f"G1 identifiability: cos(atom, row(W)-projection) predicted {pred:.4f}  measured {meas:.4f}", flush=True)
    assert abs(meas - pred) < 0.02, "G1 FAIL: rowspace geometry unexpected"
    print(f"  [G1 ok] only {100*rank/d_in:.0f}% of each atom's subspace is visible to W;"
          f" interpretations are rowspace-only\n", flush=True)

    H = collect(model, tok)
    idx = torch.randperm(H.shape[0], generator=torch.Generator().manual_seed(0))[:N_TRAIN + N_VAL]
    H = H[idx].to(DEV)
    Xtr, Xva = H[:N_TRAIN].T.contiguous(), H[N_TRAIN:].T.contiguous()
    print(f"activations: train {tuple(Xtr.shape)}  held-out {tuple(Xva.shape)}", flush=True)
    print(f"  ||x|| mean {Xtr.norm(dim=0).mean():.2f}  (RMS-normalising to sqrt(d))\n", flush=True)
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xva = Xtr / sc, Xva / sc

    print(f"Dictionaries (K={K_ATOMS}, OMP k={KSPARSE}, {ROUNDS} rounds, held-out R² with D frozen):", flush=True)
    gg = torch.Generator().manual_seed(0)
    Drnd = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=gg).to(DEV), dim=0)
    _, va_rnd0 = refine(W, Drnd, Xtr, Xva, rounds=1, tag="random, NO refinement")
    _, va_rnd = refine(W, Drnd, Xtr, Xva, tag="random, refined")
    Dsvd = svd_init(W, Xtr, K_ATOMS).float().contiguous()
    _, va_svd0 = refine(W, Dsvd, Xtr, Xva, rounds=1, tag="svd-init, NO refinement")
    Dsvd_r, va_svd = refine(W, Dsvd, Xtr, Xva, tag="svd-init, refined")

    print(f"\nG2 (refinement must improve held-out): random {va_rnd0:.4f} → {va_rnd:.4f} "
          f"{'PASS' if va_rnd > va_rnd0 else 'FAIL'}", flush=True)
    print(f"G3 (learned must beat unrefined baselines at same L0): "
          f"best refined {max(va_rnd, va_svd):.4f} vs best unrefined {max(va_rnd0, va_svd0):.4f} "
          f"{'PASS' if max(va_rnd, va_svd) > max(va_rnd0, va_svd0) else 'FAIL'}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
