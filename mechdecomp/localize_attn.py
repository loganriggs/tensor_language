"""Does Pythia's attention output-projection LOCALIZE, or does it merely reconstruct well?

This decides the method's scope. Last tick: `attention.dense` has a large learned−random R² gap
(+0.606), which refuted "attention maps are not decomposable". But Tier 1.5 taught that a large
reconstruction gap does not imply the atoms carry the mechanism — the tiny OV map had a (small) gap
at K/rank=1 and localized nothing (atoms-to-halve = K/4 for learned AND random).

So run the SAME three gates that `down_proj` passed, at MATCHED K/rank = 2 (K=2048, rank 1024):

  (1) learned − random R² gap at several k
  (2) BEHAVIOURAL faithfulness: splice the reconstruction back in, read LM cross-entropy,
      calibrated against zeroing the module entirely
  (3) LOCALIZATION, causal-ranked: measured single-atom ΔCE, learned vs random

down_proj reference (logs/tier2_audit.log, logs/ce_scale.log):
    gap +0.4952 @k=8 | zeroed ΔCE +0.2247 | spliced +0.0486 (78.4% recovered) | top atom +0.0112 vs +0.0009

Run: python -m mechdecomp.localize_attn
"""

import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .mstep import rowspace_basis
from .release_d import omp_codes
from .tier1_recovery import mstep_gs

DEV = "cuda"
LAYER = 3
K_ATOMS = 2048          # rank(attention.dense) = 1024  ->  K/rank = 2, matching the down_proj audit
ROUNDS = 6
N_TRAIN = 12000
N_SEQ = 24


def r2_of(Yh, Y):
    return float(1 - ((Yh - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())


def module_of(model):
    return model.gpt_neox.layers[LAYER].attention.dense


def collect(model, tok, n_docs=300):
    grabbed, acts, seqs = {}, [], []
    h = module_of(model).register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", i[0].detach()))
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = tok(d["text"], return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEV)
            if ids.shape[1] < 32:
                continue
            model(ids)
            acts.append(grabbed["h"][0]); seqs.append(ids[0])
    h.remove()
    return torch.cat(acts).float(), seqs


@torch.no_grad()
def ce_of(model, seqs, hook=None):
    hd = module_of(model).register_forward_hook(hook) if hook else None
    tot, n = 0.0, 0
    for ids in seqs[:N_SEQ]:
        ids = ids.unsqueeze(0)
        out = model(ids, labels=ids)
        tot += float(out.loss) * (ids.shape[1] - 1); n += ids.shape[1] - 1
    if hd:
        hd.remove()
    return tot / n


def splice_hook(W, D, k, drop=None):
    WD = W @ D

    def hook(mod, inp, out):
        h = inp[0][0].float()
        C, _ = omp_codes(WD, W @ h.T, k)
        if drop is not None and len(drop):
            C = C.clone(); C[drop] = 0
        return (WD @ C).T.unsqueeze(0).to(out.dtype)
    return hook


def learn(W, X, K, k, rounds=ROUNDS, seed=0):
    RS = rowspace_basis(W); Wp = torch.linalg.pinv(W); Y = W @ X
    g = torch.Generator().manual_seed(seed)
    D = Fn.normalize(torch.randn(X.shape[0], K, generator=g).to(X.device), dim=0)
    for _ in range(rounds):
        C, _ = omp_codes(W @ D, Y, k)
        D, C = mstep_gs(W, D, C, Y, RS, Wp)
    return D


def main():
    name = "EleutherAI/pythia-410m"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    W = module_of(model).weight.detach().float()
    d_out, d_in = W.shape
    rank = torch.linalg.matrix_rank(W).item()
    print(f"W = L{LAYER} attention.dense {tuple(W.shape)}  rank {rank}  K={K_ATOMS}  K/rank={K_ATOMS/rank:.2f}", flush=True)
    print(f"(down_proj audit used K/rank = 2048/1024 = 2.00 — matched)\n", flush=True)

    H, seqs = collect(model, tok)
    idx = torch.randperm(H.shape[0], generator=torch.Generator().manual_seed(0))[:N_TRAIN + 4000]
    H = H[idx].to(DEV)
    Xtr, Xva = H[:N_TRAIN].T.contiguous(), H[N_TRAIN:].T.contiguous()
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xva = Xtr / sc, Xva / sc
    Yva = W @ Xva

    ce0 = ce_of(model, seqs)
    ce_zero = ce_of(model, seqs, hook=lambda m, i, o: torch.zeros_like(o))
    print(f"baseline CE {ce0:.4f}   |   attention.dense output ZEROED: CE {ce_zero:.4f}  ΔCE {ce_zero-ce0:+.4f}")
    print(f"  (down_proj zeroed cost +0.2247 — this module is worth {ce_zero-ce0:.4f} nats)\n", flush=True)

    gg = torch.Generator().manual_seed(1)
    Drnd = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=gg).to(DEV), dim=0)

    print("(1)+(2) gap and BEHAVIOURAL faithfulness:", flush=True)
    print("     k   R2 learn  R2 rand    gap    spliced CE(L)  ΔCE   %recovered   spliced CE(R)  %rec", flush=True)
    keep = {}
    for k in (8, 32):
        Dl = learn(W, Xtr, K_ATOMS, k)
        Cl, _ = omp_codes(W @ Dl, Yva, k); rl = r2_of((W @ Dl) @ Cl, Yva)
        Cr, _ = omp_codes(W @ Drnd, Yva, k); rr = r2_of((W @ Drnd) @ Cr, Yva)
        cel = ce_of(model, seqs, hook=splice_hook(W, Dl * sc, k))
        cer = ce_of(model, seqs, hook=splice_hook(W, Drnd * sc, k))
        recl = 100 * (1 - (cel - ce0) / (ce_zero - ce0))
        recr = 100 * (1 - (cer - ce0) / (ce_zero - ce0))
        print(f"   {k:3d}   {rl:.4f}   {rr:.4f}  {rl-rr:+.4f}    {cel:.4f}  {cel-ce0:+.4f}   {recl:5.1f}%      "
              f"{cer:.4f}   {recr:5.1f}%", flush=True)
        keep[k] = Dl

    print("\n(3) LOCALIZATION, causal-ranked at k=32 (128 sampled atoms per dictionary):", flush=True)
    k = 32
    for tag, D in (("LEARNED", keep[k]), ("RANDOM ", Drnd)):
        base = ce_of(model, seqs, hook=splice_hook(W, D * sc, k))
        g = torch.Generator().manual_seed(2)
        cand = torch.randperm(K_ATOMS, generator=g)[:128].tolist()
        eff = sorted(((ce_of(model, seqs, hook=splice_hook(W, D * sc, k, drop=[j])) - base, j) for j in cand),
                     reverse=True)
        order = [j for _, j in eff]
        row = [ce_of(model, seqs, hook=splice_hook(W, D * sc, k, drop=order[:m])) for m in (1, 4, 16, 64, 128)]
        print(f"  {tag}: spliced CE {base:.4f}   top single-atom ΔCE {eff[0][0]:+.4f}", flush=True)
        print(f"    atoms dropped:  " + "".join(f"{v:>8d}" for v in (1, 4, 16, 64, 128)), flush=True)
        print(f"    CE           :  " + "".join(f"{v:8.4f}" for v in row), flush=True)
    print("\n  down_proj reference: top single-atom ΔCE  learned +0.0112  vs  random +0.0009  (12x)", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
