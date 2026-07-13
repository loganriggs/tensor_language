"""AUDIT of the only surviving positive result: Pythia down_proj.

Tier 1.5 taught that a high R² gate can be passed by a random dictionary, and that reconstruction
quality says nothing about whether atoms are mechanisms. Pythia's result (learned 0.6020 vs PCA-32
0.3509 vs random-unlearned 0.2587, held-out, matched L0=32) has never faced those controls.

Three tests, in the order that can kill the claim earliest:

  (1) DECOMPOSABILITY PRE-TEST (proposed for the spec). Sweep k, record
        - learned − random R² gap  (does the dictionary matter?)
        - BEHAVIOURAL faithfulness: splice the atom-reconstructed down_proj output back into the
          model and measure LM cross-entropy. R² is not behaviour.
      A map is decomposable only if some k gives BOTH a gap and preserved behaviour.

  (2) LOCALIZATION GATE, causal-ranked. Ablate atoms in order of their MEASURED effect on CE.
      If the learned dictionary localizes a mechanism, its curve must fall faster than random's.

  (3) The reconstruction floors, re-checked at the k chosen by (1).

Everything is held-out: D is frozen, codes re-solved on unseen tokens.

Run: python -m mechdecomp.tier2_audit
"""

import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .release_d import omp_codes
from .tier1_recovery import mstep_gs
from .mstep import rowspace_basis

DEV = "cuda"
LAYER = 3
K_ATOMS = 2048
ROUNDS = 6
N_TRAIN = 12000


def r2_of(Yh, Y):
    return float(1 - ((Yh - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())


def collect(model, tok, n_docs=300, max_len=128):
    grabbed, acts, seqs = {}, [], []
    h = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("h", i[0].detach()))
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = tok(d["text"], return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEV)
            if ids.shape[1] < 32:
                continue
            model(ids)
            acts.append(grabbed["h"][0])
            seqs.append(ids[0])
    h.remove()
    return torch.cat(acts).float(), seqs


@torch.no_grad()
def ce_with_reconstruction(model, seqs, W, D, k, drop=None):
    """Splice the atom-reconstructed down_proj output into the model; return mean LM CE.
    `drop` = atom indices whose contribution is removed from the reconstruction."""
    WD = W @ D
    state = {}

    def hook(mod, inp, out):
        h = inp[0][0].float()                       # (T, d_in)
        C, _ = omp_codes(WD, W @ h.T, k)           # (m, T)
        if drop is not None and len(drop):
            C = C.clone(); C[drop] = 0
        return (WD @ C).T.unsqueeze(0).to(out.dtype)

    hd = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.register_forward_hook(hook)
    tot, n = 0.0, 0
    for ids in seqs[:24]:
        ids = ids.unsqueeze(0)
        out = model(ids, labels=ids)
        tot += float(out.loss) * (ids.shape[1] - 1); n += ids.shape[1] - 1
    hd.remove()
    return tot / n


@torch.no_grad()
def baseline_ce(model, seqs):
    tot, n = 0.0, 0
    for ids in seqs[:24]:
        ids = ids.unsqueeze(0)
        out = model(ids, labels=ids)
        tot += float(out.loss) * (ids.shape[1] - 1); n += ids.shape[1] - 1
    return tot / n


def learn(W, X, K, k, rounds=ROUNDS, seed=0):
    RS = rowspace_basis(W); Wp = torch.linalg.pinv(W)
    Y = W @ X
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
    W = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.weight.detach().float()
    d_out, d_in = W.shape
    H, seqs = collect(model, tok)
    print(f"W {tuple(W.shape)}   activations {tuple(H.shape)}   {len(seqs)} seqs\n", flush=True)

    idx = torch.randperm(H.shape[0], generator=torch.Generator().manual_seed(0))[:N_TRAIN + 4000]
    H = H[idx].to(DEV)
    Xtr, Xva = H[:N_TRAIN].T.contiguous(), H[N_TRAIN:].T.contiguous()
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xva = Xtr / sc, Xva / sc
    Yva = W @ Xva
    ce0 = baseline_ce(model, seqs)
    print(f"baseline LM CE (unmodified model) = {ce0:.4f}\n", flush=True)

    print("(1) DECOMPOSABILITY PRE-TEST — does any k give a dictionary gap AND preserved behaviour?\n", flush=True)
    print("     k   R2 learn  R2 rand    gap     spliced CE   ΔCE vs base", flush=True)
    gg = torch.Generator().manual_seed(1)
    Drnd = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=gg).to(DEV), dim=0)
    keep = {}
    for k in (8, 16, 32, 64):
        Dl = learn(W, Xtr, K_ATOMS, k)
        Cl, _ = omp_codes(W @ Dl, Yva, k); rl = r2_of((W @ Dl) @ Cl, Yva)
        Cr, _ = omp_codes(W @ Drnd, Yva, k); rr = r2_of((W @ Drnd) @ Cr, Yva)
        ce = ce_with_reconstruction(model, seqs, W, Dl * sc, k)
        print(f"   {k:3d}   {rl:.4f}   {rr:.4f}  {rl-rr:+.4f}     {ce:.4f}     {ce-ce0:+.4f}", flush=True)
        keep[k] = (Dl, rl, rr, ce)

    print("\n(2) LOCALIZATION GATE (causal-ranked) at k=32", flush=True)
    k = 32
    Dl = keep[k][0]
    for tag, D in (("LEARNED", Dl), ("RANDOM ", Drnd)):
        WD = W @ (D * sc)
        # measured single-atom effect on CE, over a subsample of atoms for cost
        g = torch.Generator().manual_seed(2)
        cand = torch.randperm(K_ATOMS, generator=g)[:128]
        base_ce = ce_with_reconstruction(model, seqs, W, D * sc, k)
        eff = []
        for j in cand.tolist():
            eff.append((ce_with_reconstruction(model, seqs, W, D * sc, k, drop=[j]) - base_ce, j))
        eff.sort(reverse=True)
        order = [j for _, j in eff]
        print(f"  {tag}: spliced CE {base_ce:.4f}   top single-atom ΔCE {eff[0][0]:+.4f}", flush=True)
        row = []
        for m in (1, 4, 16, 64, 128):
            row.append(ce_with_reconstruction(model, seqs, W, D * sc, k, drop=order[:m]))
        print(f"    atoms dropped :  " + "".join(f"{v:>8d}" for v in (1, 4, 16, 64, 128)), flush=True)
        print(f"    CE            :  " + "".join(f"{v:8.4f}" for v in row), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
