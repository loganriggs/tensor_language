"""IRREPLACEABILITY on real maps — the test that decides whether atoms are mechanisms.

Every previous localization result used single-atom ablation, which the Tier-1 toy showed cannot
distinguish a KNOWN-TRUE dictionary from a random one (1.54x mean, 1.00x top1). Irreplaceability —
drop the atom from the DICTIONARY and let OMP re-select codes — separates them by 8.4x, and by
34x under a matched-R2 control (random dicts get MORE replaceable as k rises, so the confound works
against the criterion, not for it).

Reference numbers (Tier-1 toy, logs above):
    true dict   mean loss 0.00253  top1 0.00368   loss/base 0.00255
    random      mean loss 0.00030  top1 0.00084   loss/base 0.00051      -> 8.4x / 4.4x
    random @ matched R2 (k=24)     mean loss 0.00007                     -> 34x

Question here: on real maps, does the LEARNED dictionary look like the true dict (irreplaceable
atoms = generative factors) or like the random dict (replaceable = arbitrary basis)?

Run: python -m mechdecomp.irreplaceability
"""

import statistics as st

import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .mstep import rowspace_basis
from .release_d import omp_codes
from .tier1_recovery import mstep_gs

DEV = "cuda"
LAYER = 3
K_ATOMS = 1024
K_SPARSE = 16
N_SAMPLE = 40          # atoms probed per dictionary
N_EVAL = 1200          # held-out points
ROUNDS = 6


def r2_of(WD, C, Y, den):
    return float(1 - ((WD @ C - Y) ** 2).sum() / den)


def omp_chunked(WD, Y, k, chunk=400):
    return torch.cat([omp_codes(WD, Y[:, i:i + chunk].contiguous(), k)[0]
                      for i in range(0, Y.shape[1], chunk)], 1)


def irreplaceability(W, D, Y, k, tag, n_sample=N_SAMPLE, seed=3):
    den = ((Y - Y.mean(1, keepdim=True)) ** 2).sum()
    WD = W @ D
    C = omp_chunked(WD, Y, k)
    base = r2_of(WD, C, Y, den)
    K = D.shape[1]
    g = torch.Generator().manual_seed(seed)
    cand = torch.randperm(K, generator=g)[:n_sample].tolist()
    losses = []
    for j in cand:
        keep = [i for i in range(K) if i != j]
        WDk = WD[:, keep]
        Ck = omp_chunked(WDk, Y, k)
        losses.append(base - r2_of(WDk, Ck, Y, den))
    m, t = st.mean(losses), max(losses)
    print(f"  {tag:34s} k={k:3d}  base R2 {base:.4f}  mean loss {m:.6f}  top1 {t:.6f}  loss/base {m/max(base,1e-9):.6f}",
          flush=True)
    return base, m, t


def learn(W, X, K, k, rounds=ROUNDS, seed=0):
    RS = rowspace_basis(W); Wp = torch.linalg.pinv(W); Y = W @ X
    g = torch.Generator().manual_seed(seed)
    D = Fn.normalize(torch.randn(X.shape[0], K, generator=g).to(X.device), dim=0)
    for _ in range(rounds):
        C, _ = omp_codes(W @ D, Y, k)
        D, C = mstep_gs(W, D, C, Y, RS, Wp)
    return D


def pythia_map(model, which):
    L = model.gpt_neox.layers[LAYER]
    return (L.mlp.dense_4h_to_h if which == "mlp" else L.attention.dense)


def collect(model, tok, mod, n_docs=160):
    grabbed, acts = {}, []
    h = mod.register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", i[0].detach()))
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = tok(d["text"], return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEV)
            if ids.shape[1] < 32:
                continue
            model(ids); acts.append(grabbed["h"][0])
    h.remove()
    return torch.cat(acts).float()


def run_map(W, X, name):
    d_in = W.shape[1]
    idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))
    Xtr = X[idx[:10000]].T.contiguous().to(DEV)
    Xva = X[idx[10000:10000 + N_EVAL]].T.contiguous().to(DEV)
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xva = Xtr / sc, Xva / sc
    Y = W @ Xva
    print(f"\n=== {name}  W {tuple(W.shape)}  K={K_ATOMS}  k={K_SPARSE} ===", flush=True)
    Dl = learn(W, Xtr, K_ATOMS, K_SPARSE)
    bl, ml, tl = irreplaceability(W, Dl, Y, K_SPARSE, "LEARNED")
    g = torch.Generator().manual_seed(1)
    Dr = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=g).to(DEV), dim=0)
    br, mr, tr = irreplaceability(W, Dr, Y, K_SPARSE, "RANDOM, same k")
    print(f"    unmatched ratio: mean {ml/max(mr,1e-12):.2f}x   top1 {tl/max(tr,1e-12):.2f}x"
          f"   (base R2 {bl:.3f} vs {br:.3f})", flush=True)
    # matched-R2 control: raise random's k until its base R2 reaches the learned dict's
    for k2 in (32, 64, 128, 256):
        b2, m2, t2 = irreplaceability(W, Dr, Y, k2, f"RANDOM, k raised")
        if b2 >= bl - 0.02:
            print(f"    MATCHED-R2 ratio: mean {ml/max(m2,1e-12):.2f}x   top1 {tl/max(t2,1e-12):.2f}x", flush=True)
            break
    else:
        print(f"    random never matched learned R2; loss/base ratio "
              f"{(ml/bl)/max(m2/max(b2,1e-9),1e-12):.2f}x", flush=True)


def main():
    print("TOY REFERENCE: true 8.4x (mean) / 4.4x (top1) unmatched; 34x matched.\n", flush=True)
    name = "EleutherAI/pythia-410m"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    for which, label in (("mlp", "pythia L3 down_proj"), ("attn", "pythia L3 attention.dense")):
        mod = pythia_map(model, which)
        X = collect(model, tok, mod)
        run_map(mod.weight.detach().float(), X, label)
        del X; torch.cuda.empty_cache()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
