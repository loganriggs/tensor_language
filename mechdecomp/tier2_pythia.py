"""Tier 2 on Pythia-70m: decompose the MLP down-projection at a middle layer over real
Pile activations, with SVD init + overcomplete dictionary. Measures the R²/L0 frontier
and top-activating tokens per atom (interpretability). Spec §2 Tier 2.

W = down_proj.weight (d_model × d_mlp); X = MLP-hidden (post-GELU) inputs to it.

Run: python -m mechdecomp.tier2_pythia
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from .estep import solve_codes
from .mstep import resample_dead, update_dictionary
from .objective import r2
from .tier0 import svd_init

DEV = "cuda"
LAYER = 3
N_TOK = 30_000


def collect(model, tok, n_docs=250):
    acts, toks = [], []
    grabbed = {}

    def hook(mod, inp, out):
        grabbed["h"] = inp[0].detach()          # input to down_proj = MLP hidden

    handle = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.register_forward_hook(hook)
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = tok(d["text"], return_tensors="pt", truncation=True, max_length=128).input_ids.cuda()
            if ids.shape[1] < 8:
                continue
            model(ids)
            acts.append(grabbed["h"][0])
            toks.append(ids[0])
    handle.remove()
    H = torch.cat(acts).float()
    T = torch.cat(toks)
    return H, T


def main():
    name = "EleutherAI/pythia-70m"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    W = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h.weight.detach().float()   # (d_model, d_mlp)
    H, T = collect(model, tok)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(H.shape[0], generator=g)[:N_TOK]
    X = H[idx].T.contiguous()                    # (d_mlp, N)
    Ttok = T[idx]
    print(f"Pythia-70m L{LAYER} down-proj: W {tuple(W.shape)}, X {tuple(X.shape)}", flush=True)

    # baseline: a dense SAE-style linear autoencoder R² ceiling at various L0 is implicit;
    # here trace our method's R²/L0 frontier with SVD init + overcomplete dict.
    print("R²/L0 frontier (SVD init, overcomplete):")
    best = None
    for m, lam in [(1024, 0.02), (1024, 0.005), (2048, 0.01), (2048, 0.003)]:
        torch.cuda.empty_cache()
        D = svd_init(W, X, m)
        C = None
        for _ in range(10):
            C = solve_codes(W, D, X, lam, C0=C)
            D = update_dictionary(W, D, C, X)
            D, _ = resample_dead(W, D, C, X)
        C = solve_codes(W, D, X, lam, C0=C)
        R2 = r2(W, D, C, X)
        L0 = (C.abs() > 1e-8).sum(0).float().mean().item()
        print(f"  m {m} lam {lam}: R2 {R2:.4f} L0 {L0:.1f}", flush=True)
        if best is None or R2 > best[0]:
            best = (R2, m, lam, D.cpu(), C.cpu())

    # interpretability: top-activating tokens for a few atoms of the best run
    R2, m, lam, D, C = best
    D, C = D.cuda(), C.cuda()
    print(f"\nTop tokens per atom (best: m {m} lam {lam} R2 {R2:.3f}):")
    usage = (C.abs() > 1e-8).sum(1)
    live = torch.where(usage > 20)[0][:8]
    for j in live.tolist():
        top = torch.argsort(-C[j].abs())[:6]
        words = [repr(tok.decode([Ttok[t].item()])) for t in top]
        print(f"  atom {j} (support {int(usage[j])}): {' '.join(words)}")


if __name__ == "__main__":
    main()
