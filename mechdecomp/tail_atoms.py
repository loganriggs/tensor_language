"""Are the irreplaceable TAIL atoms of down_proj interpretable?

Last tick: `down_proj`'s learned dictionary is only mildly irreplaceable on average (1.84x a matched
random basis) but has a heavy tail — its top atom costs 20x the dictionary's own mean. If the method
has value, it should be there.

An "interpretable" claim needs a control, so this measures TOKEN PURITY of an atom's top-activating
positions and compares three groups drawn from the SAME run:

    TAIL   : the most irreplaceable learned atoms
    BULK   : median-irreplaceability learned atoms  (matched by usage where possible)
    RANDOM : atoms of an unlearned random dictionary

purity(atom) = max over tokens of (count of that token among its top-N activating positions) / N
Also reports the entropy of that token distribution, and the top tokens themselves.

If TAIL purity ≈ BULK ≈ RANDOM, then irreplaceability does not buy interpretability, and the tail is
just "atoms that carry more variance". That is a real, reportable negative.

Guard: purity of a random SUBSET of positions (not selected by activation) gives the chance level —
computed, not assumed.

Run: python -m mechdecomp.tail_atoms
"""

import statistics as st
from collections import Counter

import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .irreplaceability import learn, omp_chunked, r2_of
from .release_d import omp_codes

DEV = "cuda"
LAYER = 3
K_ATOMS = 1024
K_SPARSE = 16
N_PROBE = 96        # atoms whose irreplaceability we measure
N_EVAL = 1000       # points for the (expensive) irreplaceability re-solves
N_PURITY = 8000     # separate, larger set for top-activation statistics
TOPN = 40


def collect(model, tok, n_docs=200):
    grabbed, acts, toks = {}, [], []
    mod = model.gpt_neox.layers[LAYER].mlp.dense_4h_to_h
    h = mod.register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", i[0].detach()))
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    with torch.no_grad():
        for _, d in zip(range(n_docs), ds):
            ids = tok(d["text"], return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEV)
            if ids.shape[1] < 32:
                continue
            model(ids); acts.append(grabbed["h"][0]); toks.append(ids[0])
    h.remove()
    return torch.cat(acts).float(), torch.cat(toks), mod.weight.detach().float()


def purity(codes_row, tok_ids, tok, n=TOPN):
    """max-token fraction among the atom's top-n activating positions, + entropy + examples."""
    v = codes_row.abs()
    if int((v > 1e-8).sum()) < n:
        return None
    idx = torch.topk(v, n).indices
    ts = [int(tok_ids[i]) for i in idx]
    c = Counter(ts)
    top, cnt = c.most_common(1)[0]
    tot = sum(c.values())
    ent = -sum((x / tot) * torch.log(torch.tensor(x / tot)).item() for x in c.values())
    ex = [tok.decode([t]) for t, _ in c.most_common(4)]
    return cnt / n, ent, ex


def main():
    name = "EleutherAI/pythia-410m"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    H, T, W = collect(model, tok)
    d_in = W.shape[1]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(H.shape[0], generator=g)
    tr = perm[:10000]; ev = perm[10000:10000 + N_EVAL]
    pu = perm[10000 + N_EVAL:10000 + N_EVAL + N_PURITY]
    Xtr = H[tr].T.contiguous().to(DEV)
    Xev = H[ev].T.contiguous().to(DEV); Tev = T[ev]
    Xpu = H[pu].T.contiguous().to(DEV); Tpu = T[pu]
    sc = Xtr.norm(dim=0).mean() / d_in ** 0.5
    Xtr, Xev, Xpu = Xtr / sc, Xev / sc, Xpu / sc
    Y = W @ Xev
    Ypu = W @ Xpu       # atoms fire on ~k/K of tokens, so purity needs a big set
    den = ((Y - Y.mean(1, keepdim=True)) ** 2).sum()

    Dl = learn(W, Xtr, K_ATOMS, K_SPARSE)
    WD = W @ Dl
    C = omp_chunked(WD, Y, K_SPARSE)
    base = r2_of(WD, C, Y, den)
    print(f"learned dict: base R2 {base:.4f}   K={K_ATOMS} k={K_SPARSE}\n", flush=True)

    Cpu = omp_chunked(WD, Ypu, K_SPARSE)     # codes on the large set, for purity only
    fire = (Cpu.abs() > 1e-8).float().mean(1)
    print(f"mean atom usage {fire.mean():.4f}  -> ~{fire.mean()*N_PURITY:.0f} firings per atom "
          f"in the {N_PURITY}-point purity set\n", flush=True)

    gg = torch.Generator().manual_seed(3)
    cand = torch.randperm(K_ATOMS, generator=gg)[:N_PROBE].tolist()
    print(f"measuring irreplaceability of {N_PROBE} atoms...", flush=True)
    loss = {}
    for j in cand:
        keep = [i for i in range(K_ATOMS) if i != j]
        Ck = omp_chunked(WD[:, keep], Y, K_SPARSE)
        loss[j] = base - r2_of(WD[:, keep], Ck, Y, den)
    ranked = sorted(cand, key=lambda j: -loss[j])
    ms = st.mean(loss.values())
    print(f"  mean loss {ms:.6f}   top1 {loss[ranked[0]]:.6f} ({loss[ranked[0]]/ms:.1f}x its own mean)\n", flush=True)

    tail = ranked[:5]
    mid = len(ranked) // 2
    bulk = ranked[mid - 2:mid + 3]

    grd = torch.Generator().manual_seed(1)
    Dr = Fn.normalize(torch.randn(d_in, K_ATOMS, generator=grd).to(DEV), dim=0)
    Cr = omp_chunked(W @ Dr, Ypu, K_SPARSE)
    rnd_atoms = [j for j in torch.randperm(K_ATOMS, generator=grd)[:60].tolist()
                 if int((Cr[j].abs() > 1e-8).sum()) >= TOPN][:5]

    # chance level: purity of a RANDOM set of positions (not activation-selected)
    ch = []
    for s in range(20):
        gsel = torch.Generator().manual_seed(100 + s)
        idx = torch.randperm(Tpu.shape[0], generator=gsel)[:TOPN]
        c = Counter(int(Tpu[i]) for i in idx)
        ch.append(c.most_common(1)[0][1] / TOPN)
    print(f"CHANCE purity (random positions, measured): {st.mean(ch):.3f}\n", flush=True)

    print(f"{'group':8s} {'atom':>5s} {'irrepl':>9s} {'usage':>7s} {'purity':>7s} {'entropy':>8s}   top tokens", flush=True)
    for tag, atoms, codes in (("TAIL", tail, Cpu), ("BULK", bulk, Cpu), ("RANDOM", rnd_atoms, Cr)):
        ps = []
        for j in atoms:
            r = purity(codes[j], Tpu, tok)
            if r is None:
                print(f"{tag:8s} {j:5d}   (fires < {TOPN} times)"); continue
            p, e, ex = r
            use = float((codes[j].abs() > 1e-8).float().mean())
            lo = loss.get(j, float('nan'))
            ps.append(p)
            print(f"{tag:8s} {j:5d} {lo:9.6f} {use:7.4f} {p:7.3f} {e:8.3f}   {ex}", flush=True)
        if ps:
            print(f"  -> {tag} mean purity {st.mean(ps):.3f}\n", flush=True)
    print("Interpretability requires TAIL purity >> BULK and >> RANDOM and >> chance.", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
