"""C1c follow-up (same plan, CIRCUITS_PLAN_V4.md, per-block breakdown): which blocks' unit cross terms carry the closed-form
prediction, and which units within them. No fitting.

  python scripts/run_circuits_c1c_blocks.py --smoke
  python scripts/run_circuits_c1c_blocks.py --device cuda --c1a results/circuits_c1a.json --out results/circuits_c1c_blocks.json
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import vjp, jacrev
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for
from scripts.run_circuits_c1c import make_runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--contexts", type=int, default=8); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--c1a", default="results/circuits_c1a.json"); ap.add_argument("--n-tokens", type=int, default=16); ap.add_argument("--out", default="results/circuits_c1c_blocks.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.contexts = 2; a.sequence_length = 8; a.chunk = 8; a.n_tokens = 2
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); readers = F.normalize(torch.randn(cfg.n_embd, 2), dim=0)
        held_rows = torch.randint(0, 64, (2, a.sequence_length), generator=torch.Generator().manual_seed(20)); toks = [3, 5]
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device)
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu"); held_rows = rows[96:96 + a.contexts, :a.sequence_length].long()
        toks = json.load(open(a.c1a))["tokens"][:a.n_tokens]
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; T = a.sequence_length; positions = [T - 3, T - 2, T - 1]
    W = model.lm_head.weight.float(); tokdirs = F.normalize(W[toks].T, dim=0)
    U = torch.cat([readers.to(dev), tokdirs.to(dev)], 1); names = [f"reader{k}" for k in range(readers.shape[1])] + [f"tok{t}" for t in toks]; K = U.shape[1]
    layers = list(range(a.source_layer, n_layer)); run = make_runner(model, a.source_layer, n_layer, d)
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(held_rows[i:i + 1]) for i in range(len(held_rows))]; zero = torch.zeros(d, device=dev)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        run(ctxs[0], zero)
    mlpw = {L: (model.transformer.h[L].mlp.Left.weight.float(), model.transformer.h[L].mlp.Right.weight.float()) for L in layers}
    rec = []; unit_energy = collections.defaultdict(list); t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            Hc = full_hessian(lambda th: U.T @ run(ctx, th), d, dev, a.chunk); Bc = 0.5 * (Hc + Hc.transpose(1, 2)); nB = Bc.flatten(1).square().sum(1)
            per_block = {L: torch.zeros(K, d, d, device=dev) for L in layers}; unit_e = {L: torch.zeros(K, mlpw[L][0].shape[0], device=dev) for L in layers}
            for L in layers:
                A_, B_ = mlpw[L]
                for i in positions:
                    def fout(delta, L=L, i=i): return U.T @ run(ctx, zero, hidden_add=(L, i, delta))
                    _, pullw = vjp(fout, torch.zeros(A_.shape[0], device=dev)); wts = torch.stack([pullw(e)[0] for e in torch.eye(K, device=dev)])   # [K, width]
                    Jli = jacrev(lambda th, L=L, i=i: run(ctx, th, stop=(L, i)))(zero); AJ = A_ @ Jli; BJ = B_ @ Jli
                    for k in range(K):
                        w = wts[k]; per_block[L][k] += (AJ.T * w) @ BJ + (BJ.T * w) @ AJ
                        unit_e[L][k] += 2 * w * torch.einsum("hd,de,he->h", AJ, Bc[k], BJ) / nB[k]      # signed energy of each unit's form against B
            cum = torch.zeros(K, d, d, device=dev)
            for k in range(K):
                r = dict(context=c, direction=k, name=names[k], block_r2_alone={}, cum_r2={}, block_energy={})
                cum_k = torch.zeros(d, d, device=dev)
                for L in layers:
                    P = per_block[L][k]; r["block_r2_alone"][L] = 1 - float((Bc[k] - P).square().sum() / nB[k]); cum_k += P; r["cum_r2"][L] = 1 - float((Bc[k] - cum_k).square().sum() / nB[k])
                    r["block_energy"][L] = float(unit_e[L][k].sum())
                    top = unit_e[L][k].abs().topk(min(5, unit_e[L].shape[1])).indices.tolist(); r.setdefault("top_units", {})[L] = [(int(h), round(float(unit_e[L][k][h]), 4)) for h in top]
                    for h in range(unit_e[L].shape[1]):
                        if abs(float(unit_e[L][k][h])) > 0.01: unit_energy[f"{L}.{h}"].append(float(unit_e[L][k][h]))
                rec.append(r)
            mb = {L: float(np.median([r["block_energy"][L] for r in rec])) for L in layers}
            print(f"[blocks] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): all-units R2 {np.median([r['cum_r2'][layers[-1]] for r in rec]):.3f} | signed energy by block {({L: round(v, 3) for L, v in mb.items()})}", flush=True)
            json.dump(dict(partial=True, records=rec), open(a.out + ".partial", "w"), default=float)
    blk = {L: float(np.median([r["block_energy"][L] for r in rec])) for L in layers}; alone = {L: float(np.median([r["block_r2_alone"][L] for r in rec])) for L in layers}
    ue = {u: (float(np.mean(v)), len(v)) for u, v in unit_energy.items()}; top = sorted(ue.items(), key=lambda kv: -abs(kv[1][0]))[:40]
    res = dict(args=vars(a), summary=dict(all_units_r2=float(np.median([r["cum_r2"][layers[-1]] for r in rec])), block_signed_energy=blk, block_r2_alone=alone, top_units=top,
               block8_r2_alone_readers=float(np.median([r["block_r2_alone"][a.source_layer] for r in rec if r["name"].startswith("reader")])), block8_r2_alone_tokens=float(np.median([r["block_r2_alone"][a.source_layer] for r in rec if r["name"].startswith("tok")]))),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(summary={k: v for k, v in res["summary"].items() if k != "top_units"}, top_units=top[:15]), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
