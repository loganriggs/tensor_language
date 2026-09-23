"""Circuits lane rung C1a (plans/CIRCUITS_PLAN_V1.md): per-direction attribution of transport heads and curvature units for
100 output directions on bilin18, with batched freezes and power-iteration probe directions.

  python scripts/run_circuits_c1a.py --smoke
  python scripts/run_circuits_c1a.py --device cuda --out results/circuits_c1a.json
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad, jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec
from circuit_checks.checks import output_score, hidden_response, top_units, all_units
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for

BARS = dict(control=1e-6, batched=1e-5, rayleigh=0.2, reuse_frac=0.5, blocks_frac=0.7, units_min=100, tokens_mlp=0.9)
KEY_HEADS = [(9, 8), (9, 7), (8, 2)]


def power_top(f, d, dev, iters=30, seed=0):
    """Top-|eigenvalue| direction of the Hessian of scalar f at 0 via forward-over-reverse Hessian-vector products."""
    g = torch.Generator().manual_seed(seed); x = F.normalize(torch.randn(d, generator=g), dim=0).to(dev); zero = torch.zeros(d, device=dev)
    hvp = lambda vec: jvp(lambda th: grad(f)(th), (zero,), (vec,))[1]
    for _ in range(iters):
        x = F.normalize(hvp(x), dim=0)
    return x, float(x @ hvp(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--contexts", type=int, default=16); ap.add_argument("--tokens", type=int, default=84)
    ap.add_argument("--chunk", type=int, default=30); ap.add_argument("--out", default="results/circuits_c1a.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.contexts = 2; a.sequence_length = 8; a.tokens = 3; a.chunk = 4
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); readers = F.normalize(torch.randn(cfg.n_embd, 2), dim=0)
        rows = torch.randint(0, 64, (8, a.sequence_length), generator=torch.Generator().manual_seed(20)); held_rows = rows[:a.contexts]; token_rows = rows
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device)
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu")
        held_rows = rows[96:96 + a.contexts, :a.sequence_length].long(); token_rows = rows[96:192, 1:a.sequence_length + 1].long()
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; H = cfg.n_head
    counts = collections.Counter(token_rows.flatten().tolist()); toks = [t for t, _ in counts.most_common(a.tokens)]
    W = model.lm_head.weight.float(); tokdirs = F.normalize(W[toks].T, dim=0)                             # [d, n_tok]
    U = torch.cat([readers.to(dev), tokdirs.to(dev)], 1); names = [f"reader{k}" for k in range(readers.shape[1])] + [f"tok{t}" for t in toks]
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True, per_head=True); layers = spans.all_layers
    spans_nonorm = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=False, per_head=True)
    widths = {L: spans.mlp_width(L) for L in layers}
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(held_rows[i:i + 1]) for i in range(len(held_rows))]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)
    nL = len(layers); head_masks = torch.eye(nL * H, device=dev); block_masks = torch.zeros(nL, nL * H, device=dev)
    for i in range(nL):
        block_masks[i, i * H:(i + 1) * H] = 1
    everything = FreezeSpec(mlp_masks=all_units(widths, layers).mlp_masks, attn_layers=set(layers))

    def head_value_scores(sp, u, v, masks):
        def one(mask):
            return output_score(sp, u, v, v, FreezeSpec(head_value_masks={L: mask[i * H:(i + 1) * H] for i, L in enumerate(layers)}))
        return vmap(one, chunk_size=a.chunk)(masks)

    def mlp_block_scores(sp, u, v):
        masks = torch.zeros(nL, nL, max(widths.values()), device=dev)
        for i, L in enumerate(layers):
            masks[i, i, :widths[L]] = 1
        one = lambda mask: output_score(sp, u, v, v, FreezeSpec(mlp_masks={L: mask[i, :widths[L]] for i, L in enumerate(layers)}))
        return vmap(one, chunk_size=a.chunk)(masks)

    # ---- controls -------------------------------------------------------------------------------------------------------------------
    ctrl = dict(everything=[], batched=[], rayleigh=[])
    with sdpa_kernel(SDPBackend.MATH):
        sp0 = spans.make_span(ctxs[0]); sp0n = spans_nonorm.make_span(ctxs[0]); g = torch.Generator().manual_seed(1)
        for k in (0, U.shape[1] - 1):
            u = U[:, k]; f = lambda th: u @ sp0(th, None)[0]; v, ray = power_top(f, d, dev)
            f0 = float(output_score(sp0n, u, v, v)); ctrl["everything"].append(abs(float(output_score(sp0n, u, v, v, everything)) / f0))   # residual score after freezing everything (must be 0)
            if not a.smoke or True:
                Hk = full_hessian(lambda th: (u @ sp0(th, None)[0])[None], d, dev, a.chunk if a.smoke else 64)[0]; e = torch.linalg.eigvalsh(0.5 * (Hk + Hk.T)); top = float(e[e.abs().argmax()])
                ctrl["rayleigh"].append(abs(abs(ray) - abs(top)) / abs(top))
            masks = torch.stack([(torch.rand(nL * H, generator=g) < 0.3).float() for _ in range(3)]).to(dev)
            b = head_value_scores(sp0, u, v, masks); s = torch.stack([output_score(sp0, u, v, v, FreezeSpec(head_value_masks={L: masks[i, j * H:(j + 1) * H] for j, L in enumerate(layers)})) for i in range(3)])
            ctrl["batched"].append(float((b - s).abs().max() / s.abs().max().clamp_min(1e-30)))
    print(f"[controls] everything {max(ctrl['everything']):.2e} | batched vs sequential {max(ctrl['batched']):.2e} | rayleigh vs exact top {max(ctrl['rayleigh']):.3f}", flush=True)

    # ---- attribution -----------------------------------------------------------------------------------------------------------------
    rec = []; unit_hits = collections.defaultdict(collections.Counter); t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            sp = spans.make_span(ctx)
            for k in range(U.shape[1]):
                u = U[:, k]; v, ray = power_top(lambda th: u @ sp(th, None)[0], d, dev)
                full = float(output_score(sp, u, v, v)); fr = lambda s: 1 - float(s) / full if abs(full) > 1e-12 else float("nan")
                hv = head_value_scores(sp, u, v, head_masks); bv = head_value_scores(sp, u, v, block_masks); allv = head_value_scores(sp, u, v, torch.ones(1, nL * H, device=dev))[0]
                mb = mlp_block_scores(sp, u, v); allm = output_score(sp, u, v, v, all_units(widths, layers))
                resp = hidden_response(sp, v, v, layers); t20 = top_units(resp, 20); t100 = top_units(resp, 100)
                s20 = output_score(sp, u, v, v, t20); s100 = output_score(sp, u, v, v, t100)
                units100 = [(L, int(i)) for L, m in t100.mlp_masks.items() for i in m.nonzero().flatten()]
                for un in units100:
                    unit_hits[k][un] += 1
                rec.append(dict(context=c, direction=k, rayleigh=ray, full=full, head_values={f"{layers[i // H]}.{i % H}": fr(hv[i]) for i in range(nL * H)},
                                block_values={L: fr(bv[i]) for i, L in enumerate(layers)}, all_values=fr(allv), mlp_blocks={L: fr(mb[i]) for i, L in enumerate(layers)},
                                all_mlps=fr(allm), top20_units=fr(s20), top100_units=fr(s100)))
            done = len(rec); print(f"[attr] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s, {done} pairs): median all-MLPs {np.nanmedian([r['all_mlps'] for r in rec]):.2f} all-values {np.nanmedian([r['all_values'] for r in rec]):.2f} top100 {np.nanmedian([r['top100_units'] for r in rec]):.2f}", flush=True)
            json.dump(dict(partial=True, records=rec), open(a.out + ".partial", "w"), default=float)
    # ---- aggregation ------------------------------------------------------------------------------------------------------------------
    nU = U.shape[1]; per_dir = []
    for k in range(nU):
        R = [r for r in rec if r["direction"] == k]
        hs = {h: float(np.nanmean([r["head_values"][h] for r in R])) for h in R[0]["head_values"]}
        top3 = sorted(hs, key=lambda h: -hs[h])[:3]
        mb = {L: float(np.nanmean([r["mlp_blocks"][L] for r in R])) for L in layers}; top2b = sorted(mb, key=lambda L: -mb[L])[:2]
        stable = sorted([un for un, n in unit_hits[k].items() if n >= len(ctxs) / 2])
        per_dir.append(dict(name=names[k], head_share=hs, top3_heads=top3, heads_over_0_1=[h for h in hs if hs[h] >= 0.1], mlp_block_share=mb, top2_mlp_blocks=top2b,
                            stable_units=[f"{L}.{i}" for L, i in stable], all_mlps=float(np.nanmedian([r["all_mlps"] for r in R])), all_values=float(np.nanmedian([r["all_values"] for r in R])),
                            top100_units=float(np.nanmedian([r["top100_units"] for r in R])), rayleigh_median=float(np.median([abs(r["rayleigh"]) for r in R]))))
    key_frac = {f"{L}.{h}": float(np.mean([f"{L}.{h}" in p["top3_heads"] for p in per_dir])) for (L, h) in KEY_HEADS}
    blocks_ok = float(np.mean([set(p["top2_mlp_blocks"]) == {a.source_layer, n_layer - 1} for p in per_dir]))
    unit_dirs = collections.Counter(u for p in per_dir for u in p["stable_units"]); reuse25 = sum(1 for u, n in unit_dirs.items() if n >= 0.25 * nU); reuse50 = sum(1 for u, n in unit_dirs.items() if n >= 0.5 * nU)
    head_dirs = collections.Counter(h for p in per_dir for h in p["heads_over_0_1"])
    tok_mlp = float(np.median([p["all_mlps"] for p in per_dir if p["name"].startswith("tok")])) if a.tokens else float("nan")
    # clustering: Jaccard on stable units (single-linkage at 0.5) and cosine on head-share vectors
    sets = [set(p["stable_units"]) for p in per_dir]; J = np.zeros((nU, nU))
    for i in range(nU):
        for j in range(nU):
            J[i, j] = len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
    hv = np.array([[p["head_share"][h] for h in per_dir[0]["head_share"]] for p in per_dir]); hn = hv / np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-12); Cs = hn @ hn.T
    def clusters(S, thr):
        parent = list(range(nU))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(nU):
            for j in range(i + 1, nU):
                if S[i, j] >= thr: parent[find(i)] = find(j)
        groups = collections.defaultdict(list)
        for i in range(nU): groups[find(i)].append(names[i])
        return sorted(groups.values(), key=len, reverse=True)
    preds = dict(pred_a_instrument=max(ctrl["everything"]) <= BARS["control"] and max(ctrl["batched"]) <= BARS["batched"] and max(ctrl["rayleigh"]) <= BARS["rayleigh"],
                 pred_b_transport_reuse=all(f >= BARS["reuse_frac"] for f in key_frac.values()), pred_c_curvature_blocks=blocks_ok >= BARS["blocks_frac"],
                 pred_d_unit_reuse=reuse25 >= (BARS["units_min"] if not a.smoke else 0), pred_e_tokens_like_readers=(tok_mlp >= BARS["tokens_mlp"]) if a.tokens else True)
    res = dict(args=vars(a), predictions=preds, controls={k: max(v) for k, v in ctrl.items()}, directions=names, tokens=toks,
               summary=dict(key_head_top3_fraction=key_frac, blocks_8_17_fraction=blocks_ok, units_reused_25=reuse25, units_reused_50=reuse50, most_reused_units=unit_dirs.most_common(30),
                            most_reused_heads=head_dirs.most_common(15), token_median_all_mlps=tok_mlp, reader_median_all_mlps=float(np.median([p["all_mlps"] for p in per_dir if p["name"].startswith("reader")])),
                            median_all_values=float(np.median([p["all_values"] for p in per_dir])), median_top100=float(np.median([p["top100_units"] for p in per_dir])),
                            jaccard_clusters_0_5=clusters(J, 0.5)[:10], head_cosine_clusters_0_9=clusters(Cs, 0.9)[:10], mean_jaccard=float(J[np.triu_indices(nU, 1)].mean()), mean_head_cosine=float(Cs[np.triu_indices(nU, 1)].mean())),
               per_direction=per_dir, records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary={k: v for k, v in res["summary"].items() if k not in ("most_reused_units", "jaccard_clusters_0_5", "head_cosine_clusters_0_9")}), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
