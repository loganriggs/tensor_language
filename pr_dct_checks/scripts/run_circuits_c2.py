"""Circuits lane rung C2 (plans/CIRCUITS_PLAN_V3.md): finite-scale check of the derivative-level attribution with clean patching.

  python scripts/run_circuits_c2.py --smoke
  python scripts/run_circuits_c2.py --device cuda --c1a results/circuits_c1a.json --out results/circuits_c2.json
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec
from circuit_checks.checks import output_score, hidden_response, top_units, clean_cache
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import readers_for
from scripts.run_circuits_c1a import power_top

BARS = dict(quadratic=0.1, quadratic_frac=0.8, zero_mask=1e-6, heads_rel=0.7, heads_abs=0.3, units_rel=0.7, random=0.05, scale=0.15)
SCALES = (0.003, 0.01, 0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--contexts", type=int, default=8); ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--c1a", default="results/circuits_c1a.json"); ap.add_argument("--out", default="results/circuits_c2.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.contexts = 2; a.sequence_length = 8; a.n_tokens = 1
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); readers = F.normalize(torch.randn(cfg.n_embd, 2), dim=0)
        held_rows = torch.randint(0, 64, (2, a.sequence_length), generator=torch.Generator().manual_seed(20)); toks = [3]
        c1a = None
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device)
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu"); held_rows = rows[96:96 + a.contexts, :a.sequence_length].long()
        c1a = json.load(open(a.c1a)); toks = c1a["tokens"][:a.n_tokens]
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; H = cfg.n_head
    W = model.lm_head.weight.float(); tokdirs = F.normalize(W[toks].T, dim=0)
    U = torch.cat([readers.to(dev), tokdirs.to(dev)], 1); names = [f"reader{k}" for k in range(readers.shape[1])] + [f"tok{t}" for t in toks]
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True, per_head=True); layers = spans.all_layers
    widths = {L: spans.mlp_width(L) for L in layers}
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(held_rows[i:i + 1]) for i in range(len(held_rows))]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)
    rho = float(torch.stack([c[0].norm(dim=-1).mean() for c in ctxs]).mean())
    per_dir = {p["name"]: p for p in c1a["per_direction"]} if c1a else {}
    g = torch.Generator().manual_seed(0)
    def head_spec(hlist, cache):
        masks = {L: torch.zeros(H) for L in layers}
        for (L, h) in hlist: masks[L][h] = 1
        return FreezeSpec(head_value_masks={L: m.to(dev) for L, m in masks.items() if m.sum() > 0}, clean_values=cache["values"])
    def unit_spec(spec_units, cache):
        spec_units.clean_hidden = cache["hidden"]; return spec_units
    def both_spec(hs, us):
        return FreezeSpec(head_value_masks=hs.head_value_masks, clean_values=hs.clean_values, mlp_masks=us.mlp_masks, clean_hidden=us.clean_hidden)
    rec = []; quad = []; zero_mask = []; t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            span = spans.make_span(ctx); cache = clean_cache(span, d)
            for k in range(U.shape[1]):
                u = U[:, k]; f = lambda th, fz=None: u @ span(th, fz)[0]; v, ray = power_top(lambda th: f(th), d, dev)
                deriv = float(output_score(span, u, v, v)); resp = hidden_response(span, v, v, layers)
                t100 = top_units(resp, 100 if not a.smoke else 5); rand_units = FreezeSpec(mlp_masks={L: (torch.rand(widths[L], generator=g) < (100 / sum(widths.values()))).float() for L in layers})
                if c1a: top_heads = [tuple(int(z) for z in h.split(".")) for h in per_dir[names[k]]["top3_heads"]]; d_heads = float(np.mean([per_dir[names[k]]["head_share"][h] for h in per_dir[names[k]]["top3_heads"]])); d_units = per_dir[names[k]]["top100_units"]
                else: top_heads = [(layers[0], 0), (layers[1], 1), (layers[2], 0)]; d_heads = float("nan"); d_units = float("nan")
                # derivative-level fractions for THIS context (for a like-for-like comparison)
                dh = 1 - float(output_score(span, u, v, v, FreezeSpec(head_value_masks={L: m for L, m in head_spec(top_heads, cache).head_value_masks.items()}))) / deriv
                du = 1 - float(output_score(span, u, v, v, t100)) / deriv
                rnd_heads = [(layers[int(i) // H], int(i) % H) for i in torch.randperm(len(layers) * H, generator=g)[:3]]
                specs = {"top3_heads": head_spec(top_heads, cache), "top100_units": unit_spec(t100, cache), "both": both_spec(head_spec(top_heads, cache), unit_spec(t100, cache)),
                         "random3_heads": head_spec(rnd_heads, cache), "random_units": unit_spec(rand_units, cache), "all_heads_and_units": both_spec(head_spec([(L, h) for L in layers for h in range(H)], cache), unit_spec(FreezeSpec(mlp_masks={L: torch.ones(widths[L]) for L in layers}), cache))}
                r = dict(context=c, direction=k, name=names[k], derivative=deriv, deriv_frac_heads=dh, deriv_frac_units=du, c1a_frac_heads=d_heads, c1a_frac_units=d_units, scales={})
                with torch.no_grad():
                    f0 = f(torch.zeros(d, device=dev))
                    zero_mask.append(float((f(0.01 * rho * v, head_spec([], cache)) - f(0.01 * rho * v)).abs() / f0.abs().clamp_min(1e-30)) if c == 0 and k == 0 else 0.0)
                    for s in SCALES:
                        al = s * rho; D = lambda fz=None: float(f(2 * al * v, fz) - 2 * f(al * v, fz) + f0 if fz is None else f(2 * al * v, fz) - 2 * f(al * v, fz) + f(torch.zeros(d, device=dev), fz))
                        base = D(); q = base / (2 * al * al); ratio = q / deriv if abs(deriv) > 1e-30 else float("nan")
                        if s == SCALES[0]: quad.append(abs(ratio - 1))
                        r["scales"][s] = dict(finite_over_derivative=ratio, **{n: (1 - D(sp) / base if abs(base) > 1e-30 else float("nan")) for n, sp in specs.items()})
                rec.append(r)
            m = lambda s, key: float(np.nanmedian([x["scales"][s][key] for x in rec]))
            print(f"[finite] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): quad-ratio@0.003 {np.nanmedian([x['scales'][0.003]['finite_over_derivative'] for x in rec]):.3f} | @1%: heads {m(0.01,'top3_heads'):.2f} (deriv {np.nanmedian([x['deriv_frac_heads'] for x in rec]):.2f}) units {m(0.01,'top100_units'):.2f} (deriv {np.nanmedian([x['deriv_frac_units'] for x in rec]):.2f}) both {m(0.01,'both'):.2f} rnd-heads {m(0.01,'random3_heads'):.2f} rnd-units {m(0.01,'random_units'):.2f}", flush=True)
    med = lambda s, key: float(np.nanmedian([x["scales"][s][key] for x in rec]))
    dh_med = float(np.nanmedian([x["deriv_frac_heads"] for x in rec])); du_med = float(np.nanmedian([x["deriv_frac_units"] for x in rec]))
    quad_ok = float(np.mean([q <= BARS["quadratic"] for q in quad]))
    preds = dict(pred_a_instrument=quad_ok >= BARS["quadratic_frac"] and max(zero_mask) <= BARS["zero_mask"],
                 pred_b_heads_finite=med(0.01, "top3_heads") >= BARS["heads_rel"] * dh_med and med(0.01, "top3_heads") >= BARS["heads_abs"],
                 pred_c_units_finite=med(0.01, "top100_units") >= BARS["units_rel"] * du_med,
                 pred_d_random_null=all(abs(med(s, "random3_heads")) <= BARS["random"] and abs(med(s, "random_units")) <= BARS["random"] for s in SCALES),
                 pred_e_scale_stability=all(abs(med(s, key) - med(0.01, key)) <= BARS["scale"] for s in (0.003, 0.03) for key in ("top3_heads", "top100_units")))
    summary = {str(s): {key: med(s, key) for key in ("finite_over_derivative", "top3_heads", "top100_units", "both", "random3_heads", "random_units", "all_heads_and_units")} for s in SCALES}
    res = dict(args=vars(a), rho=rho, predictions=preds, controls=dict(quadratic_regime_fraction=quad_ok, zero_mask_max=max(zero_mask)), derivative_medians=dict(heads=dh_med, units=du_med),
               summary=summary, records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], derivative=res["derivative_medians"], summary=summary), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
