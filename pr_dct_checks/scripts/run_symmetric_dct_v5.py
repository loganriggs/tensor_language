"""Rung 5 of the symmetric DCT lane (plans/SYMMETRIC_DCT_PLAN_V5.md): block-level location of the curvature (MLPs) and the
transport (attention values) for each context's top interaction direction.

  python scripts/run_symmetric_dct_v5.py --smoke
  python scripts/run_symmetric_dct_v5.py --device cuda --out results/symmetric_dct_v5.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec, participation_ratio
from circuit_checks.checks import output_score, hidden_response, top_units, random_units, all_units
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for

BARS = dict(control=1e-6, mlp=0.6, two_blocks=0.5, transport=0.6, units=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--heldout-contexts", type=int, default=16); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", default="results/symmetric_dct_v5.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.heldout_contexts = 2; a.sequence_length = 8; a.chunk = 8
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); K = 2; readers = F.normalize(torch.randn(cfg.n_embd, K), dim=0)
        held_ids = [torch.randint(0, 64, (1, a.sequence_length), generator=torch.Generator().manual_seed(20 + i)) for i in range(2)]
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device); K = readers.shape[1]
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu")
        held_ids = [rows[96 + i:97 + i, :a.sequence_length].long() for i in range(a.heldout_contexts)]
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True, per_head=True); layers = spans.all_layers
    spans_nonorm = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=False, per_head=True)   # control: without the final norm freezing everything must give exactly 1
    widths = {L: spans.mlp_width(L) for L in layers}; s89 = [a.source_layer, a.source_layer + 1]
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(ids) for ids in held_ids]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)
    vals = lambda Ls: FreezeSpec(attn_modes={L: {"values": True} for L in Ls})
    everything = FreezeSpec(mlp_masks=all_units(widths, layers).mlp_masks, attn_layers=set(layers))
    rec = []; g = torch.Generator().manual_seed(0); t0 = time.time(); ctrl = []
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            span = spans.make_span(ctx); Hc = full_hessian(lambda th: readers.T @ span(th, None)[0], d, dev, a.chunk)
            for k in range(K):
                B = 0.5 * (Hc[k] + Hc[k].T); e, V = torch.linalg.eigh(B); v = V[:, e.abs().argmax()]; u = readers[:, k]
                full = float(output_score(span, u, v, v)); frac = lambda fz: 1 - float(output_score(span, u, v, v, fz)) / full if abs(full) > 1e-12 else float("nan")
                r = dict(context=c, reader=k, full=full, eigen_pr=float(participation_ratio(e)), top_eig_share=float(e[e.abs().argmax()] ** 2 / e.square().sum()))
                r["everything"] = frac(everything)
                sp0 = spans_nonorm.make_span(ctx); f0 = float(output_score(sp0, u, v, v)); r["everything_no_final_norm"] = 1 - float(output_score(sp0, u, v, v, everything)) / f0 if abs(f0) > 1e-12 else float("nan")
                ctrl.append(abs(r["everything_no_final_norm"] - 1))
                r["mlp_block"] = {L: frac(all_units(widths, [L])) for L in layers}; r["all_mlps"] = frac(all_units(widths, layers))
                resp = hidden_response(span, v, v, layers)
                for kk in (20, 100):
                    r[f"top{kk}_units"] = frac(top_units(resp, kk))
                r["random20_units"] = frac(random_units(widths, 20, g))
                r["values_block"] = {L: frac(vals([L])) for L in layers}; r["values_all"] = frac(vals(layers)); r["values_8_9"] = frac(vals(s89))
                rec.append(r)
            m = lambda key: float(np.nanmedian([x[key] for x in rec]))
            print(f"[locate] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): all-MLPs {m('all_mlps'):.2f} top100 units {m('top100_units'):.2f} values-all {m('values_all'):.2f} values-8-9 {m('values_8_9'):.2f} everything {m('everything'):.3f} (no final norm {m('everything_no_final_norm'):.6f})", flush=True)
    med = lambda key: float(np.nanmedian([x[key] for x in rec]))
    mlp_blocks = {L: float(np.nanmedian([x["mlp_block"][L] for x in rec])) for L in layers}; val_blocks = {L: float(np.nanmedian([x["values_block"][L] for x in rec])) for L in layers}
    two = float(np.nanmedian([sum(sorted(x["mlp_block"].values(), key=lambda z: -abs(z))[:2]) for x in rec]))
    transport = float(np.nanmedian([x["values_8_9"] / x["values_all"] if abs(x["values_all"]) > 1e-9 else np.nan for x in rec]))
    preds = dict(pred_a_instrument=max(ctrl) <= BARS["control"], pred_b_mlp_curvature=med("all_mlps") >= BARS["mlp"], pred_c_few_mlp_blocks=two >= BARS["two_blocks"],
                 pred_d_transport_8_9=transport >= BARS["transport"], pred_e_units=med("top100_units") >= BARS["units"])
    res = dict(args=vars(a), predictions=preds, controls=dict(everything_max_dev=max(ctrl)),
               summary=dict(all_mlps=med("all_mlps"), mlp_blocks=mlp_blocks, two_largest_mlp_blocks=two, top20_units=med("top20_units"), top100_units=med("top100_units"), random20_units=med("random20_units"),
                            values_all=med("values_all"), values_8_9=med("values_8_9"), transport_ratio_8_9=transport, values_blocks=val_blocks, eigen_pr=med("eigen_pr"), top_eig_share=med("top_eig_share")),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary=res["summary"]), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
