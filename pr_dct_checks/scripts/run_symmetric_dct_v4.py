"""Rung 4 of the symmetric DCT lane (plans/SYMMETRIC_DCT_PLAN_V4.md): exact second-order mechanism budget by inclusion-exclusion
over detach-frozen Hessians.

  python scripts/run_symmetric_dct_v4.py --smoke
  python scripts/run_symmetric_dct_v4.py --device cuda --out results/symmetric_dct_v4.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec, participation_ratio
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for
from scripts.run_symmetric_dct_v3 import build_dictionary, FormDictionary, solve

BARS = dict(control=1e-6, symmetry=1e-3, pattern_path=0.5, pure_r2=0.6, cross_pr=32)
MODES = {"full": {}, "vf": {"values": True}, "pf": {"pattern": True}, "vf_af": {"values": True, "a": True}, "vf_bf": {"values": True, "b": True},
         "all": {"values": True, "pattern": True}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--heldout-contexts", type=int, default=16); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-6); ap.add_argument("--out", default="results/symmetric_dct_v4.json")
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
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; T = a.sequence_length
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True, per_head=True); layers = spans.all_layers
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(ids) for ids in held_ids]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)
    spec = lambda md: FreezeSpec(attn_modes={L: dict(md) for L in layers})
    def hess(ctx, freeze):
        span = spans.make_span(ctx)
        return full_hessian(lambda th: readers.T @ span(th, freeze)[0], d, dev, a.chunk).double()
    # ---- dictionary (rung 3) ----------------------------------------------------------------------------------------------------------
    A, C, R, table = build_dictionary(model, layers, T); dic = FormDictionary(A, C, R, dev); G = dic.gram(); idx_all = torch.arange(dic.n, device=dev)
    # ---- controls on context 0 --------------------------------------------------------------------------------------------------------
    with sdpa_kernel(SDPBackend.MATH):
        H_all_modes = hess(ctxs[0], spec(MODES["all"])); H_all_layers = hess(ctxs[0], FreezeSpec(attn_layers=set(layers)))
        c1 = float((H_all_modes - H_all_layers).norm() / H_all_layers.norm().clamp_min(1e-30))
        H_ab = hess(ctxs[0], spec({"a": True, "b": True})); H_pf = hess(ctxs[0], spec(MODES["pf"]))
        c2 = float((H_ab - H_pf).norm() / H_pf.norm().clamp_min(1e-30))
    print(f"[controls] all-modes vs all-layers freeze {c1:.2e} | a+b frozen vs pattern frozen {c2:.2e}", flush=True)
    # ---- budget per context -------------------------------------------------------------------------------------------------------------
    pieces = ["pure_a", "pure_b", "cross_ab", "pure_v", "cross_pv", "rest"]; rec = []; sym_max = 0.0; t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            H = {m: hess(ctx, spec(md)) for m, md in MODES.items()}
            for m in H:
                sym_max = max(sym_max, float(max((H[m][k] - H[m][k].T).norm() / H[m][k].norm().clamp_min(1e-30) for k in range(K))))
            P = {"pure_a": H["vf_bf"] - H["all"], "pure_b": H["vf_af"] - H["all"], "cross_ab": H["vf"] - H["vf_af"] - H["vf_bf"] + H["all"],
                 "pure_v": H["pf"] - H["all"], "cross_pv": H["full"] - H["pf"] - H["vf"] + H["all"], "rest": H["all"]}
            for k in range(K):
                Hf = H["full"][k]; n2 = float(Hf.square().sum()); r = dict(context=c, reader=k, norm2=n2, signed={}, energy={}, pr={})
                for p in pieces:
                    X = P[p][k]; r["signed"][p] = float((X * Hf).sum() / n2); r["energy"][p] = float(X.square().sum() / n2)
                    r["pr"][p] = float(participation_ratio(torch.linalg.eigvalsh(0.5 * (X + X.T))))
                pure = 0.5 * ((P["pure_a"][k] + P["pure_b"][k]) + (P["pure_a"][k] + P["pure_b"][k]).T); cross = 0.5 * (P["cross_ab"][k] + P["cross_ab"][k].T)
                _, e1 = solve(G, dic.inner(pure), idx_all, a.ridge); r["r2_pure_dictionary"] = e1 / float(pure.square().sum())
                _, e2 = solve(G, dic.inner(cross), idx_all, a.ridge); r["r2_cross_dictionary"] = e2 / float(cross.square().sum())
                rec.append(r)
            med = lambda key: {p: float(np.median([x[key][p] for x in rec])) for p in pieces}
            print(f"[budget] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): signed shares {({p: round(v, 3) for p, v in med('signed').items()})} | cross PR {np.median([x['pr']['cross_ab'] for x in rec]):.1f} | R2 pure {np.median([x['r2_pure_dictionary'] for x in rec]):.3f} cross {np.median([x['r2_cross_dictionary'] for x in rec]):.3f}", flush=True)
    med_signed = {p: float(np.median([x["signed"][p] for x in rec])) for p in pieces}; med_energy = {p: float(np.median([x["energy"][p] for x in rec])) for p in pieces}
    med_pr = {p: float(np.median([x["pr"][p] for x in rec])) for p in pieces}
    pattern_path = float(np.median([x["signed"]["pure_a"] + x["signed"]["pure_b"] + x["signed"]["cross_ab"] for x in rec]))
    preds = dict(pred_a_instrument=c1 <= BARS["control"] and c2 <= BARS["control"] and sym_max <= BARS["symmetry"],
                 pred_b_pattern_path=pattern_path >= BARS["pattern_path"],
                 pred_c_cross_dominates=all(med_signed["cross_ab"] > med_signed[p] for p in ("pure_a", "pure_b", "pure_v", "cross_pv")),
                 pred_d_pure_is_fixed_form=float(np.median([x["r2_pure_dictionary"] for x in rec])) >= BARS["pure_r2"],
                 pred_e_cross_low_rank=med_pr["cross_ab"] <= BARS["cross_pr"])
    res = dict(args=vars(a), predictions=preds, controls=dict(all_modes_vs_layers=c1, ab_vs_pattern=c2, symmetry_max=sym_max),
               summary=dict(median_signed_share=med_signed, median_energy_share=med_energy, median_eigen_pr=med_pr, median_pattern_path=pattern_path,
                            median_r2_pure=float(np.median([x["r2_pure_dictionary"] for x in rec])), median_r2_cross=float(np.median([x["r2_cross_dictionary"] for x in rec]))),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary=res["summary"]), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
