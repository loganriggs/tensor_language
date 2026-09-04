#!/usr/bin/env python
"""circuit_battery_attn5_class_gate_price -- is attention 5's CE price the price of the CLASS GATE? (cross-lane)

SS2829: attention 5 is the type gate -- ablating its write costs more candidate-class mass than any other component on 6 of 7 behaviours,
across classes as different as digits, roman numerals, month names and a repeated word. The frontier lane has independently carried
"attn5's write = the price cliff" as one of the three largest gaps in the explained fraction for weeks. Two lanes, opposite directions,
same component. This rung tests whether they are the same fact: it measures every component's CE damage on NATURAL DOCUMENTS with the
same ablation the circuit lane uses, and asks whether the class-mass map from SS2829 predicts the document CE map across components --
and whether it predicts it better than the margin map does.

# BQGATE: EXPERIMENT  pred_a_attn5_leads_the_document_ce_map pred_b_class_mass_predicts_the_ce_price
#                     pred_c_attn5_is_disproportionately_expensive pred_d_class_beats_margin_as_predictor
#                     pred_e_instrument_reproduces_module_ce

SIGN CONVENTION: CE damage d_ce = CE_arm - CE_NATIVE in NATS, POSITIVE = the arm HURTS the model (this is the local-ablation convention;
it is NOT the SS312 frontier's L2 quantity and nothing here installs into that frontier). Class-mass damage from SS2829 is
logmass_NATIVE - logmass_arm, POSITIVE = the arm REMOVES class mass.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ATTN5_CLASS_GATE_PRICE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_attn5_class_gate_price.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ATTN5_CLASS_GATE_PRICE_PREREGISTRATION.md"
CLASSMAP = ROOT / "circuit_battery_class_mass_localisation_results.json"
RUNG = "circuit_battery_attn5_class_gate_price"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "58b2f18824faa4d1e5ebbc42360bdaed2e044c8b93acdc1f0865749942760223",
          CLASSMAP: "b4d83e6790ab83eeb088faf53a4b5d24133244cf43eed40400aa708028a73793",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL, V, T = R.D, R.NL, R.V, R.T
NAT = R.ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
COMPONENTS = [(kd, l) for l in range(NL) for kd in ("attn", "mlp")]
NDOCS = 8 if SMOKE else 32
CHUNK = 8
BARS = {"attn5_top": 3, "rho_class": 0.50, "dispro": 2.0, "beats_margin": 0.10, "ce_tol": 0.01}
NULLS = {"attn5_top_ge": 8, "rho_class_le": 0.10, "dispro_le": 1.0, "beats_margin_le": 0.0}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def doc_forward(m, idx, ablate=None):
    """Full-sequence forward with one component's write zeroed at EVERY position; returns logits and its write norm."""
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x; v1 = None
    norm = None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if ablate == ("attn", site):
            norm = float(write.float().pow(2).sum(-1).sqrt().mean())
            write = torch.zeros_like(write)
        x = x + write
        out = blk.mlp(F.rms_norm(x, (D,)))
        if ablate == ("mlp", site):
            norm = float(out.float().pow(2).sum(-1).sqrt().mean())
            out = torch.zeros_like(out)
        x = x + out
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0), norm


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, dtype=float))).astype(float)
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    t0 = time.time()
    check_hashes()
    cm = json.load(open(CLASSMAP))
    m = R.load_model().to(DEV).eval()
    nat = torch.load(NAT, map_location="cpu")
    nat = (nat["rows"] if isinstance(nat, dict) else nat).long()[:NDOCS]
    fwd = 0

    def ce_of(ablate):
        nonlocal fwd
        s, n, nm = 0.0, 0, []
        for i in range(0, nat.shape[0], CHUNK):
            idx = nat[i:i + CHUNK, :T - 1].to(DEV)
            tgt = nat[i:i + CHUNK, 1:T].to(DEV)
            lg, w = doc_forward(m, idx, ablate); fwd += 1
            s += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum"))
            n += tgt.numel()
            if w is not None:
                nm.append(w)
        return s / n, (float(np.mean(nm)) if nm else float("nan"))

    ce_nat, _ = ce_of(None)
    idx0 = nat[:CHUNK, :T - 1].to(DEV); tgt0 = nat[:CHUNK, 1:T].to(DEV)
    ce_module = float(m(idx0.contiguous(), tgt0.contiguous()))
    lg0, _ = doc_forward(m, idx0); fwd += 1
    ce_manual = float(F.cross_entropy(lg0.reshape(-1, V), tgt0.reshape(-1)))

    ce, norms = {}, {}
    for comp in COMPONENTS:
        c, w = ce_of(comp)
        name = f"{comp[0]}{comp[1]}"
        ce[name] = c - ce_nat
        norms[name] = w
        print(f"[price] {name:8s} d_ce={ce[name]:+.4f} nats  write_norm={w:.3f}", flush=True)

    # SS2829's maps, pooled across its behaviours
    names = sorted(ce)
    cls_map, mar_map = {}, {}
    for n in names:
        cls_map[n] = float(np.median([cm["tasks"][t]["class_damage"].get(n, 0.0) for t in cm["tasks"]]))
        mar_map[n] = float(np.median([cm["tasks"][t]["margin_damage"].get(n, 0.0) for t in cm["tasks"]]))
    rho_class = spearman([cls_map[n] for n in names], [ce[n] for n in names])
    rho_margin = spearman([mar_map[n] for n in names], [ce[n] for n in names])
    order = sorted(names, key=lambda n: -ce[n])
    attn5_rank = order.index("attn5") + 1
    per_norm = {n: ce[n] / max(norms[n], 1e-6) for n in names}
    med_per_norm = float(np.median([per_norm[n] for n in names]))
    dispro = per_norm["attn5"] / max(med_per_norm, 1e-9)

    preds = {
        'pred_a_attn5_leads_the_document_ce_map': bool(attn5_rank <= BARS["attn5_top"]),
        'pred_b_class_mass_predicts_the_ce_price': bool(rho_class >= BARS["rho_class"]),
        'pred_c_attn5_is_disproportionately_expensive': bool(dispro >= BARS["dispro"]),
        'pred_d_class_beats_margin_as_predictor': bool(rho_class - rho_margin >= BARS["beats_margin"]),
        'pred_e_instrument_reproduces_module_ce': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_attn5_outside_top8": bool(attn5_rank >= NULLS["attn5_top_ge"]),
        "b_null_no_class_correlation": bool(rho_class <= NULLS["rho_class_le"]),
        "c_null_not_disproportionate": bool(dispro <= NULLS["dispro_le"]),
        "d_null_margin_predicts_as_well": bool(rho_class - rho_margin <= NULLS["beats_margin_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "ce_native_nats": ce_nat, "ce_module_check": ce_module, "ce_manual_check": ce_manual,
              "n_docs": int(nat.shape[0]),
              "summary": {"ce_damage_top8": [[n, round(ce[n], 4)] for n in order[:8]],
                          "attn5_ce_rank": attn5_rank, "attn5_ce_damage": ce["attn5"],
                          "attn5_per_norm": per_norm["attn5"], "median_per_norm": med_per_norm,
                          "disproportion": dispro, "rho_class_vs_ce": rho_class,
                          "rho_margin_vs_ce": rho_margin, "rho_gap": rho_class - rho_margin},
              "ce_damage": ce, "write_norms": norms, "class_map": cls_map, "margin_map": mar_map,
              "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1400])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(COMPONENTS)} component ablations x {NDOCS} natural documents; "
              f"correlates document CE damage with SS2829's class and margin maps; no model loaded")
        sys.exit(0)
    main()
