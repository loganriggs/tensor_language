"""PROJECTOR SINGLETON DIAGNOSTIC (instrument probe, non-scoring).

The projector-form factorial receipt failed its theorem-grade singleton
bridge UNIFORMLY (.11-.20 nat across all six window x source cells) while
replay/empty/closure/doubled-delta were all exactly 0.  The theorem said: at
a singleton patch the live product equals the captured source product, so
deviation == frozen delta and projector removal == subtractive removal
exactly.  The receipt says otherwise.  Exactly one link in that chain is
broken; this probe measures every link directly on ONE batch (code_validation
first batch, singleton m8, slot-0 positions, both sources):

  L1: live product at hook  vs  captured source product   (max abs)
  L2: projector 'updated'   vs  captured absent product   (max abs)
  L3: subtractive 'updated' vs  captured absent product   (max abs)
  L4: projector nll damage  vs  subtractive nll damage    (max abs nat)
  L5: subtractive damage    vs  rung474 bundle slot-0 singleton values

Diagnosis tree (frozen): L1 large -> capture path != live path (474's
exactness needs re-examination; theorem's premise false).  L1 ~0 but L2
large -> the projector hook algebra is wrong (implementation defect in my
rung).  L1,L2,L3 ~0 but L4 large -> downstream nll/index accounting.  L4 ~0
but L5 large -> the BRIDGE COMPARISON in analyze() mis-indexes the bundle.

pred labels are diagnostic markers, not scientific claims; there is no null
and nothing is licensed by any outcome.  Price: ~10 forwards, <120s, 0
deployed parameters; touches only already-opened objects.
"""
# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "projector_singleton_diagnostic_results.json"
for _p in (ROOT, ROOT / "basis_aligned/polynomial_causal", ROOT / "basis_aligned/qk_mdl", BQ, OPS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import equality_query_projector_factorial as proj  # noqa: E402
import equality_query_subtractive_factorial_rung474 as sub474  # noqa: E402
from receipt import dump  # noqa: E402

position_parent = sub474.position_parent
product_parent = sub474.product_parent
source_parent = sub474.source_parent

WINDOW = "code_validation"
SITE = "m8"
LAYER = sub474.MODULES[sub474.SITES.index(SITE)]


def probe_patch(model, tokens, *, arm, scale, delta, absent, position_mask, form):
    """Run one singleton patch capturing the live product and the update."""
    seen = {}
    down = model.transformer.h[LAYER].mlp.Down

    def hook(_module, inputs):
        product = inputs[0]
        current = product[position_mask].float()
        seen["live"] = current.detach().clone()
        if form == "projector":
            direction = delta[position_mask]
            norms = direction.norm(dim=-1, keepdim=True)
            unit = torch.where(norms > 1e-12, direction / norms.clamp_min(1e-12),
                               torch.zeros_like(direction))
            deviation = current - absent[position_mask].float()
            component = (deviation * unit).sum(-1, keepdim=True)
            new = current - unit * component
        else:
            new = current - delta[position_mask]
        seen["updated"] = new.detach().clone()
        updated = product.clone()
        updated[position_mask] = new.to(product.dtype)
        return (updated,)

    handle = down.register_forward_pre_hook(hook)
    try:
        logits, _, _, _ = source_parent.run_forward(model, tokens, arm=arm, scale=scale)
    finally:
        handle.remove()
    return logits, seen


def main():
    (roles, scale, old_effects, selections, old_position,
     old_factorial, metadata) = sub474.validate_inputs()
    sub_bundle = torch.load(proj.SUB_BUNDLE, map_location="cpu", weights_only=True)
    if os.environ.get("BQLIB_DRYRUN") == "1":
        role = next(role for name, role, _, _ in sub474.WINDOWS if name == WINDOW)
        payload, _ = roles[role]
        assert "rows" in payload and WINDOW in selections
        print(json.dumps({"status": "dry_run_passed", "rung": "projector_singleton_diagnostic",
                          "model_loaded": False}))
        return
    started = time.time()
    model, _ = sub474.facade.load_bilin18()
    device = next(model.parameters()).device
    role = next(role for name, role, _, _ in sub474.WINDOWS if name == WINDOW)
    payload, _ = roles[role]
    selection = selections[WINDOW]
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    first_doc = min(by_doc)
    rows = payload["rows"][first_doc:first_doc + sub474.BATCH]
    tokens = rows[:, :-1].to(device)
    chosen = []
    for doc in range(first_doc, first_doc + len(rows)):
        if by_doc.get(doc):
            output_index, query = by_doc[doc][0]
            chosen.append((output_index, doc - first_doc, query))
    targets = [(local_doc, query) for _, local_doc, query in chosen]
    query_mask, _, _, _ = position_parent.position_masks(
        len(rows), tokens.shape[1], targets, device)

    _, absent_products, _, _, _ = product_parent.run_term_forward(
        model, tokens, arm="base", capture_products=True)
    absent = absent_products[SITE].float()
    subset_index = sub474.SUBSETS.index((sub474.SITES.index(SITE),))
    result = {"status": "complete", "rung": "projector_singleton_diagnostic",
              "window": WINDOW, "site": SITE, "sources": {}}
    links_ok = {"L1": True, "L2": True, "L4": True, "L5": True}
    for si, source in enumerate(sub474.SOURCES):
        arm = source_parent.SOURCE_ARMS[source]
        source_logits, source_products, _, _, _ = product_parent.run_term_forward(
            model, tokens, arm=arm, scale=scale, capture_products=True)
        source_nll = position_parent._nll(source_logits, rows)
        delta = source_products[SITE].float() - absent
        captured = source_products[SITE].float()[query_mask]
        row = {}
        proj_logits, seen_p = probe_patch(
            model, tokens, arm=arm, scale=scale, delta=delta, absent=absent,
            position_mask=query_mask, form="projector")
        row["L1_live_vs_captured_max_abs"] = float((seen_p["live"] - captured).abs().max())
        row["L2_projector_updated_vs_absent_max_abs"] = float(
            (seen_p["updated"] - absent[query_mask].float()).abs().max())
        sub_logits, seen_s = probe_patch(
            model, tokens, arm=arm, scale=scale, delta=delta, absent=absent,
            position_mask=query_mask, form="subtractive")
        row["L3_subtractive_updated_vs_absent_max_abs"] = float(
            (seen_s["updated"] - absent[query_mask].float()).abs().max())
        damage_p = position_parent._nll(proj_logits, rows) - source_nll
        damage_s = position_parent._nll(sub_logits, rows) - source_nll
        pv = torch.tensor([float(damage_p[ld, q]) for _, ld, q in chosen])
        sv = torch.tensor([float(damage_s[ld, q]) for _, ld, q in chosen])
        row["L4_projector_vs_subtractive_damage_max_abs_nat"] = float((pv - sv).abs().max())
        bundle_vals = sub_bundle["windows"][WINDOW]["effects"][si, subset_index]
        bv = torch.tensor([float(bundle_vals[oi]) for oi, _, _ in chosen])
        row["L5_subtractive_vs_bundle_max_abs_nat"] = float((sv - bv).abs().max())
        row["chosen_positions"] = len(chosen)
        result["sources"][source] = row
        links_ok["L1"] &= row["L1_live_vs_captured_max_abs"] <= 1e-5
        links_ok["L2"] &= row["L2_projector_updated_vs_absent_max_abs"] <= 1e-4
        links_ok["L4"] &= row["L4_projector_vs_subtractive_damage_max_abs_nat"] <= 1e-6
        links_ok["L5"] &= row["L5_subtractive_vs_bundle_max_abs_nat"] <= 1e-6
    result.update({
        'pred_a_live_equals_captured': bool(links_ok["L1"]),
        'pred_b_projector_update_reaches_absent': bool(links_ok["L2"]),
        'pred_c_damages_and_bundle_consistent': bool(links_ok["L4"] and links_ok["L5"]),
        "runtime_s": time.time() - started,
        "raw_tokens_logits_or_hidden_states_included": False,
    })
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=1))
    for source, row in result["sources"].items():
        print(source, {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
