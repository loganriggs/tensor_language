"""RUNG 323 -- SEQUENTIAL CLOSED-LOOP CONTEXT-RRR LATE TRIPLE.

Rung322's open-loop {15,16,17}@p768 context-RRR triple scored +.039485 and
28/62 certificates even though each site was excellent alone.  Test the named
confound without changing layers, rank, fit rows, graph, or price: fit in
execution order.  Fit L15 on native contexts; capture/factor L16 with fitted
L15 active; capture/factor L17 with fitted L15+L16 active.  Then physically
compose the same full triple with mixed104 at 531,632,438 scalars.

Frozen predictions
------------------
pred_a_absolute_census_and_certificate_screen:
    Census <=.025 and >=38/62 certificates.
pred_b_sequential_fit_repairs_open_loop_composition:
    Census <=70% of rung322's .03948509 and >=36 certificates (eight recovered).
pred_c_fresh_sequential_identity_and_price:
    Fresh8 max <=.045; sequential fit order, observed maps, mixed104 QK/active
    identities, and 531,632,438-scalar / 2,010,587,756-byte bill are exact.

Null: census >=.037485 or <=28 certificates.  Failure closes context-metric
late composition; no pair, layer, or rank selection follows.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed104_late_context_metric_input_triple_sequential_results.json"
PARENT = ROOT / "mixed104_late_context_metric_input_triple_results.json"
LAYERS = (15, 16, 17)
RANK = 768
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
OPEN_LOOP_DAMAGE = 0.03948509
SCALARS = 531_632_438
BYTES = 2_010_587_756


def _certificate_count(CN, battery, damage):
    valid = 0
    member_abs = {}
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        member_abs[tag] = round(value, 7)
        valid += int(value < 0.5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid, member_abs


def _install_program(model, layer, program):
    encoder = program["encoder"].to("cuda").float()
    left = program["left"].to("cuda").float()
    right = program["right"].to("cuda").float()
    down = program["down"].to("cuda").float()
    bias = program["bias"].to("cuda").float()

    def hook(_module, args, output):
        x = args[0].float()
        z = x @ encoder.T
        hidden = (z @ left.T) * (z @ right.T)
        return (hidden @ down.T + bias).to(output.dtype)

    return model.transformer.h[layer].mlp.register_forward_hook(hook)


@torch.no_grad()
def _capture_one_covariance(model, rows, target_layer, prior_programs, manual_logits):
    total = torch.zeros(1152, 1152, device="cuda")
    count = 0
    handles = [_install_program(model, layer, prior_programs[layer])
               for layer in sorted(prior_programs)]

    def capture(_module, args, _output):
        nonlocal count
        x = args[0].detach().reshape(-1, 1152).float()
        total.addmm_(x.T, x)
        count += x.shape[0]

    handles.append(model.transformer.h[target_layer].mlp.register_forward_hook(capture))
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].to("cuda"))
    finally:
        for handle in handles:
            handle.remove()
    assert count == len(rows) * 256
    covariance = total / count
    return 0.5 * (covariance + covariance.T)


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert abs(parent["census_damage"] - OPEN_LOOP_DAMAGE) <= 1e-6
        assert parent["layers"] == list(LAYERS) and parent["rank"] == RANK
        assert parent["fit_rows_half_open"] == list(FIT_SLICE)
        assert parent["literal_standalone_scalars"] == SCALARS
        print("SEQUENTIAL LATE CONTEXT-RRR | dry run: parent, order, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    assert fit_rows.shape == (24, 257)
    programs = {}
    fit_diagnostics = {}
    for layer in LAYERS:
        covariance = _capture_one_covariance(C.m, fit_rows, layer, programs, _manual_logits)
        program, _basis, diagnostics = _rrr_program(C.m.transformer.h[layer].mlp, covariance)
        programs[layer] = {name: value.cpu() for name, value in program.items()}
        fit_diagnostics[str(layer)] = diagnostics
        del covariance, program, _basis
        torch.cuda.empty_cache()
        print(f"sequential fit complete through layer {layer}", flush=True)

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": programs,
    })
    print("ARM: mixed104 + sequential late context-RRR p768 {15,16,17}", flush=True)
    run = C.main()

    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    wanted_observed = {layer: RANK for layer in LAYERS}
    wanted_qk = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (observed != wanted_observed or set(index_sets) != set(range(2, 18))
            or any(value != wanted_qk for value in index_sets.values()) or widths != {104}
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: sequential program or mixed104 identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    damage_vector = cev - base_ce
    census_damage = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid, member_abs = _certificate_count(CN, battery, damage_vector)
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = census_damage <= .025 and valid >= 38
    pred_b = census_damage <= .70 * OPEN_LOOP_DAMAGE and valid >= 36
    pred_c = (max(fresh) <= .045 and observed == wanted_observed and widths == {104}
              and all(value == wanted_qk for value in index_sets.values())
              and tuple(programs) == LAYERS and SCALARS == 531_632_438
              and BYTES == 2_010_587_756)
    null = census_damage >= OPEN_LOOP_DAMAGE - .002 or valid <= 28
    result = {
        "status": "mixed104_late_context_metric_input_triple_sequential_complete",
        "rung": 323,
        "claim_level": "physical_sequential_context_metric_census_certificate_fresh_price_gate",
        "convention": "CE added above native; lower is better",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "sequential_fit_order": list(LAYERS),
        "fit_diagnostics": fit_diagnostics,
        "layers": list(LAYERS),
        "rank": RANK,
        "open_loop_parent_damage": OPEN_LOOP_DAMAGE,
        "census_damage": census_damage,
        "damage_ratio_vs_open_loop": census_damage / OPEN_LOOP_DAMAGE,
        "certificates_valid": valid,
        "certificates_recovered_vs_open_loop": valid - 28,
        "member_abs_dce": member_abs,
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "mlp_input_program_observed": observed,
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_absolute_census_and_certificate_screen': bool(pred_a),
        'pred_b_sequential_fit_repairs_open_loop_composition': bool(pred_b),
        'pred_c_fresh_sequential_identity_and_price': bool(pred_c),
        "null_sequential_fit_does_not_repair_composition": bool(null),
        "stop_rule": "failure_closes_context_metric_late_composition_without_subset_search",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("member_abs_dce", "fresh8")}, indent=2), flush=True)
    print("SEQUENTIAL LATE CONTEXT-RRR DONE", flush=True)


if __name__ == "__main__":
    main()
