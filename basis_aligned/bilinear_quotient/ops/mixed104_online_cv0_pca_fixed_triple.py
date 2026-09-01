"""RUNG 310 -- PHYSICAL COMPOSITION OF MIXED104 WITH FIXED PCA MLP TRIPLE.

Compose the adopted 539,595,062-scalar mixed104 online-c_v0 program with the
large-sample-stable rank-256 MLP-output PCA projections at layers {0,8,17}.
PCA bases are fit on the same frozen 16 FineWeb rows used by rungs 306--309,
then installed only after all mixed104 dictionaries are frozen and only on the
candidate side of paired evaluations.

Each semantic replacement stores native Left/Right, Q^T Down, Q, and adjusted
constant instead of native Left/Right/Down/bias, saving 3,833,856 scalars. The
proposed disjoint bill is therefore

    539,595,062 - 3*3,833,856 = 528,093,494 scalars
    2,042,438,252 - 3*3,833,856*4 = 1,996,431,980 raw bytes.

Frozen predictions
------------------
pred_a_combined_census_and_certificates:
    Combined census damage <=.075 and >=10/62 certificates remain valid.
pred_b_mlp_surcharge_composes:
    Census surcharge over adopted mixed104's +.00469195 is in [.045,.075].
pred_c_fresh_and_identity:
    Every fresh8 damage <=.10; observed MLP projectors are exactly layers
    {0,8,17} rank256; QK indices/factor width and mixed active-set tripwires
    remain exact; no a0/a1v/tailE table is active.

Null: combined census damage >=.12 or <=3 certificates.  This is a physical
composition gate, not adoption; exact dependency auditing and shifted OOD are
still required after a pass.
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
OUT = ROOT / "mixed104_online_cv0_pca_fixed_triple_results.json"
CEV = ROOT / "cev_mixed104_online_cv0_pca_fixed_triple.pt"
LAYERS = (0, 8, 17)
RANK = 256
FIT_ROWS = 16
ADOPTED_SCALARS = 539_595_062
ADOPTED_BYTES = 2_042_438_252
ADOPTED_DAMAGE = 0.00469195
SAVING_EACH = 3_833_856


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (ROOT / "circuits/BATTERY.json", ROOT / "census_state_diverse.pt",
                     ROOT / "mixed104_online_cv0_results.json",
                     ROOT / ".rowcache/fineweb_n480_skip80.pt"):
            assert path.exists(), path
        assert ADOPTED_SCALARS - 3 * SAVING_EACH == 528_093_494
        assert ADOPTED_BYTES - 3 * SAVING_EACH * 4 == 1_996_431_980
        print("MIXED104 + PCA FIXED TRIPLE | dry run: parent, price, active sets, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    import mlp_activation_pca_four_layer_composition as pca_base
    from mlp0_signed_response_rank_screen import _manual_logits

    CN.use_state("census_state_diverse.pt")
    rows = CN.rows().cpu()
    base_ce = CN.base_ce().float().cpu()
    nflat = CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    fit = pca_base._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    pca = pca_base._fit_pca(pca_base._capture_outputs(C.m, fit, _manual_logits))
    projectors = {layer: pca[layer] for layer in LAYERS}
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_projectors": projectors,
    })
    print("ARM: mixed104 online-c_v0 + rank256 MLP PCA at {0,8,17}", flush=True)
    run = C.main()

    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_projectors_observed", {}).items()}
    if observed != {0: RANK, 8: RANK, 17: RANK}:
        raise SystemExit(f"INSTRUMENT FAIL: MLP projectors observed {observed}")
    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    if set(index_sets) != set(range(2, 18)) or any(value != wanted for value in index_sets.values()):
        raise SystemExit("INSTRUMENT FAIL: QK index sets changed")
    qk = C.SEL.get("_QKR", {})
    factor_widths = {int(factor[0].shape[1]) for heads in qk.values()
                     for factors in heads.values() for factor in factors}
    if factor_widths != {104}:
        raise SystemExit(f"INSTRUMENT FAIL: QK widths {factor_widths}")
    active = tuple(C.SEL.get("_ORDER2", ()))
    if any(name in active for name in ("a0", "a1v", "tailE")):
        raise SystemExit(f"INSTRUMENT FAIL: forbidden active object {active}")
    if not all(name in active for name in
               ("m0E", "m1", "m2E", "m3E", "c4", "c5", "c6", "c7", "c8", "c9")):
        raise SystemExit(f"INSTRUMENT FAIL: incomplete mixed active set {active}")

    cev = C.SEL["cev"].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    member_abs = {}
    valid = 0
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage_vector[member].abs().mean())
        member_abs[tag] = round(value, 7)
        valid += int(value < 0.5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])

    damage = float(damage_vector.mean())
    surcharge = damage - ADOPTED_DAMAGE
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = damage <= 0.075 and valid >= 10
    pred_b = 0.045 <= surcharge <= 0.075
    pred_c = bool(max(fresh) <= 0.10 and observed == {0: 256, 8: 256, 17: 256}
                  and factor_widths == {104} and all(value == wanted for value in index_sets.values())
                  and not any(name in active for name in ("a0", "a1v", "tailE")))
    null = damage >= 0.12 or valid <= 3
    result = {
        "status": "mixed104_online_cv0_pca_fixed_triple_complete",
        "rung": 310,
        "claim_level": "physical_composition_census_certificates_fresh_gate_only",
        "convention": "CE added above native model; lower is better",
        "active_mixed_replacements": list(active),
        "mlp_projectors_observed": observed,
        "qk_singular_indices": list(wanted),
        "qk_factor_widths": sorted(factor_widths),
        "census_damage": damage,
        "mlp_surcharge_over_adopted_mixed104": surcharge,
        "certificates_valid": valid,
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "member_abs_dce": member_abs,
        "price": {"parent_scalars": ADOPTED_SCALARS, "parent_raw_bytes": ADOPTED_BYTES,
                  "saving_scalars": 3 * SAVING_EACH,
                  "literal_standalone_scalars": ADOPTED_SCALARS - 3 * SAVING_EACH,
                  "literal_raw_tensor_bytes": ADOPTED_BYTES - 3 * SAVING_EACH * 4},
        'pred_a_combined_census_and_certificates': bool(pred_a),
        'pred_b_mlp_surcharge_composes': bool(pred_b),
        'pred_c_fresh_and_identity': bool(pred_c),
        "null_combined_program_unusable": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "member_abs_dce"},
                     indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
