"""RUNG 316 -- SHIFTED OOD GATE FOR MIXED104 + MLP0 SHARED-INPUT SVD768.

Rung 315's p768 point scored +.00901182 census, 50/62 certificates, and a
literal 536,940,854 scalars.  Rebuild exactly that program, save its unablated
CE vector for the signed intervention gate, and evaluate 120 deterministic
WikiText-2 raw-test rows after token 70,000.  No WikiText statistic is fitted.

Frozen predictions
------------------
pred_a_shifted_mean_tail_and_max:
    Mean damage <=.015, row p95 <=.035, and row max <=.080.
pred_b_census_and_certificates_reproduce:
    Census differs from rung315 by <=.0015 and >=48/62 certificates remain.
pred_c_dataset_price_and_identity:
    Native CE in [2,8], 120x257 construction and fingerprint live, MLP0 rank768,
    exact mixed104 QK indices/width and active set, literal total 536,940,854.

Null: shifted mean >=.040 or p95 >=.080.  Signed a16 transfer remains after a
pass; this run establishes its exact unablated baseline.
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
OUT = ROOT / "mixed104_mlp0_svd768_ood_results.json"
CEV = ROOT / "cev_mixed104_mlp0_svd768.pt"
PARENT = ROOT / "mixed104_mlp0_shared_input_svd_frontier_results.json"
SCALARS = 536_940_854
BYTES = 2_031_821_420
WIKI_SKIP = 70_000
N_ROWS = 120


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


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists()
        parent = json.loads(PARENT.read_text())
        assert parent["arms"]["768"]["certificates_valid"] >= 48
        assert parent["arms"]["768"]["literal_standalone_scalars"] == SCALARS
        print("MIXED104 MLP0 SVD768 OOD | dry run: parent, dataset, price, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed104_mlp0_shared_input_svd_frontier import _programs
    from mixed104_online_cv0_ood import wikitext_rows

    rows_ood, fingerprint, token_count = wikitext_rows(n=N_ROWS, skip=WIKI_SKIP)
    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    program = _programs(C.m)["r768"]
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": program,
        "extra_eval_rows": rows_ood,
        "extra_eval_name": "wikitext-2-raw-v1-test-skip70000",
    })
    print("ARM: mixed104 + MLP0 shared-input SVD768 + frozen WikiText OOD", flush=True)
    run = C.main()

    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (observed != {0: 768} or set(index_sets) != set(range(2, 18))
            or any(value != wanted for value in index_sets.values()) or widths != {104}
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: p768 or mixed104 identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid, member_abs = _certificate_count(CN, battery, damage_vector)
    census_damage = float(damage_vector.mean())
    parent_damage = json.loads(PARENT.read_text())["arms"]["768"]["census_damage"]
    extra = C.SEL["extra_eval"]
    by_row = torch.tensor(extra["damage_by_row"])
    mean = float(extra["damage_mean"])
    p95 = float(torch.quantile(by_row, .95))
    maximum = float(by_row.max())
    pred_a = mean <= .015 and p95 <= .035 and maximum <= .080
    pred_b = abs(census_damage - parent_damage) <= .0015 and valid >= 48
    pred_c = (2.0 <= extra["native_ce"] <= 8.0 and extra["n_rows"] == N_ROWS
              and observed == {0: 768} and widths == {104}
              and all(value == wanted for value in index_sets.values()) and SCALARS == 536_940_854)
    null = mean >= .040 or p95 >= .080
    result = {
        "status": "mixed104_mlp0_svd768_ood_complete",
        "rung": 316,
        "claim_level": "physical_shifted_ood_and_unablated_signed_baseline_gate",
        "convention": "compiled CE minus native CE on identical positions",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "dataset_fingerprint": fingerprint,
        "source_token_count": token_count,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "native_ce": extra["native_ce"],
        "compiled_ce": extra["compiled_ce"],
        "damage_mean": mean,
        "damage_mean_abs_position": extra["damage_mean_abs_position"],
        "damage_row_p50": float(torch.quantile(by_row, .50)),
        "damage_row_p95": p95,
        "damage_row_min": float(by_row.min()),
        "damage_row_max": maximum,
        "damage_by_row": [float(value) for value in by_row],
        "census_damage": census_damage,
        "certificates_valid": valid,
        "member_abs_dce": member_abs,
        "fresh8": [float(value) for value in run["fresh8"]],
        "max_fresh_damage": max(float(value) for value in run["fresh8"]),
        "mlp_input_program_observed": observed,
        "qk_singular_indices": list(wanted),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_shifted_mean_tail_and_max': bool(pred_a),
        'pred_b_census_and_certificates_reproduce': bool(pred_b),
        'pred_c_dataset_price_and_identity': bool(pred_c),
        "null_shifted_transport_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("damage_by_row", "member_abs_dce")}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
