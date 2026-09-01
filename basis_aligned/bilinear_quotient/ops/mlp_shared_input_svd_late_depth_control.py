"""RUNG 318B -- LATE-DEPTH FULL-RANK CONTROL AND p1024 FRONTIER.

The all-layer p512/p768 screen found broad success through MLP14 but extreme
damage at MLP15--17 despite nearly unchanged paired-weight energy capture.  Do
not interpret that depth boundary until the factorized hook is proven inert at
full input rank in the same late layers.

For layers 15,16,17, compare weight-SVD p1024 with p1152.  p1152 is an
instrument-only reconstruction control (it stores an unnecessary square
encoder and receives no compression credit).  Use exactly rung318's untouched
FineWeb rows 176:188 and WikiText skip80000 so changes isolate rank.

Frozen predictions
------------------
pred_a_full_rank_reconstruction_is_inert:
    Every p1152 late-layer damage is <=.005 on both corpora.
pred_b_p1024_repairs_each_late_layer_materially:
    At every late layer, p1024 mean nonnegative damage is <=75% of the frozen
    p768 mean nonnegative damage.
pred_c_two_late_layers_have_a_p1024_regime:
    At least 2/3 p1024 late layers add <=.08 on both corpora.

Null: any full-rank arm adds >=.03 on either corpus (instrument invalid), OR
all three p1024 arms add >=.10 on at least one corpus (late sensitivity is not
repaired at this rank).  This is diagnosis only; no late layer is selected for
composition from these results.
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
OUT = ROOT / "mlp_shared_input_svd_late_depth_control_results.json"
PARENT = ROOT / "mlp_shared_input_svd_all_layers_screen_results.json"
D = 1152
H = 4608
LAYERS = (15, 16, 17)
RANKS = (1024, 1152)
FINEWEB_SLICE = (176, 188)
WIKI_SKIP = 80000


def _mean_nonnegative(arm) -> float:
    return (max(0.0, arm["fineweb_damage"]) + max(0.0, arm["wikitext_damage"])) / 2


@torch.no_grad()
def _programs(mlp):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    gram = stacked.T @ stacked
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    vectors = vectors[:, torch.argsort(values, descending=True)]
    result = {}
    for rank in RANKS:
        basis = vectors[:, :rank]
        coefficient = stacked @ basis
        result[rank] = {
            "encoder": basis.T.contiguous(),
            "left": coefficient[:H].contiguous(),
            "right": coefficient[H:].contiguous(),
            "down": down,
            "bias": bias,
        }
    return result


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists()
        parent = json.loads(PARENT.read_text())
        assert parent["rung"] == 318 and parent["evaluation"]["wikitext_skip"] == WIKI_SKIP
        assert parent["evaluation"]["fineweb_rows_half_open"] == list(FINEWEB_SLICE)
        assert set(LAYERS) == {15, 16, 17} and RANKS[-1] == D
        print("LATE-DEPTH SVD CONTROL | dry run: parent, rows, ranks, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_signed_response_rank_screen import _wikitext_rows
    from mlp_shared_input_svd_all_layers_screen import _score
    from tier2_model import load_elriggs

    parent = json.loads(PARENT.read_text())
    cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fineweb = cached[FINEWEB_SLICE[0]:FINEWEB_SLICE[1], :257].long().contiguous()
    wikitext, fingerprint = _wikitext_rows(12, skip=WIKI_SKIP)
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    native = {
        "fineweb": _score(model, fineweb),
        "wikitext": _score(model, wikitext),
    }
    for corpus in native:
        if abs(native[corpus] - parent["native_ce"][corpus]) > 1e-5:
            raise SystemExit(f"INSTRUMENT FAIL: native {corpus} baseline changed")

    arms = {}
    for layer in LAYERS:
        programs = _programs(model.transformer.h[layer].mlp)
        arms[str(layer)] = {}
        for rank in RANKS:
            program = programs[rank]
            ce_fw = _score(model, fineweb, layer, program)
            ce_wt = _score(model, wikitext, layer, program)
            arm = {
                "fineweb_damage": ce_fw - native["fineweb"],
                "wikitext_damage": ce_wt - native["wikitext"],
            }
            arms[str(layer)][str(rank)] = arm
            print(f"L{layer} p{rank}: FW/WT {arm['fineweb_damage']:+.7f}/"
                  f"{arm['wikitext_damage']:+.7f}", flush=True)
        del programs
        torch.cuda.empty_cache()

    full_rank_inert = all(
        max(arms[str(layer)]["1152"].values()) <= .005 for layer in LAYERS
    )
    repaired = []
    for layer in LAYERS:
        old = parent["arms"][str(layer)]["768"]
        new = arms[str(layer)]["1024"]
        if _mean_nonnegative(new) <= .75 * _mean_nonnegative(old):
            repaired.append(layer)
    p1024_qualifying = [layer for layer in LAYERS
                        if max(arms[str(layer)]["1024"].values()) <= .08]
    pred_a = full_rank_inert
    pred_b = len(repaired) == len(LAYERS)
    pred_c = len(p1024_qualifying) >= 2
    null = (any(max(arms[str(layer)]["1152"].values()) >= .03 for layer in LAYERS)
            or all(max(arms[str(layer)]["1024"].values()) >= .10 for layer in LAYERS))
    result = {
        "status": "mlp_shared_input_svd_late_depth_control_complete",
        "rung": "318B",
        "claim_level": "late_depth_full_rank_instrument_and_capacity_diagnosis_only",
        "convention": "CE added above native; lower is better",
        "parent_result": PARENT.name,
        "native_ce": native,
        "wikitext_fingerprint": str(fingerprint),
        "arms": arms,
        "p1024_layers_repaired_by_at_least_25pct": repaired,
        "p1024_qualifying_layers": p1024_qualifying,
        'pred_a_full_rank_reconstruction_is_inert': bool(pred_a),
        'pred_b_p1024_repairs_each_late_layer_materially': bool(pred_b),
        'pred_c_two_late_layers_have_a_p1024_regime': bool(pred_c),
        "null_late_depth_screen_invalid_or_irreducibly_sensitive": bool(null),
        "decision_rule": "diagnosis_only_no_late_layer_selection_for_composition",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "p1024_repaired": repaired,
        "p1024_qualifying": p1024_qualifying,
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("LATE-DEPTH SVD CONTROL DONE", flush=True)


if __name__ == "__main__":
    main()
