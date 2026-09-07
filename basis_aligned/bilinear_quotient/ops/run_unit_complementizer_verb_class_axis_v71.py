#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, verb pairs and bars fixed before the run.
"""v71: is the verb_complementizer removal direction a verb-CLASS axis? Transfer to three unseen verb pairs.

v69/v70: the hub {06:03, 11:03, 07:08} direction (diff-in-means of remarked - wondered) damages the behaviour's own C
family (noted/replied -> that) by 0.31 on the that/whether margin. The reading offered was that the direction encodes
wh-taking vs that-taking verb class, used by every reporting verb. Causal test: fit the direction on the ORIGINAL pair's
A1 even rows and remove it on the odd rows of three lexical variants that the fit never saw -- asked/said,
inquired/insisted (v15 maps) and questioned/declared (v15 fourth map) -- for the hub and the v66 hub+8 set. Per-pair
refits (even rows of that pair) bound what a pair-specific direction could do; block cosines between the per-pair
directions say whether they are the same axis geometrically. Random rank-1 direction as floor.

REGISTERED BEFORE THE RUN (CE removal damage in nat on odd rows; own = original pair)
    pred_a_transfer_hub8   original-fit direction on each unseen pair >= 0.50 x own-pair damage for hub+8 (1.119 -> >= 0.56).
                           Worked: 0.70 True; 0.40 False.
    pred_b_transfer_hub    same for the 3-head hub (0.598 -> >= 0.30). Worked: 0.35 True; 0.20 False.
    pred_c_not_keyed       for hub+8, each pair's own refit <= 1.5 x the transferred damage on that pair. Worked: 0.75 vs 0.70 True; 1.2 vs 0.7 False.
    pred_d_random          random rank-1 on hub+8 <= 0.05 x own damage on the original pair. Worked: 0.004 True.
    pred_e_same_axis       mean block |cos| between the four per-pair hub+8 directions (6 pairs of pairs) >= 0.80.
                           Worked: 0.88 True; 0.60 False.
    Prior: a, b, e True if the verb-class reading is right (v15 found a common extraction axis across these maps);
    c True; d True. A False on a/b with e True would mean the axis is shared but the unseen verbs sit elsewhere on it.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_complementizer_verb_class_axis_v71_result.json"
SRC = ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json"
NAME = "verb_complementizer"
TRANSFER, KEY_MAX, RAND_FRAC, COS_MIN = 0.50, 1.5, 0.05, 0.80
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 5000


def _plan():
    return {"candidate_id": "corpus.unit_complementizer_verb_class_axis_v71", "pairs": ["orig"] + [str(m) for m in v15.SETS[NAME][2]] + [str(v15.SETS[NAME][3])],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    module, hub, maps, fourth = v15.SETS[NAME]
    hub = list(hub)
    hub8 = json.loads(SRC.read_text())["sets"][NAME]["final"]
    a1 = g.rows_of(module, "A1")
    pairs = {"orig": a1, "v1": g.lexical_variant(a1, maps[0]), "v2": g.lexical_variant(a1, maps[1]), "v3": g.lexical_variant(a1, fourth)}
    even = {k: g.prepare(backend, r[0::2]) for k, r in pairs.items()}
    odd = {k: g.prepare(backend, r[1::2]) for k, r in pairs.items()}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    R = {}
    for label, units in (("hub", hub), ("hub8", hub8)):
        q = {k: g.block_diff_in_means(backend, p, units) for k, p in even.items()}
        mu = {k: mu_of(p, units) for k, p in even.items()}
        dmg = {}
        for src in pairs:
            for tgt in pairs:
                if src == "orig" or src == tgt:
                    dmg[f"{src}->{tgt}"] = v51.summary(torch, v51.removal(backend, odd[tgt], units, q[src], mu[src]))
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        dmg["random->orig"] = v51.summary(torch, v51.removal(backend, odd["orig"], units, q_rand, mu["orig"]))
        keys = list(pairs)
        cos = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                cos[f"{a}|{b}"] = {f"{k[0]}:{k[1]}": abs(float((q[a][k][:, 0] * q[b][k][:, 0]).sum())) for k in q[a]}
        mean_cos = sum(sum(v.values()) / len(v) for v in cos.values()) / len(cos)
        R[label] = {"units": units, "damage": dmg, "block_abs_cos": cos, "mean_abs_cos": mean_cos}
        print(label, json.dumps({k: round(v["ce_damage"], 3) for k, v in dmg.items()}), "cos", round(mean_cos, 3))

    d = lambda lab, k: R[lab]["damage"][k]["ce_damage"]
    predictions = {
        'pred_a_transfer_hub8': all(d("hub8", f"orig->{t}") >= TRANSFER * d("hub8", "orig->orig") for t in ("v1", "v2", "v3")),
        'pred_b_transfer_hub': all(d("hub", f"orig->{t}") >= TRANSFER * d("hub", "orig->orig") for t in ("v1", "v2", "v3")),
        'pred_c_not_keyed': all(d("hub8", f"{t}->{t}") <= KEY_MAX * d("hub8", f"orig->{t}") for t in ("v1", "v2", "v3")),
        'pred_d_random': d("hub8", "random->orig") <= RAND_FRAC * d("hub8", "orig->orig"),
        'pred_e_same_axis': R["hub8"]["mean_abs_cos"] >= COS_MIN,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_complementizer_verb_class_axis_result_v1", "candidate_id": "corpus.unit_complementizer_verb_class_axis_v71",
              "pairs": {"v1": maps[0], "v2": maps[1], "v3": fourth}, "rows_odd": {k: len(p.rows) for k, p in odd.items()},
              "bars": {"transfer": TRANSFER, "key_max": KEY_MAX, "rand_frac": RAND_FRAC, "cos_min": COS_MIN},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
