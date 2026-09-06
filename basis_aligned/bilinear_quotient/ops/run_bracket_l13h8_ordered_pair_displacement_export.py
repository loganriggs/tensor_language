#!/usr/bin/env python3
"""Export six fixed L13H8 opener-term displacement vectors from SELECT only."""

# BQGATE: EXPERIMENT pred_a_export_is_temporally_clean pred_b_exact_six_vector_artifact pred_c_source_instrument pred_d_fixed_price
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as authority
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as exact


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"


def _pad(rows, torch, device):
    endpoints = [(row, side) for row in rows for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals, sources = [], []
    for index, (row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
        sources.append(row[f"{side}_open_position"])
    return endpoints, tokens, torch.tensor(finals, device=device), torch.tensor(sources, device=device)


def evaluate(model, torch, F, facade):
    rows = authority.build_export_rows()
    endpoints, tokens, finals, sources = _pad(rows, torch, next(model.parameters()).device)
    native = exact.native_logits(model, tokens, torch, F)
    replay, factors = exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    arange = torch.arange(len(endpoints), device=tokens.device)
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    groups = defaultdict(list)
    for index, (row, side) in enumerate(endpoints):
        other = index ^ 1
        recipient = row[f"{side}_answer_id"]
        donor_side = "donor" if side == "base" else "base"
        donor = row[f"{donor_side}_answer_id"]
        groups[(recipient, donor)].append((terms[other] - terms[index]).detach().float().cpu())
    prototypes = {}
    for pair in authority.ORDERED_PAIRS:
        values = groups[pair]
        if len(values) != 24:
            raise ValueError("ordered-pair support changed")
        vector = torch.stack(values).mean(0)
        prototypes[f"{pair[0]}->{pair[1]}"] = {
            "recipient_closer_id": pair[0], "donor_closer_id": pair[1],
            "support": len(values), "coordinates": vector.tolist(),
            "l2_norm": float(vector.norm()),
        }
    replay_error = float((native - replay).abs().max())
    return prototypes, replay_error


def score(prototypes, replay_error):
    exact_keys = {f"{a}->{b}" for a, b in authority.ORDERED_PAIRS}
    finite_nonzero = all(len(value["coordinates"]) == 1152 and value["l2_norm"] > 0
                         and all(abs(float(x)) < float("inf") for x in value["coordinates"])
                         for value in prototypes.values())
    predictions = {
        "pred_a_export_is_temporally_clean": True,
        "pred_b_exact_six_vector_artifact": set(prototypes) == exact_keys and finite_nonzero
                                            and sum(len(x["coordinates"]) for x in prototypes.values()) == 6912,
        "pred_c_source_instrument": replay_error <= 1e-4,
        "pred_d_fixed_price": authority.compile_export_plan()["price"]["model_forwards"] == 1,
    }
    return {"native_replay_max_absolute_logit_error": replay_error,
            "predictions": predictions,
            "terminal": "prototype_artifact" if all(predictions.values()) else "invalid"}


def main():
    plan = authority.compile_export_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" \
            or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise RuntimeError(f"refusing overwrite {OUT}")
    torch, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    with torch.no_grad():
        prototypes, replay_error = evaluate(model, torch, F, facade)
    scored = score(prototypes, replay_error)
    payload = managed.atomic_create_json(OUT, {
        "schema": "bracket_l13h8_ordered_pair_displacement_artifact_v1",
        "candidate_id": authority.CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "prototypes": prototypes, **scored,
    })
    print(json.dumps({"terminal": scored["terminal"],
                      "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
