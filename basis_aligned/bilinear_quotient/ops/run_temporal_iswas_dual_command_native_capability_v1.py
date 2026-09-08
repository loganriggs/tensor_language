#!/usr/bin/env python3
"""Native-only gate for the frozen same-sequence temporal/is-was bank."""

# BQGATE: EXPERIMENT pred_a_authority_positions_finiteness_and_exact_price pred_b_fit_every_joint_cell_and_role_is_capable pred_c_holdout_every_joint_cell_and_role_is_capable pred_d_dual_command_license_is_issued
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_iswas_dual_command_v1 as authority
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_dual_command_native_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v1.py"
AUDIT = ROOT / "circuits/followups/temporal_iswas_joint_command_composition_compatibility_audit_v1_result.json"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_dual_command_native_capability_v1_result.json"
EXPECTED = {
    "prior": "7916e3cc1c47dfe12960e4ddce5a6a589582d7573c36aadd4cb216e913ea12fc",
    "builder": "9a5430ed0527f4ebd8bda64eace1e4c6ce6c61afe2e99c5c3143a8fcafc9fe3c",
    "audit": "dcae033128e12903242aca03acb1cd4a53b493dba9f9554fb826553519903563",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
PRICE = {"model_forwards": 1, "sequence_evaluations": 128,
         "scored_token_positions": 256, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_positions_finiteness_and_exact_price",
    "pred_b_fit_every_joint_cell_and_role_is_capable",
    "pred_c_holdout_every_joint_cell_and_role_is_capable",
    "pred_d_dual_command_license_is_issued",
)
EXPECTED_CONFIG = {"vocab_size": 50304, "n_layer": 18, "n_head": 9,
                   "n_embd": 1152, "squared_mlp": False, "bilinear": True,
                   "expansion_factor": 4, "gated": False,
                   "squared_attn": True, "bilinear_attn": True}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capability_cells(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["phase"], record["template_id"], record["cell"],
                 record["role"])].append(bool(record["correct"]))
    cells = []
    for key in sorted(grouped):
        outcomes = grouped[key]
        accuracy = sum(outcomes) / len(outcomes)
        cells.append({"phase": key[0], "template_id": key[1], "cell": key[2],
                      "role": key[3], "count": len(outcomes),
                      "correct_count": sum(outcomes), "accuracy": accuracy,
                      "threshold": 0.75,
                      "passed": len(outcomes) == 8 and accuracy >= 0.75})
    return cells


def _config(model):
    return {key: getattr(model.config, key) for key in EXPECTED_CONFIG}


def _native_logits(backend, tokens):
    torch, F, model = backend.torch, backend.F, backend.model
    width = model.config.n_embd
    x = F.rms_norm(model.transformer.wte(tokens), (width,))
    x0, first = x, None
    for block in model.transformer.h:
        x, first = block(x, first, x0)
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (width,))) / 30)).float()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "audit": AUDIT, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"dual-command capability authority changed: {observed}")
    prior, audit = json.loads(PRIOR.read_text()), json.loads(AUDIT.read_text())
    rows = authority.build_rows()
    if (prior.get("candidate_id") != authority.CAPABILITY_ID
            or audit.get("terminal") != "joint_population_and_basis_capture_required"
            or authority.validate_rows(rows) != authority.EXPECTED_AUTHORITY_SHA256):
        raise RuntimeError("dual-command population or prerequisite audit changed")
    endpoints = [(row, cell, row["endpoints"][cell]) for row in rows for cell in authority.CELLS]
    dryrun = {"candidate_id": authority.CAPABILITY_ID, "dryrun": True,
              "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
              "rows": len(rows), "sequences": len(endpoints), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    token_rows = [endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                  for _row, _cell, endpoint in endpoints]
    tokens = torch.tensor(token_rows, dtype=torch.long, device=backend.device)
    with torch.no_grad():
        logits = _native_logits(backend, tokens)

    records = []
    positions_valid = True
    for index, (row, cell, endpoint) in enumerate(endpoints):
        for role, position_key, answer_key, foil_key in (
            ("temporal", "temporal_position", "temporal_answer_id", "temporal_foil_id"),
            ("iswas", "iswas_position", "iswas_answer_id", "iswas_foil_id"),
        ):
            position = endpoint[position_key]
            answer_id, foil_id = endpoint[answer_key], endpoint[foil_key]
            positions_valid &= endpoint["ids"][position] == answer_id and position > 0
            answer_logit = float(logits[index, position - 1, answer_id].item())
            foil_logit = float(logits[index, position - 1, foil_id].item())
            margin = answer_logit - foil_logit
            records.append({"row_id": row["row_id"], "phase": row["phase"],
                            "template_id": row["template_id"], "group_number": row["group_number"],
                            "cell": cell, "role": role, "position": position,
                            "answer_id": answer_id, "foil_id": foil_id,
                            "answer_logit": answer_logit, "foil_logit": foil_logit,
                            "margin": margin, "correct": margin > 0})

    cells = capability_cells(records)
    finite = all(math.isfinite(record[key]) for record in records
                 for key in ("answer_logit", "foil_logit", "margin"))
    config = _config(backend.model)
    inventory_ok = (len(rows) == 32 and len(endpoints) == 128 and len(records) == 256
                    and len({row["row_id"] for row in rows}) == 32
                    and len(cells) == 32 and all(cell["count"] == 8 for cell in cells))
    pred_a = (observed == EXPECTED and config == EXPECTED_CONFIG and positions_valid
              and finite and inventory_ok and PRICE == prior["frozen_design"]["price"])
    pred_b = all(cell["passed"] for cell in cells if cell["phase"] == "FIT")
    pred_c = all(cell["passed"] for cell in cells if cell["phase"] == "HOLDOUT")
    licensed = [row["row_id"] for row in rows] if pred_a and pred_b and pred_c else []
    pred_d = pred_a and pred_b and pred_c and len(licensed) == 32
    predictions = dict(zip(PREDICTION_KEYS, (pred_a, pred_b, pred_c, pred_d)))
    terminal = ("invalid" if not pred_a else
                "dual_command_license_issued" if pred_d else "native_capability_null")
    result = {"schema": "temporal_iswas_dual_command_native_capability_result_v1",
              "candidate_id": authority.CAPABILITY_ID,
              "execution_policy": "managed_queue_only_capability_only",
              "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": observed,
              "population_sha256": authority.EXPECTED_AUTHORITY_SHA256,
              "model_config": config, "dryrun": dryrun,
              "capability_cells": cells, "native_records": records,
              "licensed_row_ids": licensed, "causal_outcomes_opened": False,
              "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("candidate_id", "capability_cells", "licensed_row_ids",
                       "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
