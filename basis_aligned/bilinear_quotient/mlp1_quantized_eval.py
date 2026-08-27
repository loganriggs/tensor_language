#!/usr/bin/env python3
"""Evaluate decoded canonical mlp1 tiered-table/ridge programs on pinned rows."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlp1_price_frontier as prices
import ship_mlp2_diag as ship
import tiered_table_codec as codec


ROWS = HERE / "fineweb_protocol_rows.pt"
OUTPUT = HERE / "mlp1_quantized_eval_results.json"
CONSTANTS = HERE / "opt_ablation_consts_all.pt"
STATE = {"mode": None, "tokens": None, "m0": None, "a1": None,
         "table": None, "weight": None, "constant": None}


def store(name):
    def hook(module, args, output):
        STATE[name] = output.detach().float()
    return hook


def replace(module, args, output):
    if STATE["mode"] is None:
        return None
    if STATE["mode"] == "constant":
        return STATE["constant"].to(output.dtype).expand_as(output)
    x = torch.cat([STATE["a1"], STATE["m0"]], -1).float()
    table = STATE["table"][STATE["tokens"]]
    replacement = table + x @ STATE["weight"]
    return replacement.to(output.dtype)


@torch.no_grad()
def evaluate(rows, mode=None, decoded=None):
    STATE["mode"] = mode
    if decoded is not None:
        STATE["table"] = decoded["table"].float().to(ship.DEV)
        STATE["weight"] = decoded["weight"].float().to(ship.DEV)
    total = 0.0; count = 0
    for start in range(0, rows.shape[0], 8):
        batch = rows[start:start+8].to(ship.DEV)
        STATE["tokens"] = batch[:, :-1].contiguous()
        logits = ship.fwd_arm(STATE["tokens"], frozenset(), {}).float()
        targets = batch[:, 1:].contiguous()
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 targets.reshape(-1), reduction="none").view_as(targets)
        mask = torch.ones_like(targets, dtype=torch.bool); mask[:, :64] = False
        total += float(losses[mask].sum()); count += int(mask.sum())
    STATE["mode"] = None; STATE["table"] = None; STATE["weight"] = None
    return total/count


@torch.no_grad()
def main():
    started = time.time()
    artifact = torch.load(prices.ARTIFACT, map_location="cpu", weights_only=False)
    price_result = json.loads(prices.OUTPUT.read_text())
    assert price_result["artifact_sha256"] == hashlib.sha256(
        prices.ARTIFACT.read_bytes()).hexdigest()
    frozen = torch.load(ROWS, map_location="cpu", weights_only=False)["rows"]
    rows = frozen["id_skip7000_rows960"][:, :ship.T+1].contiguous()
    STATE["constant"] = torch.load(CONSTANTS, map_location=ship.DEV,
                                    weights_only=False)["mlp1"].float()
    hooks = [ship.H[0].mlp.register_forward_hook(store("m0")),
             ship.H[1].attn.c_proj.register_forward_hook(store("a1")),
             ship.H[1].mlp.register_forward_hook(replace)]
    partial = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else None
    if partial and partial.get("partial") is True \
            and partial.get("family_id") == price_result["family_id"]:
        clean = partial["clean_ce"]; anchor = partial["optimal_constant_ce"]
        points = partial["points"]
        print(f"resuming after ranks {[point['ridge_rank'] for point in points]}", flush=True)
    else:
        clean = evaluate(rows)
        anchor = evaluate(rows, "constant")
        points = []
        print(f"clean {clean:.6f}; optimal constant {anchor:.6f}", flush=True)
    prepared = prices.prepare(artifact)
    price_by_rank = {point["ridge_rank"]: point for point in price_result["points"]}
    completed = {point["ridge_rank"] for point in points}
    for rank in prices.RANKS:
        if rank in completed:
            continue
        encoded, reproduced = prices.encode_rank(artifact, prepared, rank)
        expected = price_by_rank[rank]
        assert reproduced["canonical_bytes_hash"] == expected["canonical_bytes_hash"]
        decoded = codec.decode_tiered_affine(encoded)
        ce = evaluate(rows, "program", decoded)
        fidelity = 1-(ce-clean)/(anchor-clean)
        points.append({"ridge_rank": rank, "quotient_bits": expected["quotient_bits"],
                       "program_hash": expected["canonical_bytes_hash"], "ce": ce,
                       "delta_ce": ce-clean, "fidelity": fidelity})
        OUTPUT.write_text(json.dumps({
            "partial": True, "family_id": price_result["family_id"],
            "clean_ce": clean, "optimal_constant_ce": anchor, "points": points,
        }, indent=2)+"\n")
        print(f"rank {rank}: CE {ce:.6f}; fidelity {fidelity:.6f}", flush=True)
    for hook in hooks:
        hook.remove()
    fidelities = [point["fidelity"] for point in points]
    result = {
        "schema_version": 1, "family_id": price_result["family_id"],
        "protocol": {"rows_artifact": ROWS.name, "member": "id_skip7000_rows960",
                     "rows": 960, "sequence_length": ship.T,
                     "scored_position_minimum": 64, "no_evaluation_refit": True},
        "artifact_sha256": price_result["artifact_sha256"],
        "codec_source_sha256": price_result["codec_source_sha256"],
        "clean_ce": clean, "optimal_constant_ce": anchor,
        "anchor_delta_ce": anchor-clean, "points": points,
        "pred_rank0_fidelity_at_least_0_90": points[0]["fidelity"] >= .90,
        "pred_rank128_fidelity_at_least_0_94": points[3]["fidelity"] >= .94,
        "pred_full_fidelity_at_least_0_96": points[-1]["fidelity"] >= .96,
        "pred_no_downward_step_over_0_005": all(
            right >= left-.005 for left, right in zip(fidelities, fidelities[1:])),
        "runtime_s": time.time()-started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(f"wrote {OUTPUT} ({result['runtime_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
