#!/usr/bin/env python3
"""One-shot validation of frozen MLP4 z4 programs; no fit artifact is accessible."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ship_mlp2_diag as ship
from . import affine_codec
from . import mlp4_bilinear_residual_codec as native_codec
from . import mlp4_seeded_random_bilinear_codec as random_codec

ROWS = HERE / "mlp4_frontier_validation_rows.pt"
CANDIDATES = HERE / "mlp4_z4_candidate_bytes.pt"
INVENTORY = HERE / "mlp4_z4_candidate_inventory.json"
CONTROLS = HERE / "mlp4_z4_validation_controls.pt"
PROTOCOL = HERE / "mlp4_z4_validation_protocol.json"
OUTPUT = HERE / "mlp4_z4_validation_results.json"
D = 1152
BATCH = 4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_guard(protocol):
    peak = torch.cuda.max_memory_allocated(ship.DEV)/2**30
    if peak > protocol["resources"]["hard_abort_peak_gib"]:
        raise RuntimeError(f"peak allocation {peak:.3f} GiB exceeds contract")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    temperature = max(int(x) for x in query.stdout.splitlines() if x.strip())
    if temperature > protocol["resources"]["hard_abort_temperature_c"]:
        raise RuntimeError(f"GPU temperature {temperature} C exceeds contract")
    return peak, temperature


def prepare(candidate_id, encoded):
    if candidate_id.startswith("linear_"):
        p = affine_codec.decode_affine(encoded)
        return {"family": "linear", "weight": p["weight"].float().to(ship.DEV),
                "bias": p["bias"].float().to(ship.DEV)}
    if candidate_id.startswith("native_"):
        p = native_codec.decode(encoded)
        return {"family": "native_product",
                **{k: p[k].float().to(ship.DEV) for k in ("A", "B", "C", "bias")}}
    p = random_codec.decode(encoded)
    A, B = random_codec.feature_factors(
        p["semantic"]["seed"], p["din"], p["components"])
    return {"family": "seeded_random_product", "A": A.float().to(ship.DEV),
            "B": B.float().to(ship.DEV), "C": p["C"].float().to(ship.DEV),
            "bias": p["bias"].float().to(ship.DEV)}


def execute(program, z):
    if program["family"] == "linear":
        return z.float() @ program["weight"] + program["bias"]
    return ((z.float() @ program["A"])*(z.float() @ program["B"])) \
        @ program["C"] + program["bias"]


@torch.no_grad()
def forward_inline(idx, mode="live", program=None, mean=None):
    """Exact checkpoint forward except candidate/mean modes never call MLP4."""
    x = F.rms_norm(ship.m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for layer, block in enumerate(ship.H):
        mixed = block.lambdas[0]*x + block.lambdas[1]*x0
        attention, v1 = block.attn(F.rms_norm(mixed, (D,)), v1)
        x = mixed + attention
        z = F.rms_norm(x, (D,))
        if layer == 4 and mode == "program":
            mlp = execute(program, z).to(x.dtype)
        elif layer == 4 and mode == "mean":
            mlp = mean.to(device=x.device, dtype=x.dtype).expand_as(x)
        else:
            mlp = block.mlp(z)
        x = x + mlp
    return 30.0*torch.tanh(ship.m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def evaluate(rows, protocol, mode="live", program=None, mean=None):
    total = 0.0; count = 0; peak = 0.0; temperature = 0
    for batch_id, start in enumerate(range(0, rows.shape[0], BATCH)):
        batch = rows[start:start+BATCH].to(ship.DEV)
        logits = forward_inline(batch[:, :-1].contiguous(), mode, program, mean).float()
        targets = batch[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), reduction="none").view_as(targets)
        mask = torch.ones_like(targets, dtype=torch.bool)
        mask[:, :protocol["data"]["scored_position_minimum"]] = False
        total += float(loss[mask].sum()); count += int(mask.sum())
        if batch_id % protocol["resources"]["in_loop_guard_every_batches"] == 0:
            peak, temperature = resource_guard(protocol)
    return total/count, peak, temperature


@torch.no_grad()
def main():
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text())
    for filename, expected in protocol["pinned_artifacts"].items():
        if sha(HERE/filename) != expected:
            raise ValueError(f"pinned hash mismatch: {filename}")
    payload = torch.load(ROWS, map_location="cpu", weights_only=False)
    if payload["role"] != "validation" or payload["member"] != protocol["data"]["member"]:
        raise ValueError("validation role/member mismatch")
    rows = payload["rows"][:, :ship.T+1].contiguous()
    frozen = torch.load(CANDIDATES, map_location="cpu", weights_only=False)["encoded"]
    inventory = json.loads(INVENTORY.read_text())
    controls = torch.load(CONTROLS, map_location="cpu", weights_only=False)
    mean = controls["mean_output"]
    live, peak, temp = evaluate(rows, protocol)
    anchor, p, t = evaluate(rows, protocol, "mean", mean=mean)
    peak, temp = max(peak, p), max(temp, t)
    points = []
    inventory_rows = {row["candidate_id"]: row for row in inventory["candidates"]}
    for candidate_id in protocol["candidate_order"]:
        encoded = frozen[candidate_id]
        expected_hash = inventory_rows[candidate_id]["canonical_bytes_hash"].split(":")[-1]
        if hashlib.sha256(encoded).hexdigest() != expected_hash:
            raise ValueError(f"candidate hash mismatch: {candidate_id}")
        program = prepare(candidate_id, encoded)
        ce, p, t = evaluate(rows, protocol, "program", program=program)
        point = {"candidate_id": candidate_id,
                 "family": inventory_rows[candidate_id]["family"],
                 "capacity": inventory_rows[candidate_id]["capacity"],
                 "program_hash": "sha256:"+expected_hash,
                 "ce": ce, "delta_ce": ce-live,
                 "fidelity": 1-(ce-live)/(anchor-live)}
        points.append(point); peak, temp = max(peak, p), max(temp, t)
        print(f"{candidate_id}: CE {ce:.6f}, fidelity {point['fidelity']:.4f}", flush=True)
        del program
        torch.cuda.empty_cache()
    by_id = {point["candidate_id"]: point for point in points}
    pairs = inventory["native_random_actual_bit_pairings"]
    pair_results = [{"native": pair["native_candidate_id"],
                     "random": pair["random_candidate_id"],
                     "native_wins": by_id[pair["native_candidate_id"]]["ce"]
                                    < by_id[pair["random_candidate_id"]]["ce"]}
                    for pair in pairs]
    family_ces = [[p["ce"] for p in points if p["family"] == family]
                  for family in ("linear", "native_product", "seeded_random_product")]
    result = {"schema_version": 1, "protocol_id": protocol["protocol_id"],
              "partial": False, "live_ce": live, "mean_ce": anchor,
              "anchor_delta_ce": anchor-live, "points": points,
              "pair_results": pair_results,
              "gates": {
                  "native_wins_at_least_4_of_5": sum(p["native_wins"] for p in pair_results) >= 4,
                  "no_adjacent_family_ce_increase_over_0_01": all(
                      right <= left+.01 for ces in family_ces
                      for left, right in zip(ces, ces[1:])),
                  "full_affine_fidelity_at_least_0_65": by_id["linear_r1152"]["fidelity"] >= .65,
              },
              "resources": {"peak_allocated_gib": peak, "maximum_temperature_c": temp,
                            "runtime_s": time.time()-started},
              "interpretation": {"held_out_lane_only": True, "promotion": False,
                                 "validation_refit": False}}
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(f"wrote {OUTPUT.name} in {result['resources']['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
