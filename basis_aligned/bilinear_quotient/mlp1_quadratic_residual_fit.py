#!/usr/bin/env python3
"""Fit natural-data scalar weights for exact mlp1 bilinear hidden terms."""

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
import mlp1_price_frontier as base_prices
import ship_mlp2_diag as ship
import tiered_table_codec as table_codec

ROWS = HERE / "fineweb_protocol_rows.pt"
ROWS_MANIFEST = HERE / "fineweb_protocol_rows_manifest.json"
OUTPUT = HERE / "mlp1_quadratic_residual_artifact.pt"
MANIFEST = HERE / "mlp1_quadratic_residual_artifact_manifest.json"
BATCH = 8
RANKS = (8, 16, 32, 64, 128)
RIDGE_FRACTION = 1e-3


@torch.no_grad()
def run_early(idx, capture):
    x = F.rms_norm(ship.m.transformer.wte(idx), (ship.D,)); x0 = x
    x, v0 = ship.H[0](x, None, x0)
    ship.H[1](x, v0, x0)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def capture_mlp(capture):
    def hook(_module, args, output):
        capture["z"] = args[0].detach().float()
        capture["y"] = output.detach().float()
    return hook


@torch.no_grad()
def main():
    started = time.time()
    frozen = torch.load(ROWS, map_location="cpu", weights_only=False)
    rows = frozen["rows"]["fit_skip80_rows960"][:, :ship.T+1].contiguous()
    rows_manifest = json.loads(ROWS_MANIFEST.read_text())
    base_artifact = torch.load(base_prices.ARTIFACT, map_location="cpu", weights_only=False)
    prepared = base_prices.prepare(base_artifact)
    encoded, reproduced = base_prices.encode_rank(base_artifact, prepared, 1152)
    expected = json.loads(base_prices.OUTPUT.read_text())["points"][-1]
    assert reproduced["canonical_bytes_hash"] == expected["canonical_bytes_hash"]
    base = table_codec.decode_tiered_affine(encoded)
    table = base["table"].float().to(ship.DEV)
    weight = base["weight"].float().to(ship.DEV)
    mlp = ship.H[1].mlp
    down = mlp.Down.weight.detach().float()
    down_norm2 = down.square().sum(0)
    hidden_dim = down.shape[1]
    b = torch.zeros(hidden_dim, device=ship.DEV)
    diagonal = torch.zeros(hidden_dim, device=ship.DEV)
    capture = {}
    hooks = [ship.H[0].mlp.register_forward_hook(
                 lambda _m, _a, o: capture.__setitem__("m0", o.detach().float())),
             ship.H[1].attn.c_proj.register_forward_hook(
                 lambda _m, _a, o: capture.__setitem__("a1", o.detach().float())),
             mlp.register_forward_hook(capture_mlp(capture))]
    for start in range(0, rows.shape[0], BATCH):
        idx = rows[start:start+BATCH, :-1].to(ship.DEV).contiguous()
        run_early(idx, capture)
        tokens = idx.reshape(-1)
        x = torch.cat([capture["a1"], capture["m0"]], -1).reshape(-1, 2*ship.D)
        residual = capture["y"].reshape(-1, ship.D) - (table[tokens] + x@weight)
        z = capture["z"].reshape(-1, ship.D)
        hidden = mlp.Left(z).float()*mlp.Right(z).float()
        b += (hidden*(residual@down)).sum(0)
        diagonal += hidden.square().sum(0)*down_norm2
    score = b.square()/diagonal.clamp_min(torch.finfo(diagonal.dtype).tiny)
    selected = torch.topk(score, max(RANKS)).indices
    print(f"screen pass complete; top score {float(score[selected[0]]):.6g}", flush=True)

    selected_down = down[:, selected]
    output_gram = selected_down.T@selected_down
    gram = torch.zeros(max(RANKS), max(RANKS), device=ship.DEV)
    rhs = torch.zeros(max(RANKS), device=ship.DEV)
    for start in range(0, rows.shape[0], BATCH):
        idx = rows[start:start+BATCH, :-1].to(ship.DEV).contiguous()
        run_early(idx, capture)
        tokens = idx.reshape(-1)
        x = torch.cat([capture["a1"], capture["m0"]], -1).reshape(-1, 2*ship.D)
        residual = capture["y"].reshape(-1, ship.D) - (table[tokens] + x@weight)
        z = capture["z"].reshape(-1, ship.D)
        hidden = (mlp.Left(z).float()*mlp.Right(z).float())[:, selected]
        gram += (hidden.T@hidden)*output_gram
        rhs += (hidden*(residual@selected_down)).sum(0)
    for hook in hooks:
        hook.remove()
    alpha_by_rank = {}
    regularizer_by_rank = {}
    for rank in RANKS:
        local = gram[:rank, :rank]; local_rhs = rhs[:rank]
        regularizer = RIDGE_FRACTION*float(torch.diagonal(local).mean())
        alpha = torch.linalg.solve(local + regularizer*torch.eye(
            rank, device=ship.DEV), local_rhs)
        alpha_by_rank[rank] = alpha.cpu()
        regularizer_by_rank[rank] = regularizer
        print(f"rank {rank}: alpha norm {float(alpha.norm()):.6f}", flush=True)
    selected_cpu = selected.cpu()
    artifact = {
        "schema_version": 1,
        "semantic_id": "mlp1|tier2000+full-linear-ridge+selected-exact-bilinear-terms[z1]",
        "base_program_hash": expected["canonical_bytes_hash"],
        "protocol_rows_artifact_sha256": rows_manifest["artifact_sha256"],
        "fit_rows_tensor_sha256": rows_manifest["members"]["fit_skip80_rows960"][
            "tensor_sha256"],
        "selected_hidden_indices": selected_cpu,
        "left": mlp.Left.weight.detach().float().cpu()[selected_cpu],
        "right": mlp.Right.weight.detach().float().cpu()[selected_cpu],
        "down": down.T.cpu()[selected_cpu],
        "screen_score": score[selected].cpu(),
        "alpha_by_rank": alpha_by_rank,
        "regularizer_by_rank": regularizer_by_rank,
        "ranks": list(RANKS), "ridge_fraction": RIDGE_FRACTION,
        "fit_observations": int(rows.shape[0]*ship.T),
    }
    torch.save(artifact, OUTPUT)
    manifest = {
        "schema_version": 1, "artifact": OUTPUT.name,
        "artifact_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "protocol_rows_artifact_sha256": artifact["protocol_rows_artifact_sha256"],
        "fit_rows_tensor_sha256": artifact["fit_rows_tensor_sha256"],
        "base_program_hash": artifact["base_program_hash"],
        "ranks": list(RANKS), "fit_observations": artifact["fit_observations"],
        "selection": "diagonal_residual_explanation_then_exact_selected_gram",
        "no_ood_fit": True,
        "tensor_sha256": {name: tensor_hash(value) for name, value in artifact.items()
                          if isinstance(value, torch.Tensor)},
        "runtime_s": time.time()-started,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"wrote {OUTPUT} ({manifest['runtime_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
