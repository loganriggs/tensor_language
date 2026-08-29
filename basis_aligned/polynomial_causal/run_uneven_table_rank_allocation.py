#!/usr/bin/env python3
"""Run the preregistered uneven-rank settled compiler on existing discovery roles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BQ))

import uneven_table_rank_allocation as allocator  # noqa: E402
from bilin18_joint_removal import m, DEV  # noqa: E402


D = 1152
T = 256
V = 50257
NCOV = 5419
RIDGE = 1e-2
RANK_GRID = tuple(range(64, 1153, 64))
PREREG = HERE / "UNEVEN_TABLE_RANK_ALLOCATION_PREREG_2026-08-29.md"
PURE = HERE / "uneven_table_rank_allocation.py"
TEST = HERE / "test_uneven_table_rank_allocation.py"
RUNNER = Path(__file__).resolve()
RESULT = HERE / "uneven_table_rank_allocation_results.json"
FIT_ROWS = BQ / ".rowcache" / "fineweb_n96_skip80.pt"
EVAL_ROLES = {
    "skip7000": BQ / ".rowcache" / "fineweb_n192_skip7000.pt",
    "skip11000": BQ / ".rowcache" / "fineweb_n192_skip11000.pt",
    "skip1200": BQ / ".rowcache" / "fineweb_n96_skip1200.pt",
}
UNIFORM_ANCHORS = {
    "skip7000": 5.98100,
    "skip11000": 5.94957,
    "skip1200": 5.96977,
}
SOURCE_PATHS = (PREREG, PURE, TEST, RUNNER)
STATE: dict[str, torch.Tensor] = {}
SEEN: torch.Tensor
H = m.transformer.h


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_closure() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
            check=True, stdout=subprocess.DEVNULL,
        )
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise RuntimeError(f"behavior-bearing source differs from HEAD: {relative}")
        hashes[relative] = file_sha256(path)
    return commit, hashes


def load_rows(path: Path, expected: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload["rows"] if isinstance(payload, dict) else payload
    rows = rows[:, :T + 1].contiguous()
    if rows.dtype != torch.long or rows.shape != (expected, T + 1):
        raise RuntimeError(f"row schema changed at {path}: {tuple(rows.shape)} {rows.dtype}")
    return rows


def sites() -> tuple[tuple[str, int], ...]:
    return tuple((kind, layer) for kind in ("mlp", "attn") for layer in range(18))


def site_name(site: tuple[str, int]) -> str:
    return f"{site[0]}{site[1]}"


def module(site: tuple[str, int]):
    return H[site[1]].mlp if site[0] == "mlp" else H[site[1]].attn


def row_hook(rows: torch.Tensor):
    def hook(_module, _args, output):
        native = output[0] if isinstance(output, tuple) else output
        replacement = rows[STATE["tokens"].reshape(-1)].reshape(native.shape).to(native.dtype)
        if isinstance(output, tuple):
            return (replacement,) + tuple(output[1:])
        return replacement
    return hook


@torch.no_grad()
def forward_logits(tokens: torch.Tensor, hooks=()) -> torch.Tensor:
    handles = [module(site).register_forward_hook(hook) for site, hook in hooks]
    STATE["tokens"] = tokens
    try:
        x = F.rms_norm(m.transformer.wte(tokens), (D,))
        x0 = x
        v1 = None
        for block in H:
            x, v1 = block(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def score_rows(rows: torch.Tensor, full_rows: Mapping[tuple[str, int], torch.Tensor]) -> dict[str, float]:
    totals = {name: [0.0, 0] for name in ("all", "covered", "uncovered")}
    hooks = [(site, row_hook(full_rows[site])) for site in sites()]
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        tokens = batch[:, :-1].to(DEV).contiguous()
        logits = forward_logits(tokens, hooks)
        targets = batch[:, 1:].to(DEV)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="none",
        ).reshape(targets.shape)[:, 64:].double()
        covered = SEEN[tokens[:, 64:]]
        for name, mask in (
            ("all", torch.ones_like(covered)), ("covered", covered), ("uncovered", ~covered),
        ):
            totals[name][0] += float(losses[mask].sum())
            totals[name][1] += int(mask.sum())
    return {name: total / count for name, (total, count) in totals.items()}


@torch.no_grad()
def capture_decompositions(covered_tokens: torch.Tensor) -> tuple[
    dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    table = {site: torch.empty(len(covered_tokens), D, device=DEV) for site in sites()}
    captured: dict[tuple[str, int], torch.Tensor] = {}

    def capture(site):
        def hook(_module, _args, output):
            captured[site] = (output[0] if isinstance(output, tuple) else output)[:, 0].float()
        return hook

    capture_hooks = [(site, capture(site)) for site in sites()]
    for start in range(0, len(covered_tokens), 256):
        token_batch = covered_tokens[start:start + 256].to(DEV).unsqueeze(1)
        forward_logits(token_batch, capture_hooks)
        for site in sites():
            table[site][start:start + len(token_batch)] = captured[site]

    decompositions = {}
    spectra = {}
    for index, site in enumerate(sites(), 1):
        values = table[site].double()
        mean = values.mean(0, keepdim=True)
        u, singular, vh = torch.linalg.svd(values - mean, full_matrices=False)
        decompositions[site] = (
            mean.float(), u.float(), singular.float(), vh.float(),
        )
        spectra[site_name(site)] = singular.detach().cpu().double().square()
        print(f"  SVD {site_name(site):6s} [{index}/36]", flush=True)
    return decompositions, spectra


def reconstruct(
    decomposition: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], rank: int,
) -> torch.Tensor:
    mean, u, singular, vh = decomposition
    return (mean + (u[:, :rank] * singular[:rank]) @ vh[:rank]).float()


@torch.no_grad()
def build_program_rows(
    ranks: Mapping[str, int],
    decompositions: Mapping[
        tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    covered_tokens: torch.Tensor,
    uncovered_tokens: torch.Tensor,
    solve_left: torch.Tensor,
    uncovered_embeddings: torch.Tensor,
) -> dict[tuple[str, int], torch.Tensor]:
    output = {}
    for index, site in enumerate(sites(), 1):
        rank = int(ranks[site_name(site)])
        table = reconstruct(decompositions[site], rank)
        weight = solve_left @ table.double()
        u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        map_rank = min(rank, allocator.MAP_CAP)
        weight = (u[:, :map_rank] * singular[:map_rank]) @ vh[:map_rank]
        rows = torch.empty(V, D, device=DEV)
        rows[covered_tokens] = table
        rows[uncovered_tokens] = (uncovered_embeddings @ weight).float()
        output[site] = rows
        print(
            f"    built {site_name(site):6s} table-r{rank} map-r{map_rank} [{index}/36]",
            flush=True,
        )
    return output


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def main() -> None:
    if RESULT.exists():
        raise RuntimeError(f"refusing to overwrite result: {RESULT}")
    source_commit, source_hashes = verify_source_closure()
    started = time.time()
    fit = load_rows(FIT_ROWS, 96)
    fit_tokens = fit[:, :T].reshape(-1)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit_tokens] = True
    if int(seen_cpu.sum()) != NCOV:
        raise RuntimeError("fit coverage changed")
    global SEEN
    SEEN = seen_cpu.to(DEV)
    covered_tokens = seen_cpu.nonzero(as_tuple=True)[0].to(DEV)
    uncovered_tokens = (~seen_cpu).nonzero(as_tuple=True)[0].to(DEV)

    print("UNEVEN TABLE-RANK ALLOCATION | discovery only", flush=True)
    decompositions, spectra = capture_decompositions(covered_tokens)
    budget = 36 * allocator.site_cost(512)
    normalized = allocator.allocate(spectra, RANK_GRID, budget, normalized=True)
    raw = allocator.allocate(spectra, RANK_GRID, budget, normalized=False)
    shifted = allocator.type_shifted_null(normalized.ranks)
    uniform = {site_name(site): 512 for site in sites()}
    arms = {
        "uniform_r512": uniform,
        "normalized_fit_energy": normalized.ranks,
        "type_shifted_null": shifted,
        "raw_fit_energy_diagnostic": raw.ranks,
    }
    for name, ranks in arms.items():
        cost = allocator.allocation_cost(ranks)
        if cost > budget:
            raise RuntimeError(f"arm exceeds budget: {name}")
        print(f"  {name}: cost={cost / 1e6:.4f}M ranks={ranks}", flush=True)

    embeddings = m.transformer.wte.weight.detach().float().double()
    covered_embeddings = embeddings[covered_tokens]
    uncovered_embeddings = embeddings[uncovered_tokens]
    ridge = RIDGE * torch.eye(D, dtype=torch.float64, device=DEV) * (NCOV / D)
    solve_left = torch.linalg.solve(
        covered_embeddings.T @ covered_embeddings + ridge, covered_embeddings.T,
    )
    role_rows = {
        "skip7000": load_rows(EVAL_ROLES["skip7000"], 192),
        "skip11000": load_rows(EVAL_ROLES["skip11000"], 192),
        "skip1200": load_rows(EVAL_ROLES["skip1200"], 96),
    }
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for arm, ranks in arms.items():
        print(f"\n  BUILD {arm}", flush=True)
        full_rows = build_program_rows(
            ranks, decompositions, covered_tokens, uncovered_tokens,
            solve_left, uncovered_embeddings,
        )
        scores[arm] = {}
        for role, rows in role_rows.items():
            scores[arm][role] = score_rows(rows, full_rows)
            print(f"    {role}: {scores[arm][role]}", flush=True)
        del full_rows
        torch.cuda.empty_cache()

    primary_gain = {
        role: scores["uniform_r512"][role]["all"]
        - scores["normalized_fit_energy"][role]["all"]
        for role in EVAL_ROLES
    }
    null_gain = {
        role: scores["type_shifted_null"][role]["all"]
        - scores["normalized_fit_energy"][role]["all"]
        for role in EVAL_ROLES
    }
    anchor_error = {
        role: abs(scores["uniform_r512"][role]["all"] - UNIFORM_ANCHORS[role])
        for role in EVAL_ROLES
    }
    predictions = {
        "pred_a_primary_beats_uniform_0p005_all_roles": all(
            value >= 0.005 for value in primary_gain.values()
        ),
        "pred_b_primary_beats_shifted_0p005_all_roles": all(
            value >= 0.005 for value in null_gain.values()
        ),
        "pred_c_controls": max(anchor_error.values()) <= 0.002
        and int(SEEN.sum()) == NCOV
        and all(allocator.allocation_cost(ranks) <= budget for ranks in arms.values()),
    }
    payload = {
        "schema_version": 1,
        "status": "discovery_complete",
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "inputs": {
            "fit_rows_sha256": file_sha256(FIT_ROWS),
            "evaluation_rows_sha256": {
                role: file_sha256(path) for role, path in EVAL_ROLES.items()
            },
            "coverage": NCOV,
            "rank_grid": list(RANK_GRID),
            "budget_values": budget,
            "budget_M": budget / 1e6,
        },
        "allocations": {
            name: {
                "ranks": ranks,
                "cost_values": allocator.allocation_cost(ranks),
                "cost_M": allocator.allocation_cost(ranks) / 1e6,
            }
            for name, ranks in arms.items()
        },
        "fit_objectives": {
            "normalized_utility": normalized.utility,
            "raw_utility": raw.utility,
        },
        "scores": scores,
        "primary_gain_nats": primary_gain,
        "gain_over_shifted_null_nats": null_gain,
        "uniform_anchor_absolute_error": anchor_error,
        "predictions": predictions,
        "runtime_s": time.time() - started,
        "scope": "Discovery-only; existing roles are spent and no E1-E4 cell is promoted.",
    }
    atomic_json(payload, RESULT)
    print(f"\n{json.dumps(predictions, sort_keys=True)}", flush=True)
    print(f"wrote {RESULT} in {payload['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

