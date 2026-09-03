#!/usr/bin/env python3
"""RUNG 525 -- exact MLP0 token-by-context operator quotient screen.

Construct a randomized, algebraically exact sketch of every finite-vocabulary
operator K_t: attention0 context deviation -> centered MLP0 interaction write.
Bank A selects far-token operator neighbors; disjoint-context bank B scores
their transfer against raw-token, far-random, and deranged controls. This is a
function-space grouping screen, not circuit evidence or a rank sweep.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
import torch.nn.functional as F


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
POLY = REPO / "basis_aligned/polynomial_causal"
RUNNER_PATH = Path(__file__).resolve()
PREREG = POLY / "MLP0_TOKEN_CONTEXT_OPERATOR_QUOTIENT_RUNG525_PREREGISTRATION.md"
MATH_PATH = OPS / "mlp0_token_context_operator_quotient_rung525_math.py"
PARENT_SOURCE = OPS / "mlp0_centered_context_anova_factorial.py"
PARENT_RESULT = BQ / "mlp0_centered_context_anova_exact_residual_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
OUT = BQ / "mlp0_token_context_operator_quotient_rung525_results.json"
PAIR_ARTIFACT = BQ / "mlp0_token_context_operator_quotient_rung525_pairs.pt"

for path in (OPS, POLY, BQ, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mlp0_token_context_operator_quotient_rung525_math as oq  # noqa: E402


D = 1152
H = 4608
REAL_VOCAB = 50_257
PROBE_SEED = 525_000
OUTPUT_SEED = 525_001
RANDOM_CONTROL_SEED = 525_002
TOY_SEED = 525_100
SCORING = slice(64, 256)
FROZEN_SHA256 = {
    PREREG: "fdc1575846a97e43c4834e4caa0d2081fea5e5b2ab5d73f5c36b180a3de5f683",
    MATH_PATH: "b65e875855d4dd0a65afb140c73e60568af590dfbde3b6d411f30fb921353729",
    PARENT_SOURCE: "1495ec13abf80bbd3d0bf33db8c0457e1bc5eab7421bcb1b96a780278d808322",
    PARENT_RESULT: "6650b97c9f5b53714d29f999eff6653bdbc9273c9238e4c10ce607d8d5728277",
    ROWS_RECEIPT: "ce4a6f8eeb20840711bb20677ff8310f1a39db55b50106face1157cd2feeef7f",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _row_hashes(rows: torch.Tensor) -> set[str]:
    return {_tensor_sha256(row) for row in rows}


def _validate_dependencies() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        value = _file_sha256(path)
        if value != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {value} != {expected}")
        observed[str(path.relative_to(REPO))] = value
    return observed


def _derange_coordinates(values: torch.Tensor) -> torch.Tensor:
    """Apply four deterministic token-specific cyclic coordinate permutations."""
    if values.ndim != 2 or values.shape[1] != oq.PROBES:
        raise ValueError("derangement expects 256 sketch coordinates")
    width = oq.PROBES // 4
    token = torch.arange(len(values), device=values.device, dtype=torch.int64)
    coordinate = torch.arange(width, device=values.device, dtype=torch.int64)
    result = torch.empty_like(values)
    multipliers = (17, 29, 43, 61)
    offsets = (3, 11, 23, 37)
    for block, (multiplier, offset) in enumerate(zip(multipliers, offsets, strict=True)):
        start = block * width
        shifts = (token * multiplier + offset) % width
        indices = (coordinate[None] + shifts[:, None]) % width
        result[:, start:start + width] = values[:, start:start + width].gather(1, indices)
    return result


def _fixed_far_random_controls(
    raw: torch.Tensor,
    receivers: torch.Tensor,
    donors: torch.Tensor,
    *,
    seed: int,
    count: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.randint(len(donors), (len(receivers), count), generator=generator)
    result = donors.detach().cpu()[positions].to(raw.device)
    receiver_raw = F.normalize(raw[receivers].float(), dim=1, eps=1e-12)
    all_raw = F.normalize(raw.float(), dim=1, eps=1e-12)
    for attempt in range(128):
        cosines = (receiver_raw[:, None] * all_raw[result]).sum(-1)
        invalid = cosines > oq.RAW_COSINE_CEILING
        if not bool(invalid.any()):
            return result, cosines
        replacement = torch.randint(
            len(donors), (int(invalid.sum()),), generator=generator
        )
        result[invalid] = donors.detach().cpu()[replacement].to(raw.device)
    raise RuntimeError("far-random rejection sampler did not converge")


def _probe_rows(context: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose 256 unique scoring positions and retain their flat identities."""
    if context.ndim != 3 or context.shape[1:] != (256, D):
        raise ValueError("context rows must be [document,256,1152]")
    eligible = context[:, SCORING].reshape(-1, D)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(eligible), generator=generator)[:oq.PROBES]
    return eligible[order].contiguous(), order


def _rademacher_output_probes(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(OUTPUT_SEED)
    signs = 2 * torch.randint(0, 2, (oq.PROBES, D), generator=generator) - 1
    return signs.to(device=device, dtype=torch.float32) / math.sqrt(D)


def _toy() -> dict[str, object]:
    """Known 32-class bilinear operator quotient with far raw representatives."""
    generator = torch.Generator(device="cpu").manual_seed(TOY_SEED)
    classes = 32
    per_class = 8
    tokens = classes * per_class
    toy_d = 96
    hidden = 64
    output = 32
    class_id = torch.arange(tokens) // per_class
    class_part = F.one_hot(class_id, classes).to(torch.float64)
    nuisance = 5 * torch.randn(tokens, toy_d - classes, generator=generator, dtype=torch.float64)
    raw = torch.cat((class_part, nuisance), dim=1)
    left = torch.zeros(hidden, toy_d, dtype=torch.float64)
    right = torch.zeros_like(left)
    left[:, :classes] = torch.randn(hidden, classes, generator=generator, dtype=torch.float64)
    right[:, :classes] = torch.randn(hidden, classes, generator=generator, dtype=torch.float64)
    down = torch.randn(output, hidden, generator=generator, dtype=torch.float64)
    contexts_a = torch.zeros(oq.PROBES, toy_d, dtype=torch.float64)
    contexts_b = torch.zeros_like(contexts_a)
    contexts_a[:, :classes] = torch.randn(
        oq.PROBES, classes, generator=generator, dtype=torch.float64
    )
    contexts_b[:, :classes] = torch.randn(
        oq.PROBES, classes, generator=generator, dtype=torch.float64
    )
    signs = 2 * torch.randint(
        0, 2, (oq.PROBES, output), generator=generator
    ).to(torch.float64) - 1
    q = signs / math.sqrt(output)
    token_left, token_right = raw @ left.mT, raw @ right.mT
    output_hidden = q @ down
    sketch_a = oq.operator_sketch(
        token_left, token_right, contexts_a @ left.mT, contexts_a @ right.mT, output_hidden
    )
    sketch_b = oq.operator_sketch(
        token_left, token_right, contexts_b @ left.mT, contexts_b @ right.mT, output_hidden
    )
    ids = torch.arange(tokens, dtype=torch.int64)
    donors, receivers = ids[ids % 5 != 0], ids[ids % 5 == 0]
    a = oq.standardize_from_donors(sketch_a, donors).values
    b = oq.standardize_from_donors(sketch_b, donors).values
    selected, _distance_a, _cosines = oq.nearest_far_donors(
        a, raw, receivers, donors, chunk_size=64
    )
    raw_selected, _ = oq.nearest_raw_donors(raw, receivers, donors, chunk_size=64)
    deranged, _d, _c = oq.nearest_far_donors(
        _derange_coordinates(a), raw, receivers, donors, chunk_size=64
    )
    correct = float((class_id[selected] == class_id[receivers]).to(torch.float32).mean())
    deranged_correct = float((class_id[deranged] == class_id[receivers]).to(torch.float32).mean())
    candidate_b = oq.pair_distances(b, receivers, selected)
    raw_b = oq.pair_distances(b, receivers, raw_selected)
    candidate_over_raw = float(candidate_b.median() / raw_b.median().clamp_min(1e-30))
    return {
        "tokens": tokens,
        "classes": classes,
        "correct_class_fraction": correct,
        "deranged_correct_class_fraction": deranged_correct,
        "bank_b_candidate_over_raw": candidate_over_raw,
        "passes": bool(correct >= 0.95 and candidate_over_raw <= 0.20 and deranged_correct <= 0.25),
    }


@dataclass(frozen=True)
class SketchBuild:
    raw_token: torch.Tensor
    sketch_a: torch.Tensor
    sketch_b: torch.Tensor
    bank_a_ids: torch.Tensor
    bank_b_ids: torch.Tensor
    explicit_relative_squared_error: float
    gain_mean: float


@torch.no_grad()
def _build_real_sketch(model, fit_rows: torch.Tensor, select_rows: torch.Tensor) -> SketchBuild:
    import mlp0_centered_context_anova_factorial as parent

    device = torch.device("cuda")
    reference = parent._reference_moments(model, fit_rows, device)
    _fit_token, fit_context, _fit_gain = parent._capture_inputs(model, fit_rows, device)
    _select_token, select_context, _select_gain = parent._capture_inputs(model, select_rows, device)
    fit_context = fit_context.reshape(len(fit_rows), 256, D)
    select_context = select_context.reshape(len(select_rows), 256, D)
    bank_a_context, bank_a_ids = _probe_rows(fit_context, PROBE_SEED)
    bank_b_context, bank_b_ids = _probe_rows(select_context, PROBE_SEED + 1)
    context_mean = reference["context_mean"].detach().cpu()
    bank_a_delta = (bank_a_context - context_mean).to(device)
    bank_b_delta = (bank_b_context - context_mean).to(device)

    block0 = model.transformer.h[0]
    token_ids = torch.arange(REAL_VOCAB, device=device)
    raw_token = F.rms_norm(model.transformer.wte(token_ids), (D,))
    token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
    token_delta = token_base.float() - reference["token_mean"]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    token_left = F.linear(token_delta, left)
    token_right = F.linear(token_delta, right)
    context_a_left, context_a_right = F.linear(bank_a_delta, left), F.linear(bank_a_delta, right)
    context_b_left, context_b_right = F.linear(bank_b_delta, left), F.linear(bank_b_delta, right)
    q = _rademacher_output_probes(device)
    output_hidden = q @ down
    gain = float(reference["gain_mean"])
    sketch_a = oq.operator_sketch(
        token_left, token_right, context_a_left, context_a_right, output_hidden, gain=gain
    )
    sketch_b = oq.operator_sketch(
        token_left, token_right, context_b_left, context_b_right, output_hidden, gain=gain
    )

    numerator = denominator = 0.0
    check_tokens = torch.arange(64, device=device, dtype=torch.int64) * 787 % REAL_VOCAB
    check_probes = torch.arange(64, device=device, dtype=torch.int64) * 3 % oq.PROBES
    for token, probe in zip(check_tokens.tolist(), check_probes.tolist(), strict=True):
        hidden = (
            token_left[token] * context_a_right[probe]
            + context_a_left[probe] * token_right[token]
        )
        explicit = gain * F.linear(hidden, down)
        scalar = q[probe] @ explicit
        observed = sketch_a[token, probe]
        numerator += float((scalar.double() - observed.double()).square())
        denominator += float(scalar.double().square())
    error = numerator / max(denominator, 1e-30)
    return SketchBuild(
        raw_token=token_base.float(),
        sketch_a=sketch_a,
        sketch_b=sketch_b,
        bank_a_ids=bank_a_ids,
        bank_b_ids=bank_b_ids,
        explicit_relative_squared_error=error,
        gain_mean=gain,
    )


@torch.no_grad()
def _search_real(build: SketchBuild) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    device = build.sketch_a.device
    ids = torch.arange(REAL_VOCAB, device=device, dtype=torch.int64)
    donors, receivers = ids[ids % 5 != 0], ids[ids % 5 == 0]
    a = oq.standardize_from_donors(build.sketch_a, donors).values
    b = oq.standardize_from_donors(build.sketch_b, donors).values
    candidate, distance_a, raw_cosine = oq.nearest_far_donors(
        a, build.raw_token, receivers, donors
    )
    raw_control, raw_control_cosine = oq.nearest_raw_donors(
        build.raw_token, receivers, donors
    )
    deranged_candidate, _dd, deranged_raw_cosine = oq.nearest_far_donors(
        _derange_coordinates(a), build.raw_token, receivers, donors
    )
    half0_candidate, _h0d, _h0c = oq.nearest_far_donors(
        a[:, :oq.HALF_PROBES], build.raw_token, receivers, donors
    )
    half1_candidate, _h1d, _h1c = oq.nearest_far_donors(
        a[:, oq.HALF_PROBES:], build.raw_token, receivers, donors
    )
    random_controls, random_cosines = _fixed_far_random_controls(
        build.raw_token, receivers, donors, seed=RANDOM_CONTROL_SEED
    )
    candidate_b = oq.pair_distances(b, receivers, candidate)
    raw_b = oq.pair_distances(b, receivers, raw_control)
    deranged_b = oq.pair_distances(b, receivers, deranged_candidate)
    random_b = torch.stack([
        oq.pair_distances(b, receivers, random_controls[:, column])
        for column in range(random_controls.shape[1])
    ], dim=1)
    half_b = (
        oq.pair_distances(b, receivers, half0_candidate),
        oq.pair_distances(b, receivers, half1_candidate),
    )
    score = oq.score_real(
        candidate_a_distance=distance_a,
        candidate_b_distance=candidate_b,
        raw_b_distance=raw_b,
        random_b_distances=random_b,
        deranged_b_distance=deranged_b,
        candidate_donors=candidate,
        half_a_b_distances=half_b,
    )
    score["receiver_count"] = len(receivers)
    score["donor_count"] = len(donors)
    score["maximum_candidate_raw_cosine"] = float(raw_cosine.max())
    score["maximum_deranged_candidate_raw_cosine"] = float(deranged_raw_cosine.max())
    score["minimum_random_raw_cosine_margin"] = float(
        oq.RAW_COSINE_CEILING - random_cosines.max()
    )
    score["median_raw_control_cosine"] = float(raw_control_cosine.median())
    artifact = {
        "receiver_ids": receivers.cpu(),
        "candidate_donor_ids": candidate.cpu(),
        "raw_control_donor_ids": raw_control.cpu(),
        "deranged_control_donor_ids": deranged_candidate.cpu(),
        "far_random_donor_ids": random_controls.cpu(),
        "bank_a_candidate_distance": distance_a.cpu(),
        "bank_b_candidate_distance": candidate_b.cpu(),
        "bank_b_raw_control_distance": raw_b.cpu(),
        "bank_b_deranged_control_distance": deranged_b.cpu(),
        "bank_b_far_random_distances": random_b.cpu(),
        "half0_candidate_donor_ids": half0_candidate.cpu(),
        "half1_candidate_donor_ids": half1_candidate.cpu(),
        "half0_bank_b_distance": half_b[0].cpu(),
        "half1_bank_b_distance": half_b[1].cpu(),
    }
    return score, artifact


def _atomic_torch(path: Path, value: object) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as sink:
            torch.save(value, sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--pair-artifact", type=Path, default=PAIR_ARTIFACT)
    parser.add_argument("--toy-only", action="store_true")
    args = parser.parse_args(argv)
    dependencies = _validate_dependencies()
    toy = _toy()
    if args.toy_only:
        print(json.dumps({"toy": toy, "model_loaded": False}, indent=2, sort_keys=True))
        return {"toy": toy}
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert REAL_VOCAB == 50_257 and D == 1152 and H == 4608
        assert oq.PROBES == 256 and oq.HALF_PROBES == 128
        print("DRYRUN OK: toy before model; exact K_t sketches; A selects/B scores; no downstream calls")
        return {"dry_run": True, "toy_not_scored": True, "dependencies": dependencies}
    if not toy["passes"]:
        raise RuntimeError(f"planted operator quotient failed before model load: {toy}")
    if args.output.exists() or args.pair_artifact.exists():
        raise FileExistsError("rung525 output namespace already exists")

    started = time.time()
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as parent
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    if len(fit_rows) != 96 or len(select_rows) != 96:
        raise RuntimeError("rung401 row census changed")
    if _row_hashes(fit_rows).intersection(_row_hashes(select_rows)):
        raise RuntimeError("FIT and SELECT contain an identical document row")
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True
    )
    build = _build_real_sketch(model, fit_rows, select_rows)
    del model
    score, artifact_values = _search_real(build)
    bank_disjoint = not bool(torch.equal(build.bank_a_ids, build.bank_b_ids))
    # IDs index different document roles; keep both hashes and require the roles themselves differ.
    role_hashes = {
        "FIT_rows": _tensor_sha256(fit_rows),
        "SELECT_rows": _tensor_sha256(select_rows),
        "bank_A_local_positions": _tensor_sha256(build.bank_a_ids),
        "bank_B_local_positions": _tensor_sha256(build.bank_b_ids),
    }
    nonconstant = bool(
        (build.sketch_a.float().std(0, unbiased=False) > 0).all()
        and (build.sketch_b.float().std(0, unbiased=False) > 0).all()
    )
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and toy["passes"]
        and len(build.bank_a_ids.unique()) == oq.PROBES
        and len(build.bank_b_ids.unique()) == oq.PROBES
        and bank_disjoint
        and build.sketch_a.shape == (REAL_VOCAB, oq.PROBES)
        and build.sketch_b.shape == (REAL_VOCAB, oq.PROBES)
        and nonconstant
        and build.explicit_relative_squared_error <= 1e-10
        and score["maximum_candidate_raw_cosine"] <= 0.5001
        and score["minimum_random_raw_cosine_margin"] >= -0.0001
    )
    pred_b = bool(score["prediction_b_operator_transfer"])
    pred_c = bool(score["prediction_c_repeated_groups"])
    strong_null = bool(not pred_a or score["strong_null"])
    physical_licensed = bool(pred_a and pred_b and pred_c and not strong_null)
    occupied = score["groups"]["distinct_donors"]
    assignment_bits = len(artifact_values["receiver_ids"]) * math.ceil(math.log2(max(occupied, 2)))
    representative_bits = occupied * 2 * H * 32
    full_cache_bits = REAL_VOCAB * 2 * H * 32
    artifact_payload = {
        "schema": "mlp0-token-context-operator-quotient-rung525-pairs-v1",
        **artifact_values,
        "role_hashes": role_hashes,
        "runner_sha256": _file_sha256(RUNNER_PATH),
    }
    artifact_sha = _atomic_torch(args.pair_artifact, artifact_payload)
    result = {
        "status": "complete",
        "rung": 525,
        "claim_level": "exact_weight_function_operator_grouping_screen_not_circuit_evidence",
        "dependency_sha256": dependencies,
        "runner_sha256": _file_sha256(RUNNER_PATH),
        "checkpoint": checkpoint.__dict__,
        "toy": toy,
        "operator_definition": (
            "K_t da = gain_mean*Down[(Left de_t)*(Right da)+(Left da)*(Right de_t)]"
        ),
        "population": {
            "real_tokens": REAL_VOCAB,
            "donors": int((torch.arange(REAL_VOCAB) % 5 != 0).sum()),
            "receivers": int((torch.arange(REAL_VOCAB) % 5 == 0).sum()),
            "split": "token_id_mod5",
        },
        "probe_banks": {
            "probes_per_bank": oq.PROBES,
            "half_probes": oq.HALF_PROBES,
            "FIT_documents": len(fit_rows),
            "SELECT_documents": len(select_rows),
            "role_hashes": role_hashes,
            "roles_differ": bank_disjoint,
            "gain_mean": build.gain_mean,
            "sketch_A_sha256": _tensor_sha256(build.sketch_a),
            "sketch_B_sha256": _tensor_sha256(build.sketch_b),
            "all_operator_sketches_nonconstant": nonconstant,
        },
        "instrument": {
            "explicit_scalar_identity_relative_squared_error": build.explicit_relative_squared_error,
            "maximum_candidate_raw_cosine": score["maximum_candidate_raw_cosine"],
            "minimum_random_raw_cosine_margin": score["minimum_random_raw_cosine_margin"],
            "no_downstream_model_or_circuit_calls": True,
            "FINAL_or_sealed_opened": False,
        },
        "score": score,
        "descriptive_code_length": {
            "occupied_representatives": occupied,
            "receiver_assignment_bits": assignment_bits,
            "representative_factor_bits": representative_bits,
            "total_bits": assignment_bits + representative_bits,
            "full_cached_token_factor_bits": full_cache_bits,
            "ratio_to_full_cached_factor_table": (
                assignment_bits + representative_bits
            ) / full_cache_bits,
            "native_mlp0_comparison_is_not_claimed": True,
        },
        "pred_a_exact_lawful_instrument": pred_a,
        "pred_b_operator_grouping_transfers": pred_b,
        "pred_c_repeated_groups_not_isolated_pairs": pred_c,
        "strong_null": strong_null,
        "physical_downstream_successor_licensed": physical_licensed,
        "next_action": (
            "physical_natural_context_operator_interchange"
            if physical_licensed else
            "context_only_or_downstream_conditioned_operator_metric"
        ),
        "pair_artifact": {
            "path": str(args.pair_artifact.resolve()),
            "sha256": artifact_sha,
        },
        "execution_price": {
            "attention0_capture_batches": 72,
            "downstream_model_forwards": 0,
            "circuit_evaluations": 0,
            "deployed_values_added": 0,
            "deployed_values_saved": 0,
            "runtime_seconds": time.time() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "toy": toy,
        "predictions": {"A": pred_a, "B": pred_b, "C": pred_c},
        "strong_null": strong_null,
        "physical_successor_licensed": physical_licensed,
        "score": score,
        "output": str(args.output),
    }, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
