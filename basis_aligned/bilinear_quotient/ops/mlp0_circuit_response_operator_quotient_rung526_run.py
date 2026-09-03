#!/usr/bin/env python3
"""RUNG 526 -- group exact MLP0 interaction operators by circuit response."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import argparse
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
for path in (OPS, POLY, BQ, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade  # noqa: E402
import mlp0_branch_circuit_response_rung481 as r481  # noqa: E402
import mlp0_centered_context_anova_factorial as r400  # noqa: E402
import mlp0_circuit_response_operator_quotient_rung526_math as qm  # noqa: E402
import mlp0_token_context_operator_quotient_rung525_math as r525m  # noqa: E402


RUNNER = Path(__file__).resolve()
PREREG = POLY / "MLP0_CIRCUIT_RESPONSE_OPERATOR_QUOTIENT_RUNG526_PREREGISTRATION.md"
MATH = OPS / "mlp0_circuit_response_operator_quotient_rung526_math.py"
R525_MATH = OPS / "mlp0_token_context_operator_quotient_rung525_math.py"
R525_RUNNER = OPS / "mlp0_token_context_operator_quotient_rung525_run.py"
R525_RESULT = BQ / "mlp0_token_context_operator_quotient_rung525_results.json"
R525_PAIRS = BQ / "mlp0_token_context_operator_quotient_rung525_pairs.pt"
R481_SOURCE = OPS / "mlp0_branch_circuit_response_rung481.py"
OUT = BQ / "mlp0_circuit_response_operator_quotient_rung526_results.json"
PAIR_OUT = BQ / "mlp0_circuit_response_operator_quotient_rung526_pairs.pt"
SMOKE_OUT = BQ / "mlp0_circuit_response_operator_quotient_rung526_gpu_smoke_results.json"

FROZEN_SHA256 = {
    PREREG: "53be05fdd22ff9153066bb680a9d67ab170319ecb2a335c1760f224449b3fc22",
    MATH: "126917d791282df56dc2a27a62750759c6f58bec57f8e4518da49a7409eaf6af",
    R525_MATH: "b65e875855d4dd0a65afb140c73e60568af590dfbde3b6d411f30fb921353729",
    R525_RUNNER: "8fa1d3c2022f5a3ee8aca4b4f79c64a21fa7f0940bc3207fa7295f828e838b8a",
    R525_RESULT: "34714559df04b966c503321b78fbbabd2f6150dac5e1354ed9070b1dc9e86a0b",
    R525_PAIRS: "11e295fb744bde435158578fffad6a7db994bdb3df28201c4df39354f31d8d4b",
    R481_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
}

D = 1152
H = 4608
TOKENS = 256
REAL_VOCAB = 50_257
BATCH = 4
GRADIENT_CHUNK = 4
DISCOVERY_TAGS = 32
VALIDATION_TAGS = 30
D0 = (0, 124)
D1 = (124, 248)
V0 = (500, 750)
V1 = (750, 1000)
RANDOM_SEED = 526_002
TOY_SEED = 526_100
CHECK_TOKENS = 8


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


def _validate_dependencies() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        value = _file_sha256(path)
        if value != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {value} != {expected}")
        observed[str(path.relative_to(REPO))] = value
    result = json.loads(R525_RESULT.read_text())
    if not (
        result.get("rung") == 525
        and result.get("pred_a_exact_lawful_instrument") is True
        and result.get("pred_b_operator_grouping_transfers") is False
        and result.get("strong_null") is True
        and result.get("physical_downstream_successor_licensed") is False
    ):
        raise RuntimeError("rung 525 did not license the downstream-conditioned metric")
    return observed


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


def _mask_hash(circuit_masks: Mapping[str, Mapping[str, torch.Tensor]], tags: list[str]) -> str:
    digest = hashlib.sha256()
    for tag in tags:
        digest.update(tag.encode())
        for kind in ("member", "slice_control"):
            value = circuit_masks[tag][kind].to(torch.uint8).contiguous()
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _supports(circuit_masks, tags, bounds) -> dict[str, dict[str, int]]:
    lo, hi = bounds
    return {
        tag: {
            kind: int(circuit_masks[tag][kind].view(1000, TOKENS)[lo:hi].sum())
            for kind in ("member", "slice_control")
        }
        for tag in tags
    }


def _population():
    rows, circuit_masks, discovery_tags, validation_tags, fit_rows, metadata = r481.validate_inputs()
    if tuple(rows.shape) != (1000, 257) or len(discovery_tags) != DISCOVERY_TAGS \
            or len(validation_tags) != VALIDATION_TAGS:
        raise RuntimeError("62-circuit population changed")
    bounds = (D0, D1, V0, V1)
    if any(left[1] > right[0] for left, right in zip(bounds, bounds[1:])):
        raise RuntimeError("document roles overlap")
    supports = {
        name: _supports(circuit_masks, tags, role)
        for name, tags, role in (
            ("D0", discovery_tags, D0), ("D1", discovery_tags, D1),
            ("V0", validation_tags, V0), ("V1", validation_tags, V1),
        )
    }
    if min(value for role in supports.values() for tag in role.values() for value in tag.values()) <= 0:
        raise RuntimeError("one or more circuit halves have zero support")
    return rows, circuit_masks, list(discovery_tags), list(validation_tags), fit_rows, {
        **metadata,
        "document_roles": {"D0": list(D0), "D1": list(D1), "V0": list(V0), "V1": list(V1)},
        "supports": supports,
        "mask_hashes": {
            "discovery": _mask_hash(circuit_masks, list(discovery_tags)),
            "validation": _mask_hash(circuit_masks, list(validation_tags)),
        },
    }


def _planted_toy() -> dict[str, object]:
    generator = torch.Generator().manual_seed(TOY_SEED)
    classes, per_class, dims = 32, 8, 32
    tokens = classes * per_class
    class_id = torch.arange(tokens) // per_class
    nuisance = 6 * torch.randn(tokens, 96 - classes, generator=generator, dtype=torch.float64)
    raw = torch.cat((F.one_hot(class_id, classes).to(torch.float64), nuisance), dim=1)
    signatures = {}
    for name, width in (("D0", dims), ("D1", dims), ("V0", 30), ("V1", 30)):
        centers = torch.randn(classes, width, generator=generator, dtype=torch.float64)
        signatures[name] = centers[class_id] + 0.005 * torch.randn(
            tokens, width, generator=generator, dtype=torch.float64
        )
    ids = torch.arange(tokens, dtype=torch.int64)
    donors, receivers = ids[ids % 5 != 0], ids[ids % 5 == 0]
    standardized = {
        name: r525m.standardize_from_donors(value, donors).values
        for name, value in signatures.items()
    }
    candidate, _distance, _cosine = r525m.nearest_far_donors(
        standardized["D0"], raw, receivers, donors, chunk_size=64
    )
    raw_control, _ = r525m.nearest_raw_donors(raw, receivers, donors, chunk_size=64)
    scrambled, _d, _c = r525m.nearest_far_donors(
        qm.derange_coordinates(standardized["D0"]), raw, receivers, donors, chunk_size=64
    )
    ratios = {}
    for name in ("D1", "V0", "V1"):
        candidate_distance = r525m.pair_distances(standardized[name], receivers, candidate)
        raw_distance = r525m.pair_distances(standardized[name], receivers, raw_control)
        ratios[name] = float(candidate_distance.median() / raw_distance.median().clamp_min(1e-30))
    correct = float((class_id[candidate] == class_id[receivers]).float().mean())
    scrambled_correct = float((class_id[scrambled] == class_id[receivers]).float().mean())
    passes = bool(
        correct >= 0.95 and scrambled_correct <= 0.25
        and all(value <= 0.20 for value in ratios.values())
    )
    return {
        "tokens": tokens, "classes": classes, "correct_class_fraction": correct,
        "scrambled_correct_class_fraction": scrambled_correct,
        "unseen_candidate_over_raw": ratios, "passes": passes,
    }


def _gradient_toy() -> dict[str, object]:
    generator = torch.Generator().manual_seed(TOY_SEED + 1)
    tokens, positions, dim, hidden, circuits = 7, 9, 6, 8, 5
    left = torch.randn(hidden, dim, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, dim, generator=generator, dtype=torch.float64)
    down = torch.randn(dim, hidden, generator=generator, dtype=torch.float64)
    token = torch.randn(tokens, dim, generator=generator, dtype=torch.float64)
    context = torch.randn(positions, dim, generator=generator, dtype=torch.float64)
    native = torch.randn(positions, dim, generator=generator, dtype=torch.float64, requires_grad=True)
    suffix = torch.tanh(native @ torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    weights = torch.randn(circuits, positions, dim, generator=generator, dtype=torch.float64)
    scores = torch.einsum("id,cid->c", suffix, weights)
    gradients = torch.stack([
        torch.autograd.grad(scores[c], native, retain_graph=c + 1 < circuits)[0]
        for c in range(circuits)
    ])
    token_left, token_right = token @ left.mT, token @ right.mT
    context_left, context_right = context @ left.mT, context @ right.mT
    downstream_hidden = gradients @ down
    fast = qm.circuit_signature(
        token_left, token_right, context_left, context_right, downstream_hidden, gain=0.41
    )
    explicit = []
    for t in range(tokens):
        interaction = 0.41 * (
            token_left[t][None] * context_right + context_left * token_right[t][None]
        ) @ down.mT
        explicit.append(torch.einsum("cid,id->c", gradients, interaction))
    explicit = torch.stack(explicit)
    error = float((fast - explicit).square().sum() / explicit.square().sum().clamp_min(1e-30))
    return {"relative_squared_error": error, "passes": bool(error <= 1e-10)}


def _token_factors(model, reference):
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    ids = torch.arange(REAL_VOCAB, device=device)
    raw = F.rms_norm(model.transformer.wte(ids), (D,))
    token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw
    token_delta = token_base.float() - reference["token_mean"]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    return token_base.float(), F.linear(token_delta, left), F.linear(token_delta, right)


def _native_forward(model, tokens):
    def attention(event):
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        return event.block.mlp(event.state)

    return facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)


def _leaf_forward(model, tokens):
    capture = {}

    def attention(event):
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 0:
            capture["attention0"] = write.detach()
        return write, first_value

    def mlp(event):
        write = event.block.mlp(event.state)
        if event.site == 0:
            leaf = write.detach().requires_grad_(True)
            capture["leaf"] = leaf
            return leaf
        return write

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    return logits, capture


def _phase_counts(circuit_masks, tags, bounds):
    lo, hi = bounds
    result = torch.zeros(len(tags), 2, dtype=torch.float64)
    for c, tag in enumerate(tags):
        for k, kind in enumerate(("member", "slice_control")):
            result[c, k] = circuit_masks[tag][kind].view(1000, TOKENS)[lo:hi].sum()
    if bool((result <= 0).any()):
        raise RuntimeError("circuit support is empty")
    return result


def _collect_phase(model, rows, circuit_masks, tags, bounds, reference, token_left, token_right):
    lo, hi = bounds
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    counts = _phase_counts(circuit_masks, tags, bounds)
    left_accumulator = torch.zeros(len(tags), H, device=device)
    right_accumulator = torch.zeros_like(left_accumulator)
    explicit = torch.zeros(CHECK_TOKENS, len(tags), device=device, dtype=torch.float64)
    check_ids = (torch.arange(CHECK_TOKENS, device=device) * 787) % REAL_VOCAB
    gradient_energy = torch.zeros(len(tags), dtype=torch.float64)
    weight_sums = torch.zeros(len(tags), 2, dtype=torch.float64)
    replay_error = 0.0
    calls = {"forwards": 0, "native_replays": 0, "batched_backwards": 0, "gradient_objectives": 0}
    for start in range(lo, hi, BATCH):
        stop = min(start + BATCH, hi)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        if start == lo:
            with torch.no_grad():
                native_logits = _native_forward(model, tokens)
            calls["native_replays"] += 1
        with torch.enable_grad():
            logits, capture = _leaf_forward(model, tokens)
            calls["forwards"] += 1
            if start == lo:
                replay_error = float((logits.float() - native_logits.float()).abs().max())
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).view(len(batch_rows), TOKENS)
            weights = torch.zeros(len(tags), len(batch_rows), TOKENS, device=device)
            for c, tag in enumerate(tags):
                member = circuit_masks[tag]["member"].view(1000, TOKENS)[start:stop].to(device)
                control = circuit_masks[tag]["slice_control"].view(1000, TOKENS)[start:stop].to(device)
                weights[c] = member.float() / float(counts[c, 0]) - control.float() / float(counts[c, 1])
                weight_sums[c, 0] += float(member.sum()) / float(counts[c, 0])
                weight_sums[c, 1] -= float(control.sum()) / float(counts[c, 1])

            context_delta = capture["attention0"].float() - reference["context_mean"]
            context_left = F.linear(context_delta, left).reshape(-1, H)
            context_right = F.linear(context_delta, right).reshape(-1, H)
            leaf = capture["leaf"]
            for first in range(0, len(tags), GRADIENT_CHUNK):
                last = min(first + GRADIENT_CHUNK, len(tags))
                retain = last < len(tags)
                gradient = torch.autograd.grad(
                    nll, leaf, grad_outputs=weights[first:last], retain_graph=retain,
                    is_grads_batched=True,
                )[0].reshape(last - first, -1, D).float()
                calls["batched_backwards"] += 1
                calls["gradient_objectives"] += last - first
                gradient_energy[first:last] += gradient.double().square().sum((1, 2)).cpu()
                downstream_hidden = torch.einsum("cnd,dh->cnh", gradient, down)
                right_accumulator[first:last] += torch.einsum(
                    "cnh,nh->ch", downstream_hidden, context_right
                )
                left_accumulator[first:last] += torch.einsum(
                    "cnh,nh->ch", downstream_hidden, context_left
                )
                for local, token_id in enumerate(check_ids.tolist()):
                    interaction = (
                        token_left[token_id][None] * context_right
                        + context_left * token_right[token_id][None]
                    )
                    explicit[local, first:last] += (
                        float(reference["gain_mean"])
                        * torch.einsum("cnh,nh->c", downstream_hidden, interaction).double()
                    )
                del gradient, downstream_hidden
        del logits, capture, nll, weights

    signature = float(reference["gain_mean"]) * (
        token_left @ right_accumulator.mT + token_right @ left_accumulator.mT
    )
    fast_check = signature[check_ids].double()
    contraction_error = float(
        (fast_check - explicit).square().sum() / explicit.square().sum().clamp_min(1e-30)
    )
    nonconstant = bool((signature.std(0, unbiased=False) > 0).all())
    instrument = {
        "bounds": list(bounds), "tags": list(tags), "counts": counts.tolist(),
        "identity_leaf_logit_max_abs": replay_error,
        "all_circuit_gradients_nonzero": bool((gradient_energy > 0).all()),
        "minimum_circuit_gradient_energy": float(gradient_energy.min()),
        "member_weight_sum_max_abs_error": float((weight_sums[:, 0] - 1).abs().max()),
        "control_weight_sum_max_abs_error": float((weight_sums[:, 1] + 1).abs().max()),
        "aggregate_contraction_relative_squared_error": contraction_error,
        "signature_finite": bool(torch.isfinite(signature).all()),
        "signature_nonconstant": nonconstant,
        "signature_sha256": _tensor_sha256(signature),
        "calls": calls,
    }
    return signature, instrument


def _fixed_far_random(raw, receivers, donors, *, seed):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randint(len(donors), (len(receivers), qm.RANDOM_CONTROLS), generator=generator)
    selected = donors.cpu()[positions].to(raw.device)
    receiver_raw = F.normalize(raw[receivers].float(), dim=1, eps=1e-12)
    donor_raw = F.normalize(raw.float(), dim=1, eps=1e-12)
    for _ in range(128):
        cosine = (receiver_raw[:, None] * donor_raw[selected]).sum(-1)
        invalid = cosine > qm.RAW_COSINE_CEILING
        if not bool(invalid.any()):
            return selected, cosine
        selected[invalid] = donors.cpu()[torch.randint(
            len(donors), (int(invalid.sum()),), generator=generator
        )].to(raw.device)
    raise RuntimeError("far-random rejection sampler did not converge")


def _pair_matrix(values, receivers, donors):
    return torch.stack([
        r525m.pair_distances(values, receivers, donors[:, column])
        for column in range(donors.shape[1])
    ], dim=1)


def _discovery_search(signature_d0, signature_d1, raw, rung525_donors):
    device = raw.device
    ids = torch.arange(REAL_VOCAB, device=device, dtype=torch.int64)
    donors, receivers = ids[ids % 5 != 0], ids[ids % 5 == 0]
    d0 = r525m.standardize_from_donors(signature_d0, donors).values
    d1 = r525m.standardize_from_donors(signature_d1, donors).values
    candidate, distance_d0, candidate_cosine = r525m.nearest_far_donors(d0, raw, receivers, donors)
    raw_control, raw_cosine = r525m.nearest_raw_donors(raw, receivers, donors)
    scrambled_candidate, _distance, scrambled_cosine = r525m.nearest_far_donors(
        qm.derange_coordinates(d0), raw, receivers, donors
    )
    first_candidate, _distance, _cosine = r525m.nearest_far_donors(
        d0[:, :16], raw, receivers, donors
    )
    second_candidate, _distance, _cosine = r525m.nearest_far_donors(
        d0[:, 16:], raw, receivers, donors
    )
    random_control, random_cosine = _fixed_far_random(
        raw, receivers, donors, seed=RANDOM_SEED
    )
    vectors = {
        "candidate": r525m.pair_distances(d1, receivers, candidate),
        "raw": r525m.pair_distances(d1, receivers, raw_control),
        "scrambled": r525m.pair_distances(d1, receivers, scrambled_candidate),
        "random": _pair_matrix(d1, receivers, random_control),
        "first_half": r525m.pair_distances(d1, receivers, first_candidate),
        "second_half": r525m.pair_distances(d1, receivers, second_candidate),
    }
    score = qm.score_discovery(
        distance_d0=distance_d0, distance_d1=vectors["candidate"],
        raw_d1=vectors["raw"], random_d1=vectors["random"],
        scrambled_d1=vectors["scrambled"], candidate_donors=candidate,
        circuit_half_d1=(vectors["first_half"], vectors["second_half"]),
        rung525_donors=rung525_donors.to(device),
    )
    score.update({
        "maximum_candidate_raw_cosine": float(candidate_cosine.max()),
        "maximum_scrambled_raw_cosine": float(scrambled_cosine.max()),
        "minimum_random_raw_cosine_margin": float(qm.RAW_COSINE_CEILING - random_cosine.max()),
        "median_raw_control_cosine": float(raw_cosine.median()),
    })
    ids_out = {
        "receivers": receivers, "candidate": candidate, "raw": raw_control,
        "scrambled": scrambled_candidate, "random": random_control,
        "first_half": first_candidate, "second_half": second_candidate,
    }
    return score, ids_out, vectors, distance_d0


def _validation_score(signature, raw, ids_out):
    donors = torch.arange(REAL_VOCAB, device=raw.device, dtype=torch.int64)
    donors = donors[donors % 5 != 0]
    values = r525m.standardize_from_donors(signature, donors).values
    receivers = ids_out["receivers"]
    vectors = {
        "candidate": r525m.pair_distances(values, receivers, ids_out["candidate"]),
        "raw": r525m.pair_distances(values, receivers, ids_out["raw"]),
        "scrambled": r525m.pair_distances(values, receivers, ids_out["scrambled"]),
        "random": _pair_matrix(values, receivers, ids_out["random"]),
    }
    return qm.score_validation_half(
        candidate=vectors["candidate"], raw=vectors["raw"],
        random=vectors["random"], scrambled=vectors["scrambled"],
    ), vectors


def _phase_instrument_passes(instrument):
    return bool(
        instrument["identity_leaf_logit_max_abs"] == 0.0
        and instrument["all_circuit_gradients_nonzero"]
        and instrument["member_weight_sum_max_abs_error"] <= 1e-5
        and instrument["control_weight_sum_max_abs_error"] <= 1e-5
        and instrument["aggregate_contraction_relative_squared_error"] <= 1e-5
        and instrument["signature_finite"] and instrument["signature_nonconstant"]
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    dependencies = _validate_dependencies()
    planted = _planted_toy()
    gradient_toy = _gradient_toy()
    rows, circuit_masks, discovery_tags, validation_tags, fit_rows, metadata = _population()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert planted["passes"] and gradient_toy["passes"]
        assert D0 == (0, 124) and D1 == (124, 248) and V0 == (500, 750) and V1 == (750, 1000)
        print("DRYRUN OK: planted/gradient toys pass; D0 selects, D1 gates, validation circuits sealed")
        return
    output = args.output or (SMOKE_OUT if args.gpu_smoke else OUT)
    if output.exists() or (not args.gpu_smoke and PAIR_OUT.exists()):
        raise FileExistsError("rung 526 output namespace already exists")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda")
    with torch.no_grad():
        reference = r400._reference_moments(model, fit_rows, device)
        raw, token_left, token_right = _token_factors(model, reference)

    if args.gpu_smoke:
        smoke_tags = discovery_tags[:2]
        signature, instrument = _collect_phase(
            model, rows, circuit_masks, smoke_tags, D0, reference, token_left, token_right
        )
        result = {
            "status": "gpu_smoke_complete", "rung": 526,
            "dependencies": dependencies, "runner_sha256": _file_sha256(RUNNER),
            "checkpoint": checkpoint.__dict__, "planted": planted, "gradient_toy": gradient_toy,
            "instrument": instrument, "signature_shape": list(signature.shape),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "runtime_seconds": time.time() - started,
        }
        _atomic_json(output, result)
        print(json.dumps({"output": str(output), "status": result["status"], "instrument": instrument}, indent=2))
        return

    signature_d0, instrument_d0 = _collect_phase(
        model, rows, circuit_masks, discovery_tags, D0, reference, token_left, token_right
    )
    pred_a = bool(
        planted["passes"] and gradient_toy["passes"]
        and _phase_instrument_passes(instrument_d0)
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
    )
    phases = {"D0": instrument_d0}
    validation_opened = d1_opened = False
    score = validation = None
    pair_artifact = None
    if pred_a:
        d1_opened = True
        signature_d1, instrument_d1 = _collect_phase(
            model, rows, circuit_masks, discovery_tags, D1, reference, token_left, token_right
        )
        phases["D1"] = instrument_d1
        pred_a = bool(pred_a and _phase_instrument_passes(instrument_d1))
        old = torch.load(R525_PAIRS, map_location="cpu", weights_only=True)
        score, ids_out, vectors, distance_d0 = _discovery_search(
            signature_d0, signature_d1, raw, old["candidate_donor_ids"]
        )
        pred_b = bool(pred_a and score["prediction_b_document_transfer"])
        pred_d = bool(pred_a and score["prediction_d_reusable_changed_groups"])
        pair_artifact = {
            "schema": "mlp0-circuit-response-operator-quotient-rung526-pairs-v1",
            **{name + "_ids": value.cpu() for name, value in ids_out.items()},
            "d0_candidate_distance": distance_d0.cpu(),
            **{"d1_" + name + "_distance": value.cpu() for name, value in vectors.items()},
        }
        pred_c = False
        if pred_b:
            validation_opened = True
            validation = {"halves": {}}
            validation_vectors = {}
            for name, bounds in (("V0", V0), ("V1", V1)):
                signature, instrument = _collect_phase(
                    model, rows, circuit_masks, validation_tags, bounds,
                    reference, token_left, token_right,
                )
                phases[name] = instrument
                half_score, half_vectors = _validation_score(signature, raw, ids_out)
                validation["halves"][name] = half_score
                validation_vectors[name] = half_vectors
                pair_artifact.update({
                    name.lower() + "_" + key + "_distance": value.cpu()
                    for key, value in half_vectors.items()
                })
                pred_a = bool(pred_a and _phase_instrument_passes(instrument))
            validation["candidate_distance_spearman"] = r525m.spearman(
                validation_vectors["V0"]["candidate"], validation_vectors["V1"]["candidate"]
            )
            pred_c = bool(
                pred_a and all(value["passes"] for value in validation["halves"].values())
                and validation["candidate_distance_spearman"] >= 0.40
            )
            validation["prediction_c_heldout_circuit_transfer"] = pred_c
    else:
        pred_b = pred_c = pred_d = False

    strong_null = bool(
        not pred_a or (score is not None and score["strong_null"])
    )
    physical_licensed = bool(pred_a and pred_b and pred_c and pred_d and not strong_null)
    artifact_receipt = None
    if pair_artifact is not None:
        pair_artifact["runner_sha256"] = _file_sha256(RUNNER)
        artifact_receipt = {
            "path": str(PAIR_OUT), "sha256": _atomic_torch(PAIR_OUT, pair_artifact)
        }
    calls = {
        key: sum(phase["calls"][key] for phase in phases.values())
        for key in ("forwards", "native_replays", "batched_backwards", "gradient_objectives")
    }
    result = {
        "status": "complete", "rung": 526,
        "claim_level": "downstream_circuit_conditioned_tangent_grouping_screen_not_finite_circuit_evidence",
        "dependency_sha256": dependencies, "runner_sha256": _file_sha256(RUNNER),
        "checkpoint": checkpoint.__dict__, "input_identity": metadata,
        "operator_definition": "S_H[t,c]=sum_i grad(Y_c,m0_i)^T K_t da_i",
        "planted": planted, "gradient_toy": gradient_toy, "phase_instruments": phases,
        "d1_opened": d1_opened, "validation_circuits_opened": validation_opened,
        "discovery_score": score, "validation_score": validation,
        "pred_a_exact_live_leakage_free_instrument": pred_a,
        "pred_b_same_circuit_new_document_transfer": pred_b,
        "pred_c_heldout_circuit_transfer": pred_c,
        "pred_d_reusable_changed_groups": pred_d,
        "strong_null": strong_null,
        "physical_successor_licensed": physical_licensed,
        "pair_artifact": artifact_receipt,
        "execution_price": {
            **calls, "reference_attention0_capture_batches": 24,
            "finite_intervention_forwards": 0,
            "deployed_values_added": 0, "deployed_values_saved": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "runtime_seconds": time.time() - started,
        },
        "next_action": (
            "finite_full_suffix_token_operator_swaps_rung527" if physical_licensed
            else "finite_context_only_or_predictive_state_quotient"
        ),
    }
    _atomic_json(output, result)
    print(json.dumps({
        "output": str(output), "predictions": {"A": pred_a, "B": pred_b, "C": pred_c, "D": pred_d},
        "strong_null": strong_null, "physical_successor_licensed": physical_licensed,
        "validation_circuits_opened": validation_opened,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
