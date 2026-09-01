"""RUNG 419 -- ATTENTION0 OV DOWNSTREAM-METRIC SHARED CODEBOOK.

The architectural head is provenance, not the feature basis.  Fold every real
token through each exact per-head O_h V_h payload, restrict to the independently
validated task-useful routed-write U16 interface, pull a metric back from finite
MLP0 and attention1-reader responses, and compare one global K256 one-sparse
codebook with the best equal-total-center allocation of nine private codebooks.

Natural transport keeps every native head-specific double-QK score and the
native U16-orthogonal tail.  Therefore the experiment tests whether the payload
vocabulary is shared below the head boundary; it does not substitute an invalid
all-head OV product and it is not a compressed artifact.

Frozen predictions and null are in
ATTENTION0_OV_DOWNSTREAM_CODEBOOK_PREREGISTRATION.md.  No FINAL rows are opened.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
QK = ROOT / "basis_aligned/qk_mdl"
OUT = BQ / "attention0_ov_downstream_codebook_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"

D = 1152
N_HEAD = 9
HD = 128
VOCAB = 50_257
RANK = 16
K_TOTAL = 256
PRIVATE_MAX = K_TOTAL - (N_HEAD - 1)
POSITIONS = tuple(range(16, 241, 16))
DOC_BATCH = 4
BISECT_LLOYD = 12
CONSUMERS = ("mlp0", "q1", "k1", "q2", "k2", "fresh_v")
HAAR_SEEDS = (419_101, 419_102, 419_103)


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _nearest(x: torch.Tensor, centers: torch.Tensor, chunk: int = 32_768):
    assignments, errors = [], []
    centers = centers.float()
    center_square = centers.square().sum(1)
    for start in range(0, len(x), chunk):
        part = x[start:start + chunk].float()
        distance = (part.square().sum(1, keepdim=True) + center_square[None]
                    - 2 * part @ centers.T).clamp_min_(0)
        error, assignment = distance.min(1)
        assignments.append(assignment)
        errors.append(error)
    return torch.cat(assignments), torch.cat(errors)


def _lexicographic_less(left: torch.Tensor, right: torch.Tensor) -> bool:
    a = left.detach().double().cpu().tolist()
    b = right.detach().double().cpu().tolist()
    return tuple(a) < tuple(b)


@torch.no_grad()
def _bisecting_curve(x: torch.Tensor, maximum: int, *, keep_snapshots: bool):
    """Deterministic complete 1..maximum bisecting-kmeans curve."""
    x = x.float().contiguous()
    assignments = torch.zeros(len(x), dtype=torch.long, device=x.device)
    centers = [x.double().mean(0).float()]
    _, initial_error = _nearest(x, torch.stack(centers))
    cluster_sse = [float(initial_error.double().sum())]
    curve = {1: cluster_sse[0]}
    snapshots = {1: torch.stack(centers).cpu()} if keep_snapshots else {}

    for count in range(2, maximum + 1):
        split_id = max(range(len(cluster_sse)), key=lambda idx: (cluster_sse[idx], -idx))
        global_indices = torch.nonzero(assignments == split_id, as_tuple=False).flatten()
        points = x[global_indices]
        if len(points) < 2:
            raise RuntimeError(f"cannot split cluster {split_id} with {len(points)} points")
        mean = points.double().mean(0)
        centered = points.double() - mean
        covariance = centered.T @ centered / max(len(points), 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        scale = eigenvalues[-1].clamp_min(1e-20).sqrt() * .5
        pair = torch.stack([(mean - scale * direction).float(),
                            (mean + scale * direction).float()])
        local = torch.zeros(len(points), dtype=torch.long, device=x.device)
        for _ in range(BISECT_LLOYD):
            local, residual = _nearest(points, pair)
            counts = torch.bincount(local, minlength=2)
            if int(counts.min()) == 0:
                empty = int(torch.argmin(counts))
                occupied = 1 - empty
                occupied_indices = torch.nonzero(local == occupied, as_tuple=False).flatten()
                move_local = occupied_indices[residual[occupied_indices].argmax()]
                local[move_local] = empty
            new_pair = []
            for child in range(2):
                new_pair.append(points[local == child].double().mean(0).float())
            updated = torch.stack(new_pair)
            if float((updated - pair).abs().max()) <= 1e-7:
                pair = updated
                break
            pair = updated
        if not _lexicographic_less(pair[0], pair[1]):
            pair = pair.flip(0)
            local = 1 - local

        assignments[global_indices[local == 0]] = split_id
        new_id = len(centers)
        assignments[global_indices[local == 1]] = new_id
        centers[split_id] = pair[0]
        centers.append(pair[1])
        child_sse = []
        for child in range(2):
            child_points = points[local == child]
            child_sse.append(float(
                (child_points.double() - pair[child].double()).square().sum()))
        cluster_sse[split_id] = child_sse[0]
        cluster_sse.append(child_sse[1])
        curve[count] = sum(cluster_sse)
        if keep_snapshots:
            snapshots[count] = torch.stack(centers).cpu()
    return curve, snapshots, torch.stack(centers), assignments


def _optimal_allocation(curves):
    infinity = float("inf")
    dp = [[infinity] * (K_TOTAL + 1) for _ in range(N_HEAD + 1)]
    back = [[None] * (K_TOTAL + 1) for _ in range(N_HEAD + 1)]
    dp[0][0] = 0.0
    for head in range(N_HEAD):
        remaining_heads = N_HEAD - head - 1
        for used in range(K_TOTAL + 1):
            if not math.isfinite(dp[head][used]):
                continue
            maximum = min(PRIVATE_MAX, K_TOTAL - used - remaining_heads)
            for count in range(1, maximum + 1):
                candidate = dp[head][used] + curves[head][count]
                target = used + count
                if candidate < dp[head + 1][target] - 1e-12:
                    dp[head + 1][target] = candidate
                    back[head + 1][target] = count
    if not math.isfinite(dp[N_HEAD][K_TOTAL]):
        raise RuntimeError("private-center allocation dynamic program has no solution")
    allocation = [0] * N_HEAD
    used = K_TOTAL
    for head in range(N_HEAD, 0, -1):
        count = back[head][used]
        allocation[head - 1] = count
        used -= count
    assert sum(allocation) == K_TOTAL and min(allocation) >= 1
    return allocation, dp[N_HEAD][K_TOTAL]


def _metric_factor(gram: torch.Tensor):
    eigenvalues, eigenvectors = torch.linalg.eigh((gram.double() + gram.double().T) / 2)
    eigenvalues = eigenvalues.clamp_min(0)
    return (eigenvalues.sqrt()[:, None] * eigenvectors.T).float(), eigenvalues


def _asvd(weight: torch.Tensor, inputs: torch.Tensor):
    """Historical RSPD A-SVD: SVD(W X^T), followed by pinv(X^T)."""
    target_transpose = weight.float() @ inputs.float().T
    output_vectors, singular_values, sample_vectors = torch.linalg.svd(
        target_transpose, full_matrices=False)
    left = output_vectors * singular_values[None, :]
    right = sample_vectors @ torch.linalg.pinv(inputs.float().T)
    return left, right


def _rank90(eigenvalues):
    ordered = eigenvalues.double().flip(0)
    return int(torch.searchsorted(
        (ordered / ordered.sum().clamp_min(1e-30)).cumsum(0), .90).item() + 1)


def _leading_subspace_overlap(left: torch.Tensor, right: torch.Tensor):
    left_values, left_vectors = torch.linalg.eigh(
        (left.double() + left.double().T) / 2)
    right_values, right_vectors = torch.linalg.eigh(
        (right.double() + right.double().T) / 2)
    left_rank, right_rank = _rank90(left_values), _rank90(right_values)
    left_basis = left_vectors[:, -left_rank:]
    right_basis = right_vectors[:, -right_rank:]
    denominator = max(min(left_rank, right_rank), 1)
    overlap = float((left_basis.T @ right_basis).square().sum() / denominator)
    return overlap, left_rank, right_rank


def _frobenius_cosine(left, right):
    a = left.double().flatten()
    b = right.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def _qk_fields(block, state):
    batch, length, _ = state.shape
    fields = {}
    for name, layer in (("q1", block.attn.c_q), ("k1", block.attn.c_k),
                        ("q2", block.attn.c_q2), ("k2", block.attn.c_k2)):
        fields[name] = F.rms_norm(layer(state).view(batch, length, N_HEAD, HD), (HD,))
    fields["fresh_v"] = block.attn.c_v(state).view(batch, length, N_HEAD, HD)
    return fields


def _consumer_fields(block0, block1, x0, token_base, attention0):
    residual0 = token_base + attention0
    mlp0 = block0.mlp(F.rms_norm(residual0, (D,)))
    mixed1 = block1.lambdas[0] * (residual0 + mlp0) + block1.lambdas[1] * x0
    state1 = F.rms_norm(mixed1, (D,))
    fields = {"mlp0": mlp0}
    fields.update(_qk_fields(block1, state1))
    return fields


@torch.no_grad()
def _attention0_pattern(block, state, rope_tables, apply_rot):
    batch, length, _ = state.shape
    cos, sin = rope_tables(length, HD, state.device, torch.float32, "bf16")
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def qk(layer):
        value = F.rms_norm(layer(state).view(batch, length, N_HEAD, HD), (HD,))
        return apply_rot(value, cos, sin)

    q1, k1 = qk(block.attn.c_q), qk(block.attn.c_k)
    q2, k2 = qk(block.attn.c_q2), qk(block.attn.c_k2)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / HD
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HD
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    return (score1 * score2).masked_fill(~mask, 0)


@torch.no_grad()
def _response_metric(model, rows, interface, sigma, normalizers, device):
    block0, block1 = model.transformer.h[:2]
    grams = {name: torch.zeros(RANK, RANK, dtype=torch.float64, device=device)
             for name in CONSUMERS}
    squares = {name: 0.0 for name in CONSUMERS}
    counts = {name: 0 for name in CONSUMERS}
    live = {name: 0.0 for name in CONSUMERS}
    for start in range(0, len(rows), DOC_BATCH):
        tokens = rows[start:start + DOC_BATCH, :-1].to(device)
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        attention0, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
        columns = {name: [] for name in CONSUMERS}
        for mode in range(RANK):
            delta = (sigma[mode] * interface[:, mode]).to(attention0.dtype)
            plus = _consumer_fields(block0, block1, x0, token_base, attention0 + delta)
            minus = _consumer_fields(block0, block1, x0, token_base, attention0 - delta)
            for name in CONSUMERS:
                response = ((plus[name].float() - minus[name].float())
                            / (2 * float(sigma[mode]))).flatten(2)
                response = response[:, POSITIONS, :]
                columns[name].append(response)
                live[name] = max(live[name], float((plus[name] - minus[name]).abs().max()))
        for name in CONSUMERS:
            # [position-row,mode,consumer-coordinate]
            value = torch.stack(columns[name], 1).permute(0, 2, 1, 3).flatten(0, 1)
            gram = torch.einsum("nmd,nkd->mk", value.double(), value.double())
            grams[name] += gram
            squares[name] += float(value.double().square().sum())
            counts[name] += value.numel()
    if normalizers is None:
        normalizers = {
            name: math.sqrt(squares[name] / max(counts[name], 1)) for name in CONSUMERS}
    total = torch.zeros(RANK, RANK, dtype=torch.float64, device=device)
    for name in CONSUMERS:
        scale = max(normalizers[name], 1e-30)
        # Equal total weight for every consumer after RMS normalization.
        total += grams[name] / (scale * scale * max(counts[name] // RANK, 1))
    total /= len(CONSUMERS)
    return total, normalizers, live


@torch.no_grad()
def _capture_cproj_input(model, rows, device):
    captured = []
    module = model.transformer.h[0].attn.c_proj
    handle = module.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().float()[:, POSITIONS].cpu()))
    try:
        for start in range(0, len(rows), DOC_BATCH):
            tokens = rows[start:start + DOC_BATCH, :-1].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            x, first_value = x0, None
            for block in model.transformer.h:
                x, first_value = block(x, first_value, x0)
    finally:
        handle.remove()
    return torch.cat(captured).reshape(-1, D)


@torch.no_grad()
def _forward(model, tokens, *, cproj_weight=None, quantizer=None,
             rope_tables=None, apply_rot=None, return_front=False):
    x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
    x, first_value = x0, None
    front = None
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        state = F.rms_norm(x, (D,))
        if site == 0 and (cproj_weight is not None or quantizer is not None):
            if cproj_weight is not None:
                captured = {}
                handle = block.attn.c_proj.register_forward_pre_hook(
                    lambda _module, args: captured.setdefault("joined", args[0]))
                try:
                    native, first_value = block.attn(state, first_value)
                finally:
                    handle.remove()
                attention = F.linear(captured["joined"].float(), cproj_weight.float()).to(native.dtype)
            else:
                native, first_value = block.attn(state, first_value)
                pattern = _attention0_pattern(block, state, rope_tables, apply_rot)
                delta_codes = quantizer["delta_codes"][tokens].to(pattern.dtype)
                routed = torch.einsum("bhqk,bkhc->bqc", pattern, delta_codes)
                correction = routed.float() @ quantizer["interface"].float().T
                attention = native + correction.to(native.dtype)
        else:
            attention, first_value = block.attn(state, first_value)
        x = x + attention
        mlp = block.mlp(F.rms_norm(x, (D,)))
        x = x + mlp
        if site == 0 and return_front:
            front = {"x0": x0, "token_base": block.lambdas[0] * x0 + block.lambdas[1] * x0,
                     "attention0": attention, "mlp0": mlp, "first_value": first_value}
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30)
    return (logits, front) if return_front else logits


@torch.no_grad()
def _ce(model, rows, device, scoring, **forward_kwargs):
    values = []
    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to(device)
        logits = _forward(model, batch[:, :-1], **forward_kwargs)
        for row in range(len(batch)):
            values.append(scoring.document_mean_ce(logits[row], batch[row, 1:]))
    return torch.stack(values).double().cpu()


def _payload_codes(model, interface, embedding):
    block = model.transformer.h[0]
    assert block.attn.c_v.bias is None and block.attn.c_proj.bias is None
    values = []
    for head in range(N_HEAD):
        band = slice(head * HD, (head + 1) * HD)
        value_weight = block.attn.c_v.weight[band].detach().float()
        output_weight = block.attn.c_proj.weight[:, band].detach().float()
        transform = value_weight.T @ output_weight.T @ interface.float()
        values.append(embedding.float() @ transform)
    return torch.stack(values, 1)


def _payload_exactness(model, embedding):
    block = model.transformer.h[0]
    generator = torch.Generator(device="cpu").manual_seed(419_001)
    token_ids = torch.randperm(VOCAB, generator=generator)[:256].to(embedding.device)
    maximum = 0.0
    for head in range(N_HEAD):
        band = slice(head * HD, (head + 1) * HD)
        x = embedding[token_ids].double()
        v = block.attn.c_v.weight[band].detach().double()
        o = block.attn.c_proj.weight[:, band].detach().double()
        direct = (x @ v.T) @ o.T
        folded = x @ (o @ v).T
        maximum = max(maximum, float((direct - folded).abs().max()))
    return maximum


@torch.no_grad()
def _native_forward_replay(model, rows, facade, device):
    tokens = rows[:DOC_BATCH, :-1].to(device)

    def native_attention(event):
        return event.block.attn(event.state, event.first_value)

    def native_mlp(event):
        return event.block.mlp(event.state)

    reference = facade.forward_with_dispatch(
        model, tokens, native_attention, native_mlp)
    local = _forward(model, tokens)
    return float((reference - local).abs().max())


def _fit_codebook(train_a, select_a, factor, fit_mask, select_mask, *, permute=False):
    train = (train_a[fit_mask] @ factor.T).permute(1, 0, 2).contiguous()
    select = (select_a[select_mask] @ factor.T).permute(1, 0, 2).contiguous()
    # [head,token,dim]
    if permute:
        shuffled = []
        for head in range(N_HEAD):
            generator = torch.Generator(device="cpu").manual_seed(419_500 + head)
            permutation = torch.randperm(train.shape[1], generator=generator).to(train.device)
            shuffled.append(train[head, permutation])
        train_for_global = torch.stack(shuffled)
    else:
        train_for_global = train
    global_curve, _, global_centers, _ = _bisecting_curve(
        train_for_global.flatten(0, 1), K_TOTAL, keep_snapshots=False)
    private_curves, private_snapshots = [], []
    for head in range(N_HEAD):
        curve, snapshots, _, _ = _bisecting_curve(
            train[head], PRIVATE_MAX, keep_snapshots=True)
        private_curves.append(curve)
        private_snapshots.append(snapshots)
    allocation, fit_private_sse = _optimal_allocation(private_curves)

    global_assign, global_error = _nearest(select.flatten(0, 1), global_centers)
    global_assign = global_assign.view(N_HEAD, -1)
    global_error = global_error.view(N_HEAD, -1)
    private_assignments, private_errors, private_centers = [], [], []
    for head, count in enumerate(allocation):
        centers = private_snapshots[head][count].to(select.device)
        assignment, error = _nearest(select[head], centers)
        private_assignments.append(assignment)
        private_errors.append(error)
        private_centers.append(centers)

    energy = float(select.double().square().sum())
    global_distortion = float(global_error.double().sum()) / max(energy, 1e-30)
    private_distortion = float(torch.cat(private_errors).double().sum()) / max(energy, 1e-30)
    advantage = (private_distortion - global_distortion) / max(private_distortion, 1e-30)

    # Means in original a coordinates are the metric-optimal representatives.
    select_a_head = select_a[select_mask].permute(1, 0, 2).contiguous()
    train_a_head = train_a[fit_mask].permute(1, 0, 2).contiguous()
    global_train_assign, _ = _nearest(train_for_global.flatten(0, 1), global_centers)
    global_center_a = torch.zeros(K_TOTAL, RANK, device=train_a.device, dtype=torch.float64)
    global_count = torch.zeros(K_TOTAL, device=train_a.device, dtype=torch.float64)
    original_global_a = train_a_head.flatten(0, 1).double()
    global_center_a.index_add_(0, global_train_assign, original_global_a)
    global_count.index_add_(0, global_train_assign, torch.ones_like(global_train_assign, dtype=torch.float64))
    global_center_a /= global_count.clamp_min(1)[:, None]
    private_center_a = []
    for head, count in enumerate(allocation):
        centers = private_snapshots[head][count].to(train.device)
        assignment, _ = _nearest(train[head], centers)
        sums = torch.zeros(count, RANK, dtype=torch.float64, device=train.device)
        counts = torch.zeros(count, dtype=torch.float64, device=train.device)
        sums.index_add_(0, assignment, train_a_head[head].double())
        counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float64))
        private_center_a.append((sums / counts.clamp_min(1)[:, None]).float())

    support = torch.zeros(K_TOTAL, N_HEAD, dtype=torch.long, device=select.device)
    for head in range(N_HEAD):
        support[:, head] = torch.bincount(global_assign[head], minlength=K_TOTAL)
    occupied = support.sum(1) > 0
    diverse = (support >= 100).sum(1) >= 3
    diverse_fraction = float((diverse & occupied).sum()) / max(int(occupied.sum()), 1)

    halves = {}
    select_ids = torch.arange(VOCAB, device=select.device)[select_mask]
    for parity in (0, 1):
        mask = (select_ids.div(5, rounding_mode="floor").remainder(2) == parity)
        ge = float(global_error[:, mask].double().sum())
        pe = float(torch.stack(private_errors)[:, mask].double().sum())
        halves[str(parity)] = {
            "global_distortion": ge,
            "private_distortion": pe,
            "global_advantage": (pe - ge) / max(pe, 1e-30),
        }

    return {
        "global_centers_metric": global_centers,
        "global_centers_a": global_center_a.float(),
        "private_centers_a": private_center_a,
        "allocation": allocation,
        "global_select_assignments": global_assign,
        "private_select_assignments": torch.stack(private_assignments),
        "global_distortion": global_distortion,
        "private_distortion": private_distortion,
        "global_advantage": advantage,
        "diverse_center_fraction": diverse_fraction,
        "occupied_centers": int(occupied.sum()),
        "heldout_halves": halves,
        "fit_global_sse": global_curve[K_TOTAL],
        "fit_private_sse": fit_private_sse,
    }


def _replacement_table(all_a, factor, codebook, private):
    transformed = all_a @ factor.T
    result = torch.empty_like(all_a)
    if not private:
        assignments, _ = _nearest(transformed.flatten(0, 1),
                                   codebook["global_centers_metric"])
        result = codebook["global_centers_a"][assignments].view_as(all_a)
    else:
        for head in range(N_HEAD):
            centers_metric = codebook["private_centers_a"][head] @ factor.T
            assignments, _ = _nearest(transformed[:, head], centers_metric)
            result[:, head] = codebook["private_centers_a"][head][assignments]
    return result


@torch.no_grad()
def _transport(model, rows, device, interface, all_a, codebook, factor,
               rope_tables, apply_rot, scoring):
    replacements = {
        "global": _replacement_table(all_a, factor, codebook, False),
        "private": _replacement_table(all_a, factor, codebook, True),
    }
    quantizers = {
        name: {"delta_codes": (replacement - all_a).to(device),
               "interface": interface.to(device)}
        for name, replacement in replacements.items()
    }
    ce = {"native": [], "global": [], "private": []}
    routed = {name: {"sse": 0.0} for name in ("global", "private")}
    consumer = {arm: {name: {"sse": 0.0} for name in CONSUMERS}
                for arm in ("global", "private")}
    target_stats = {"routed_square": 0.0,
                    "consumer_square": {name: 0.0 for name in CONSUMERS}}
    replay_before_remainder_max = 0.0
    replay_after_remainder_max = 0.0
    replay_num = replay_den = 0.0
    block0, block1 = model.transformer.h[:2]

    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to(device)
        tokens = batch[:, :-1]
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state0 = F.rms_norm(token_base, (D,))
        native_attention, _ = block0.attn(state0, None)
        native_fields = _consumer_fields(block0, block1, x0, token_base, native_attention)
        native_u = native_attention.float() @ interface.float()
        tail_attention = native_attention - (
            native_u @ interface.float().T).to(native_attention.dtype)
        tail_fields = _consumer_fields(block0, block1, x0, token_base, tail_attention)
        pattern = _attention0_pattern(block0, state0, rope_tables, apply_rot)
        edge_u = torch.einsum("bhqk,bkhc->bqc", pattern, all_a[tokens].to(pattern.dtype))
        replay_delta = native_u - edge_u.float()
        replay_before_remainder_max = max(
            replay_before_remainder_max, float(replay_delta.abs().max()))
        replayed_with_measured_remainder = edge_u.float() + replay_delta
        replay_after_remainder_max = max(
            replay_after_remainder_max,
            float((native_u - replayed_with_measured_remainder).abs().max()))
        replay_num += float(replay_delta.double().square().sum())
        replay_den += float(native_u.double().square().sum())

        native_logits = _forward(model, tokens)
        for row in range(len(batch)):
            ce["native"].append(scoring.document_mean_ce(native_logits[row], batch[row, 1:]))

        selected_native_u = native_u[:, POSITIONS]
        target_stats["routed_square"] += float(selected_native_u.double().square().sum())
        for name in CONSUMERS:
            target_response = (native_fields[name].float().flatten(2)[:, POSITIONS]
                               - tail_fields[name].float().flatten(2)[:, POSITIONS])
            target_stats["consumer_square"][name] += float(
                target_response.double().square().sum())

        for arm in ("global", "private"):
            quant = quantizers[arm]
            delta = torch.einsum(
                "bhqk,bkhc->bqc", pattern,
                quant["delta_codes"][tokens].to(pattern.dtype))
            changed_attention = native_attention + (delta.float() @ interface.float().T).to(native_attention.dtype)
            changed_u = changed_attention.float() @ interface.float()
            routed[arm]["sse"] += float(
                (changed_u[:, POSITIONS].double() - selected_native_u.double()).square().sum())
            changed_fields = _consumer_fields(block0, block1, x0, token_base, changed_attention)
            for name in CONSUMERS:
                consumer[arm][name]["sse"] += float(
                    (changed_fields[name].float().flatten(2)[:, POSITIONS].double()
                     - native_fields[name].float().flatten(2)[:, POSITIONS].double()).square().sum())
            logits = _forward(model, tokens, quantizer=quant,
                              rope_tables=rope_tables, apply_rot=apply_rot)
            for row in range(len(batch)):
                ce[arm].append(scoring.document_mean_ce(logits[row], batch[row, 1:]))

    routed_r2 = {arm: 1 - value["sse"] / max(target_stats["routed_square"], 1e-30)
                 for arm, value in routed.items()}
    consumer_r2 = {arm: {} for arm in ("global", "private")}
    for arm in consumer_r2:
        for name in CONSUMERS:
            denominator = target_stats["consumer_square"][name]
            consumer_r2[arm][name] = (
                1 - consumer[arm][name]["sse"] / max(denominator, 1e-30))
    ce_tensor = {name: torch.stack(values).double().cpu() for name, values in ce.items()}
    ce_public = {}
    for name, values in ce_tensor.items():
        ce_public[name] = {
            "mean": float(values.mean()),
            "damage": float(values.mean() - ce_tensor["native"].mean()),
            "wave_damage": [
                float(values[:48].mean() - ce_tensor["native"][:48].mean()),
                float(values[48:].mean() - ce_tensor["native"][48:].mean()),
            ],
        }
    return {"r2_definition": "zero-origin native routed-U16-induced response",
            "routed_u16_r2": routed_r2, "consumer_r2": consumer_r2,
            "mean_consumer_r2": {arm: sum(values.values()) / len(values)
                                 for arm, values in consumer_r2.items()},
            "ce": ce_public,
            "u16_edge_replay_before_remainder_max_abs": replay_before_remainder_max,
            "u16_edge_replay_after_measured_remainder_max_abs": replay_after_remainder_max,
            "u16_edge_relative_squared_error_before_remainder": (
                replay_num / max(replay_den, 1e-30)),
            "u16_edge_relative_squared_error_after_measured_remainder": 0.0}


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert D == N_HEAD * HD and RANK == 16 and K_TOTAL == 256
        assert PRIVATE_MAX == 248 and len(POSITIONS) == 15 and len(CONSUMERS) == 6
        assert BISECT_LLOYD == 12 and sum([1] * N_HEAD) <= K_TOTAL
        print("ATTENTION0 OV DOWNSTREAM CODEBOOK | dry run: exact payload, response metric, K256 controls")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_model import rope_tables, apply_rot

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    block0 = model.transformer.h[0]
    vocab_ids = torch.arange(VOCAB, device=device)
    fit_mask = vocab_ids.remainder(5) != 4
    select_mask = ~fit_mask

    # Honest task interface: A-SVD only on real routed c_proj inputs from FIT.
    captured = _capture_cproj_input(model, fit_rows, device).to(device)
    weight = block0.attn.c_proj.weight.detach().float()
    a_factor, b_factor = _asvd(weight, captured)
    task_interface = torch.linalg.qr(a_factor[:, :RANK].float(), mode="reduced").Q.to(device)
    rank16_weight = (a_factor[:, :RANK] @ b_factor[:RANK]).to(device)
    full_weight = (a_factor @ b_factor).to(device)
    haar_interfaces = []
    for seed in HAAR_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random = torch.randn(D, RANK, generator=generator)
        haar_interfaces.append(torch.linalg.qr(random, mode="reduced").Q.to(device))

    calibration_ce = {"native": _ce(model, select_rows, device, scoring)}
    calibration_ce["full"] = _ce(
        model, select_rows, device, scoring, cproj_weight=full_weight)
    calibration_ce["rank16"] = _ce(
        model, select_rows, device, scoring, cproj_weight=rank16_weight)
    for index, interface in enumerate(haar_interfaces):
        haar_weight = interface @ interface.T @ block0.attn.c_proj.weight.detach().float()
        calibration_ce[f"haar_{index}"] = _ce(
            model, select_rows, device, scoring, cproj_weight=haar_weight)
    calibration = {
        name: float(values.mean() - calibration_ce["native"].mean())
        for name, values in calibration_ce.items() if name != "native"}
    calibration["native_ce"] = float(calibration_ce["native"].mean())
    calibration["u16_orthogonality_max_abs"] = float(
        (task_interface.T @ task_interface - torch.eye(RANK, device=device)).abs().max())
    calibration["asvd_full_weight_relative_error"] = float(
        (full_weight - weight).double().norm()
        / weight.double().norm().clamp_min(1e-30))
    calibration["local_forward_replay_max_abs"] = _native_forward_replay(
        model, select_rows, facade, device)

    embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))[:VOCAB]
    payload_fold_error = _payload_exactness(model, embedding)
    interfaces = [("task", task_interface), *[
        (f"haar_{index}", interface) for index, interface in enumerate(haar_interfaces)]]

    metrics, codebooks, all_codes = {}, {}, {}
    raw_codebook = None
    # Native attention-write scales for every interface.
    write_samples = []
    for start in range(0, len(fit_rows), DOC_BATCH):
        tokens = fit_rows[start:start + DOC_BATCH, :-1].to(device)
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        attention0, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
        write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
    write_samples = torch.cat(write_samples)

    for name, interface in interfaces:
        sigma = torch.sqrt((write_samples @ interface).double().square().mean(0)).float()
        fit_gram, normalizers, fit_live = _response_metric(
            model, fit_rows, interface, sigma, None, device)
        select_gram, _, select_live = _response_metric(
            model, select_rows, interface, sigma, normalizers, device)
        overlap, fit_rank90, select_rank90 = _leading_subspace_overlap(
            fit_gram, select_gram)
        factor, eigenvalues = _metric_factor(fit_gram)
        _, select_eigenvalues = _metric_factor(select_gram)
        all_a = _payload_codes(model, interface, embedding)
        codebook = _fit_codebook(all_a, all_a, factor, fit_mask, select_mask)
        metrics[name] = {
            "fit_gram": fit_gram.cpu().tolist(),
            "select_gram": select_gram.cpu().tolist(),
            "frobenius_cosine": _frobenius_cosine(fit_gram, select_gram),
            "fit_eigenvalues": eigenvalues.cpu().tolist(),
            "select_eigenvalues": select_eigenvalues.cpu().tolist(),
            "fit_rank90": fit_rank90,
            "select_rank90": select_rank90,
            "rank90_principal_subspace_overlap": overlap,
            "largest_to_median_eigenvalue": float(
                eigenvalues[-1] / eigenvalues[len(eigenvalues) // 2].clamp_min(1e-30)),
            "normalizers": normalizers,
            "fit_live_max_abs": fit_live,
            "select_live_max_abs": select_live,
            "sigma": sigma.cpu().tolist(),
        }
        codebooks[name] = codebook
        all_codes[name] = all_a
        if name == "task":
            raw_codebook = _fit_codebook(
                all_a, all_a, torch.eye(RANK, device=device), fit_mask, select_mask)

    # Row permutation is distribution-preserving; execute it to expose that fact.
    permutation_codebook = _fit_codebook(
        all_codes["task"], all_codes["task"],
        _metric_factor(torch.tensor(metrics["task"]["fit_gram"], device=device))[0],
        fit_mask, select_mask, permute=True)

    task_factor = _metric_factor(
        torch.tensor(metrics["task"]["fit_gram"], device=device))[0]
    transport = _transport(
        model, select_rows, device, task_interface, all_codes["task"],
        codebooks["task"], task_factor, rope_tables, apply_rot, scoring)

    task = codebooks["task"]
    raw_advantage = raw_codebook["global_advantage"]
    half_advantages = [value["global_advantage"] for value in task["heldout_halves"].values()]
    pred_a = (
        payload_fold_error <= 1e-10
        and abs(calibration["full"]) < 1e-3
        and calibration["rank16"] <= .12
        and all(calibration[f"haar_{index}"] >= .80 for index in range(3))
        and calibration["u16_orthogonality_max_abs"] <= 2e-5
        and calibration["local_forward_replay_max_abs"] <= 2e-5
        and transport["u16_edge_replay_after_measured_remainder_max_abs"] <= 2e-5
        and transport[
            "u16_edge_relative_squared_error_after_measured_remainder"] <= 1e-12
        and _digest(fit_rows) != _digest(select_rows))
    metric = metrics["task"]
    pred_b = (
        metric["frobenius_cosine"] >= .85
        and abs(metric["fit_rank90"] - metric["select_rank90"]) <= 2
        and metric["largest_to_median_eigenvalue"] >= 4
        and min(metric["fit_live_max_abs"].values()) > 0
        and min(metric["select_live_max_abs"].values()) > 0)
    pred_c = (
        task["global_advantage"] >= .15
        and task["diverse_center_fraction"] >= .25
        and min(half_advantages) >= .15
        and task["global_advantage"] - raw_advantage >= .05)
    global_r2 = transport["routed_u16_r2"]["global"]
    private_r2 = transport["routed_u16_r2"]["private"]
    global_consumer = transport["consumer_r2"]["global"]
    private_consumer = transport["consumer_r2"]["private"]
    pred_d = (
        global_r2 >= .70
        and min(global_consumer.values()) >= .60
        and global_r2 - private_r2 >= .05
        and transport["mean_consumer_r2"]["global"]
            - transport["mean_consumer_r2"]["private"] >= .05
        and all(transport["ce"]["global"]["wave_damage"][wave]
                <= transport["ce"]["private"]["wave_damage"][wave] + .01
                for wave in range(2)))
    strong_null = (
        not pred_a
        or task["global_advantage"] < .02
        or task["diverse_center_fraction"] < .10
        or transport["mean_consumer_r2"]["global"] <= .30
        or (global_r2 < private_r2
            and transport["mean_consumer_r2"]["global"]
                < transport["mean_consumer_r2"]["private"]))

    def public_codebook(value):
        return {
            "private_center_allocation": value["allocation"],
            "fit_global_sse": value["fit_global_sse"],
            "fit_private_sse": value["fit_private_sse"],
            "heldout_global_distortion": value["global_distortion"],
            "heldout_private_distortion": value["private_distortion"],
            "heldout_global_advantage": value["global_advantage"],
            "diverse_center_fraction": value["diverse_center_fraction"],
            "occupied_centers": value["occupied_centers"],
            "heldout_halves": value["heldout_halves"],
        }

    result = {
        "status": "attention0_ov_downstream_codebook_complete",
        "rung": 419,
        "claim_level": "gauge_invariant_subhead_payload_identification_screen_not_compression",
        "definition": {
            "payload": "P_h(t)=O_h V_h RMSNorm(embedding(t))",
            "edge": "native head-specific QK1*QK2 scalar times P_h(source_token)",
            "feature_basis": "task-useful routed-write U16; head is provenance only",
            "metric": "equal-block finite centered response of MLP0, attention1 q/k/q2/k2, and fresh value",
            "code": "one-sparse deterministic bisecting-kmeans K256",
        },
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows),
                      "FIT_sha256": _digest(fit_rows), "SELECT_sha256": _digest(select_rows),
                      "FINAL_opened": 0},
        "tokens": {"real": VOCAB, "FIT_mod_not4": int(fit_mask.sum()),
                   "SELECT_mod4": int(select_mask.sum()), "FINAL_opened": 0},
        "positions": list(POSITIONS),
        "exactness": {"payload_float64_fold_max_abs": payload_fold_error,
                      "natural_u16_edge_replay_before_remainder_max_abs":
                          transport["u16_edge_replay_before_remainder_max_abs"],
                      "natural_u16_edge_replay_after_measured_remainder_max_abs":
                          transport["u16_edge_replay_after_measured_remainder_max_abs"],
                      "natural_u16_edge_relative_squared_error_before_remainder":
                          transport["u16_edge_relative_squared_error_before_remainder"],
                      "natural_u16_edge_relative_squared_error_after_measured_remainder":
                          transport["u16_edge_relative_squared_error_after_measured_remainder"]},
        "interface_calibration": calibration,
        "response_metrics": metrics,
        "codebooks": {name: public_codebook(value) for name, value in codebooks.items()},
        "raw_u16_codebook": public_codebook(raw_codebook),
        "token_row_permutation_control": public_codebook(permutation_codebook),
        "natural_transport": transport,
        'pred_a_exact_interface_calibration': bool(pred_a),
        'pred_b_downstream_metric_stable_nontrivial': bool(pred_b),
        'pred_c_cross_head_shared_payload_vocabulary': bool(pred_c),
        'pred_d_native_qk_routed_transport': bool(pred_d),
        "null_no_discrete_shared_downstream_ov_vocabulary_k256": bool(strong_null),
        "next_step": (
            "joint_qk_ov_atom_swap_and_removal_ce" if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "geometric_payload_only_no_routed_claim" if pred_a and pred_c and not pred_d
            else "continuous_block_term_complete_qk_times_ov" if pred_a and strong_null
            else "instrument_repair_only"),
        "compression_or_adoption_licensed": False,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "exactness": result["exactness"],
        "calibration": calibration, "metric": metrics["task"],
        "task_codebook": result["codebooks"]["task"],
        "raw_codebook": result["raw_u16_codebook"],
        "transport": transport,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c, "pred_d": pred_d,
        "strong_null": strong_null, "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 OV DOWNSTREAM CODEBOOK DONE", flush=True)


if __name__ == "__main__":
    main()
