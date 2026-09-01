"""RUNG 298 — EMBEDDING-FOLDED MLP0 STRUCTURE SCREEN.

Question
--------
Can a known finite embedding population plus a bilinear layer expose a small
block/hierarchy/DAG/finite-router program in the weights/function itself, rather
than only through an SAE on sampled output activations?

This is a SCREEN, not an adoption run.  It has two preregistered halves.

I. Planted identifiability assay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Eight discrete input states play the role of token classes.  A bilinear teacher
uses hidden atoms whose state supports form a known family: root, overlapping
four-state parents, two-state blocks, and singleton leaves.  Set inclusion is a
DAG (some leaves have multiple parents), and singleton supports are fixed MoE
router states.  Random CP permutation/rescaling and Left/Right swaps hide raw
unit identity without changing the function.

Recover each atom's support only from its exhaustive activation profile over the
states.  Then independently train a dense bilinear student on teacher input/output
pairs and ask whether a functionally equivalent factorization recovers the same
support family.  This distinguishes structure present in one planted gauge from
structure identifiable from the function.

II. Real MLP0 finite-vocabulary screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For every one of the 50,257 real GPT-2 tokens, execute block 0 at sequence length
one and capture the exact normalized MLP0 input x.  Compute the native bilinear
gate H=(Lx)*(Rx).  Cluster a fixed random projection of output-weighted gates into
S in {4,8,16,32} states on a frozen 4/5 vocabulary split.  For each state, freeze
the K=512 units with largest fit-population contribution energy.  Train a linear
router x->state on the fit split and evaluate on untouched token ids.

The executable candidate computes the small router, then only the fixed subset
for its predicted state.  Its stored expert price counts the UNION of all selected
Left/Right rows and Down columns plus router weights and Down_bias.  Clustering and
the random projection are fit-time only and receive no deployment credit.  A
global fixed-K subset is the matched structural null; per-token top-K is reported
only as a combinatorial compute-policy upper comparator.

Frozen predictions
------------------
pred_a_identifiable_support_poset:
    The independently trained student reaches held-out output R2 >= 0.98 AND
    recovers the planted support family with mean best Jaccard >= 0.85 and
    inclusion-reachability F1 >= 0.85.  If R2 holds but either structure bar fails,
    the bilinear function is not canonically factorized without an extra prior.
pred_b_small_router_beats_global:
    For some S <= 32, held-out linear-router accuracy is >= 0.70 and its state-fixed
    K=512 program reduces relative output MSE by >= 15% against global fixed-K.
pred_c_literal_half_mlp0_screen:
    For some S <= 32, the selected union is <= 2304 of 4608 units and the legal
    router program has centered output R2 >= 0.80.  This is only a storage/function
    screen; live CE, composition, certificates and interventions would still gate it.

Positive-control requirement: the gauge-scrambled planted TEACHER itself must have
mean support-family Jaccard and reachability F1 >= 0.98.  Otherwise the recovery
instrument is invalid and none of the student conclusions are scored.

Null
----
If no S improves global fixed-K MSE by 5%, or router accuracy is at most twice
chance, the finite-state interpretation has no real-model signal at this K.  If the
student fits the toy but misses the support bars, raw bilinear weights alone are
non-identifying and structured priors/interventions are required.

Literal prices
--------------
Native MLP0: 4608*(1152+1152+1152)+1152 = 15,926,400 scalars.
Candidate: |union|*3456 + S*1152 + S + 1152 scalars.  The already-required input
embedding and exact block-0 attention are not double-counted.  No claim is made that
this one-layer price can be subtracted from the adopted whole program until a live
replacement removes the corresponding native tensors.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


OUT = "/workspace/tensor_language/basis_aligned/bilinear_quotient/mlp0_embedding_fold_structure_screen_results.json"
DEV = "cuda"
TOY_STATES = 8
TOY_Z = 8
TOY_D = TOY_STATES + TOY_Z
TOY_OUT = 24
TOY_UNITS_PER_MASK = 4
TOY_TRAIN = 8192
TOY_TEST = 4096
TOY_STEPS = 2500
REAL_V = 50257
REAL_D = 1152
REAL_H = 4608
REAL_K = 512
REAL_STATES = (4, 8, 16, 32)
REAL_PROJ = 48
ROUTER_STEPS = 180


def _mask_tuple(*items: int) -> tuple[int, ...]:
    return tuple(sorted(items))


TRUE_MASKS = (
    _mask_tuple(0, 1, 2, 3, 4, 5, 6, 7),
    _mask_tuple(0, 1, 2, 3),
    _mask_tuple(2, 3, 4, 5),
    _mask_tuple(4, 5, 6, 7),
    _mask_tuple(0, 1),
    _mask_tuple(2, 3),
    _mask_tuple(4, 5),
    _mask_tuple(6, 7),
    _mask_tuple(0),
    _mask_tuple(1),
    _mask_tuple(2),
    _mask_tuple(3),
    _mask_tuple(4),
    _mask_tuple(5),
    _mask_tuple(6),
    _mask_tuple(7),
)
TOY_H = len(TRUE_MASKS) * TOY_UNITS_PER_MASK


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / max(len(aa | bb), 1)


def _reachability(masks: list[tuple[int, ...]]) -> np.ndarray:
    n = len(masks)
    result = np.zeros((n, n), dtype=bool)
    sets = [set(x) for x in masks]
    for i in range(n):
        for j in range(n):
            result[i, j] = i != j and sets[i] > sets[j]
    return result


def _binary_f1(truth: np.ndarray, estimate: np.ndarray) -> float:
    keep = ~np.eye(truth.shape[0], dtype=bool)
    t, e = truth[keep], estimate[keep]
    tp = int(np.logical_and(t, e).sum())
    fp = int(np.logical_and(~t, e).sum())
    fn = int(np.logical_and(t, ~e).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _infer_masks(hidden: torch.Tensor, state: torch.Tensor) -> list[tuple[int, ...]]:
    profiles = []
    for st in range(TOY_STATES):
        profiles.append(hidden[state == st].abs().mean(0))
    profile = torch.stack(profiles, 1)
    threshold = 0.08 * profile.amax(1, keepdim=True).clamp_min(1e-10)
    inferred = []
    for row in profile:
        support = tuple(torch.nonzero(row > 0.08 * row.max().clamp_min(1e-10)).flatten().tolist())
        inferred.append(support)
    return inferred


def _family_score(inferred: list[tuple[int, ...]]) -> dict[str, object]:
    chosen = []
    best_values = []
    for true_mask in TRUE_MASKS:
        values = [_jaccard(true_mask, candidate) for candidate in inferred]
        index = int(np.argmax(values))
        chosen.append(inferred[index])
        best_values.append(values[index])
    truth_reach = _reachability(list(TRUE_MASKS))
    estimate_reach = _reachability(chosen)
    return {
        "mean_best_jaccard": float(np.mean(best_values)),
        "min_best_jaccard": float(np.min(best_values)),
        "reachability_f1": _binary_f1(truth_reach, estimate_reach),
        "matched_masks": [list(x) for x in chosen],
    }


def _toy_inputs(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEV).manual_seed(seed)
    state = torch.randint(0, TOY_STATES, (n,), generator=generator, device=DEV)
    onehot = F.one_hot(state, TOY_STATES).float()
    z = torch.randn(n, TOY_Z, generator=generator, device=DEV)
    return torch.cat((onehot, z), 1), state


def _make_teacher(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEV).manual_seed(seed)
    left = torch.zeros(TOY_H, TOY_D, device=DEV)
    right = torch.zeros_like(left)
    down = torch.randn(TOY_OUT, TOY_H, generator=generator, device=DEV) / math.sqrt(TOY_H)
    for node, support in enumerate(TRUE_MASKS):
        for repeat in range(TOY_UNITS_PER_MASK):
            unit = node * TOY_UNITS_PER_MASK + repeat
            left[unit, list(support)] = 1.0
            right[unit, TOY_STATES:] = torch.randn(
                TOY_Z, generator=generator, device=DEV,
            ) / math.sqrt(TOY_Z)
    permutation = torch.randperm(TOY_H, generator=generator, device=DEV)
    scale_left = 0.5 + torch.rand(TOY_H, generator=generator, device=DEV)
    scale_right = 0.5 + torch.rand(TOY_H, generator=generator, device=DEV)
    left = (scale_left[:, None] * left)[permutation]
    right = (scale_right[:, None] * right)[permutation]
    down = (down / (scale_left * scale_right)[None, :])[:, permutation]
    swap = torch.rand(TOY_H, generator=generator, device=DEV) > 0.5
    left_swap = left.clone()
    left[swap] = right[swap]
    right[swap] = left_swap[swap]
    return down, left, right


def _bilinear(x: torch.Tensor, down: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = (x @ left.T) * (x @ right.T)
    return hidden @ down.T, hidden


def _r2(prediction: torch.Tensor, target: torch.Tensor, centered: bool = True) -> float:
    if centered:
        denominator = (target - target.mean(0, keepdim=True)).square().sum()
    else:
        denominator = target.square().sum()
    return float(1.0 - (prediction - target).square().sum() / denominator.clamp_min(1e-12))


def _run_toy() -> dict[str, object]:
    down_teacher, left_teacher, right_teacher = _make_teacher(29800)
    x_train, state_train = _toy_inputs(TOY_TRAIN, 29801)
    x_test, state_test = _toy_inputs(TOY_TEST, 29802)
    with torch.no_grad():
        y_train, _ = _bilinear(x_train, down_teacher, left_teacher, right_teacher)
        y_test, hidden_teacher = _bilinear(x_test, down_teacher, left_teacher, right_teacher)
        teacher_family = _family_score(_infer_masks(hidden_teacher, state_test))

    generator = torch.Generator(device=DEV).manual_seed(29803)
    left_student = (torch.randn(TOY_H, TOY_D, generator=generator, device=DEV) / math.sqrt(TOY_D)).requires_grad_()
    right_student = (torch.randn(TOY_H, TOY_D, generator=generator, device=DEV) / math.sqrt(TOY_D)).requires_grad_()
    down_student = (torch.randn(TOY_OUT, TOY_H, generator=generator, device=DEV) / math.sqrt(TOY_H)).requires_grad_()
    optimizer = torch.optim.Adam((left_student, right_student, down_student), lr=4e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TOY_STEPS)
    target_scale = y_train.square().mean().detach().clamp_min(1e-8)
    curve = []
    for step in range(TOY_STEPS):
        index = torch.randint(0, TOY_TRAIN, (1024,), generator=generator, device=DEV)
        prediction, _ = _bilinear(x_train[index], down_student, left_student, right_student)
        loss = F.mse_loss(prediction, y_train[index]) / target_scale
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((left_student, right_student, down_student), 10.0)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0 or step == TOY_STEPS - 1:
            with torch.no_grad():
                test_prediction, _ = _bilinear(x_test, down_student, left_student, right_student)
                curve.append({"step": step, "heldout_r2": _r2(test_prediction, y_test)})
    with torch.no_grad():
        prediction, hidden_student = _bilinear(x_test, down_student, left_student, right_student)
        student_family = _family_score(_infer_masks(hidden_student, state_test))
        student_r2 = _r2(prediction, y_test)
    return {
        "teacher_positive_control": teacher_family,
        "student_heldout_r2": student_r2,
        "student_family": student_family,
        "curve": curve,
        "states": TOY_STATES,
        "support_nodes": len(TRUE_MASKS),
        "hidden_units": TOY_H,
    }


def _kmeans(train: torch.Tensor, evaluate: torch.Tensor, states: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=train.device).manual_seed(seed)
    centers = train[torch.randperm(len(train), generator=generator, device=train.device)[:states]].clone()
    train_label = torch.zeros(len(train), dtype=torch.long, device=train.device)
    for _ in range(18):
        train_label = torch.cdist(train, centers).argmin(1)
        updated = []
        for state in range(states):
            members = train[train_label == state]
            if len(members) == 0:
                updated.append(train[torch.randint(0, len(train), (1,), generator=generator, device=train.device)[0]])
            else:
                updated.append(members.mean(0))
        new_centers = torch.stack(updated)
        if torch.equal(new_centers, centers):
            break
        centers = new_centers
    train_label = torch.cdist(train, centers).argmin(1)
    eval_label = torch.cdist(evaluate, centers).argmin(1)
    return train_label, eval_label, centers


def _train_router(x_train: torch.Tensor, labels: torch.Tensor, x_eval: torch.Tensor, states: int, seed: int) -> tuple[torch.Tensor, float]:
    torch.manual_seed(seed)
    router = torch.nn.Linear(x_train.shape[1], states, device=DEV)
    optimizer = torch.optim.AdamW(router.parameters(), lr=8e-3, weight_decay=1e-4)
    generator = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(ROUTER_STEPS):
        index = torch.randint(0, len(x_train), (4096,), generator=generator, device=DEV)
        loss = F.cross_entropy(router(x_train[index]), labels[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((router(x_train).argmax(1) == labels).float().mean())
        eval_prediction = router(x_eval).argmax(1)
    return eval_prediction, train_accuracy


@torch.no_grad()
def _capture_real_population() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/qk_mdl")
    from tier2_model import apply_rot, load_elriggs, rope_tables

    model, cfg = load_elriggs("bilin18")
    heads = cfg["n_head"]
    head_dim = cfg["n_embd"] // heads
    width = cfg["n_embd"]
    block = model.transformer.h[0]
    mlp = block.mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    x_rows = []
    h_rows = []
    for start in range(0, REAL_V, 512):
        token = torch.arange(start, min(start + 512, REAL_V), device=DEV).view(-1, 1)
        batch, length = token.shape
        x0 = F.rms_norm(model.transformer.wte(token), (width,))
        x = block.lambdas[0] * x0 + block.lambdas[1] * x0
        attention = block.attn
        current = F.rms_norm(x, (width,))
        cosine, sine = rope_tables(length, head_dim, DEV, x.dtype, "bf16")
        cosine = cosine[None, :, None, :]
        sine = sine[None, :, None, :]

        def qk(linear: torch.nn.Module) -> torch.Tensor:
            value = F.rms_norm(linear(current).view(batch, length, heads, head_dim), (head_dim,))
            return apply_rot(value, cosine, sine)

        value = attention.c_v(current).view(batch, length, heads, head_dim)
        query, key = qk(attention.c_q), qk(attention.c_k)
        query2, key2 = qk(attention.c_q2), qk(attention.c_k2)
        score = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
        score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
        pattern = score * score2
        output = torch.einsum("bhqk,bkhd->bqhd", pattern, value).reshape(batch, length, width)
        x = x + attention.c_proj(output)
        mlp_input = F.rms_norm(x, (width,)).reshape(batch, width).float()
        hidden = (mlp_input @ left.T) * (mlp_input @ right.T)
        x_rows.append(mlp_input.half().cpu())
        h_rows.append(hidden.half().cpu())
    return torch.cat(x_rows), torch.cat(h_rows), down


@torch.no_grad()
def _approx_output(hidden: torch.Tensor, labels: torch.Tensor, subsets: list[torch.Tensor], down: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(len(hidden), down.shape[0], device=DEV)
    for state, subset in enumerate(subsets):
        index = torch.nonzero(labels == state).flatten()
        if len(index) == 0:
            continue
        result[index] = hidden[index][:, subset] @ down[:, subset].T
    return result


def _run_real() -> dict[str, object]:
    x_cpu, h_cpu, down = _capture_real_population()
    fit_index = torch.arange(REAL_V) % 5 != 0
    eval_index = ~fit_index
    x_fit = x_cpu[fit_index].float().to(DEV)
    x_eval = x_cpu[eval_index].float().to(DEV)
    h_fit = h_cpu[fit_index].float().to(DEV)
    h_eval = h_cpu[eval_index].float().to(DEV)
    down_norm = down.norm(dim=0).clamp_min(1e-8)
    generator = torch.Generator(device=DEV).manual_seed(29820)
    projection = torch.randn(REAL_H, REAL_PROJ, generator=generator, device=DEV) / math.sqrt(REAL_PROJ)
    feature_fit = (h_fit * down_norm) @ projection
    feature_eval = (h_eval * down_norm) @ projection
    mean = feature_fit.mean(0, keepdim=True)
    scale = feature_fit.std(0, keepdim=True).clamp_min(1e-5)
    feature_fit = (feature_fit - mean) / scale
    feature_eval = (feature_eval - mean) / scale
    target = h_eval @ down.T
    target_centered = target - target.mean(0, keepdim=True)

    energy_global = (h_fit.square() * down_norm.square()).mean(0)
    global_subset = energy_global.topk(REAL_K).indices
    global_prediction = h_eval[:, global_subset] @ down[:, global_subset].T
    global_relative_mse = float((global_prediction - target).square().sum() / target_centered.square().sum().clamp_min(1e-12))
    global_r2 = 1.0 - global_relative_mse

    per_token_subset = h_eval.abs().topk(REAL_K, dim=1).indices
    sparse_hidden = torch.zeros_like(h_eval)
    sparse_hidden.scatter_(1, per_token_subset, h_eval.gather(1, per_token_subset))
    topk_prediction = sparse_hidden @ down.T
    topk_relative_mse = float((topk_prediction - target).square().sum() / target_centered.square().sum().clamp_min(1e-12))

    arms = {}
    for states in REAL_STATES:
        train_label, eval_cluster, _ = _kmeans(feature_fit, feature_eval, states, 29830 + states)
        router_label, train_router_accuracy = _train_router(x_fit, train_label, x_eval, states, 29840 + states)
        router_accuracy = float((router_label == eval_cluster).float().mean())
        subsets = []
        for state in range(states):
            members = h_fit[train_label == state]
            if len(members) == 0:
                energy = energy_global
            else:
                energy = (members.square() * down_norm.square()).mean(0)
            subsets.append(energy.topk(REAL_K).indices)
        union = torch.unique(torch.cat(subsets))
        oracle_prediction = _approx_output(h_eval, eval_cluster, subsets, down)
        legal_prediction = _approx_output(h_eval, router_label, subsets, down)
        oracle_relative_mse = float((oracle_prediction - target).square().sum() / target_centered.square().sum().clamp_min(1e-12))
        legal_relative_mse = float((legal_prediction - target).square().sum() / target_centered.square().sum().clamp_min(1e-12))
        improvement = (global_relative_mse - legal_relative_mse) / max(global_relative_mse, 1e-12)
        stored = int(len(union) * 3 * REAL_D + states * REAL_D + states + REAL_D)
        arms[str(states)] = {
            "router_train_accuracy": train_router_accuracy,
            "router_eval_accuracy": router_accuracy,
            "chance_accuracy": 1.0 / states,
            "union_units": int(len(union)),
            "union_fraction": float(len(union) / REAL_H),
            "stored_scalars": stored,
            "native_mlp0_scalars": REAL_H * 3 * REAL_D + REAL_D,
            "storage_fraction_of_native_mlp0": stored / (REAL_H * 3 * REAL_D + REAL_D),
            "oracle_state_r2": 1.0 - oracle_relative_mse,
            "legal_router_r2": 1.0 - legal_relative_mse,
            "legal_mse_improvement_vs_global": improvement,
        }
    return {
        "population": REAL_V,
        "fit_tokens": int(fit_index.sum()),
        "eval_tokens": int(eval_index.sum()),
        "k": REAL_K,
        "global_fixed_k_r2": global_r2,
        "global_fixed_k_relative_mse": global_relative_mse,
        "per_token_topk_r2": 1.0 - topk_relative_mse,
        "arms": arms,
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print("MLP0 EMBEDDING FOLD STRUCTURE SCREEN | dry run: registered toy and real plans valid")
        return
    started = time.time()
    print("MLP0 EMBEDDING FOLD STRUCTURE SCREEN | planted identifiability then exhaustive real vocabulary", flush=True)
    toy = _run_toy()
    positive_valid = (
        toy["teacher_positive_control"]["mean_best_jaccard"] >= 0.98
        and toy["teacher_positive_control"]["reachability_f1"] >= 0.98
    )
    pred_a = bool(
        positive_valid
        and toy["student_heldout_r2"] >= 0.98
        and toy["student_family"]["mean_best_jaccard"] >= 0.85
        and toy["student_family"]["reachability_f1"] >= 0.85
    )
    print("toy:", json.dumps(toy, indent=2), flush=True)
    real = _run_real()
    pred_b = any(
        arm["router_eval_accuracy"] >= 0.70
        and arm["legal_mse_improvement_vs_global"] >= 0.15
        for arm in real["arms"].values()
    )
    pred_c = any(
        arm["union_units"] <= 2304 and arm["legal_router_r2"] >= 0.80
        for arm in real["arms"].values()
    )
    null_real = all(
        arm["legal_mse_improvement_vs_global"] < 0.05
        or arm["router_eval_accuracy"] <= 2.0 * arm["chance_accuracy"]
        for arm in real["arms"].values()
    )
    result = {
        "status": "mlp0_embedding_fold_structure_screen_complete",
        "rung": 298,
        "toy": toy,
        "real": real,
        "positive_control_valid": positive_valid,
        'pred_a_identifiable_support_poset': pred_a,
        'pred_b_small_router_beats_global': bool(pred_b),
        'pred_c_literal_half_mlp0_screen': bool(pred_c),
        "null_real_finite_state_no_signal": bool(null_real),
        "runtime_s": time.time() - started,
        "claim_level": "screen_only",
    }
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print("real:", json.dumps(real, indent=2), flush=True)
    print(
        "predicates:",
        result["pred_a_identifiable_support_poset"],
        result["pred_b_small_router_beats_global"],
        result["pred_c_literal_half_mlp0_screen"],
        "null", result["null_real_finite_state_no_signal"],
        flush=True,
    )
    print("MLP0 EMBEDDING FOLD STRUCTURE SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
