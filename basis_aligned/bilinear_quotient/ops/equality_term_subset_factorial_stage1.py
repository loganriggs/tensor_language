#!/usr/bin/env python3
"""RUNG457 -- NATURAL-TEXT EQUALITY-TERM SUBSET IDENTIFICATION.

Preregistered in EQUALITY_TERM_SUBSET_FACTORIAL_STAGE1_PREREGISTRATION.md.
This stage opens only the already-used natural-text role's previously unopened subset
outcomes.  It does not load or score the code OOD role.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

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

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import circuit_campaign_runtime as runtime
import circuit_induction_tensor as induction


PREREG = POLY / "EQUALITY_TERM_SUBSET_FACTORIAL_STAGE1_PREREGISTRATION.md"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt"
OUT = ROOT / "equality_term_subset_factorial_stage1_results.json"
BUNDLE = ROOT / "equality_term_subset_factorial_stage1_sufficient_statistics.pt"
TERMS = (("L5H5", 5, 5), ("L7H3", 7, 3), ("L8H3", 8, 3), ("L8H4", 8, 4))
SITE_HEADS = {5: (5,), 7: (3,), 8: (3, 4)}
SUBSETS = tuple(range(16))
CELLS = (
    "matched_positive", "matched_negative", "all_positive", "near_positive",
    "far_positive", "one_predecessor_positive", "multiple_predecessor_positive",
    "off_target", "all",
)
ARMS = (
    "native",
    *(f"remove:{mask:04b}" for mask in SUBSETS),
    *(f"extract:{mask:04b}" for mask in SUBSETS),
)
DOCUMENTS = 192
TOKENS = 256
BATCH = 4
D = 1152
HEADS = 9
HEAD_DIM = 128
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = "equality-term-subset-factorial-stage1:bootstrap:0"
ARM_IDENTITY_SHA256 = "f8be9a80cc5451cda8c10ecbf1a025e856d9bc15b519c3284337d0a5d93d0b79"
HASHES = {
    PREREG: "aeadd681b24932813814c52dcb10fe3b476644ec1be440bd5624a65d7a737159",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759",
    POLY / "induction_equality_tensor_discovery.py":
        "440573992c1fcfd03d290c971c9547d65632a2821c8b7ba3aaef27d3ac521855",
    POLY / "circuit_induction_tensor.py":
        "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    POLY / "circuit_campaign_runtime.py":
        "6c00f9cd3ade035a1172327ad729c98ccbbab10c9957de2314d82930d4203a0f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _linear(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(state, weight.to(dtype=state.dtype, device=state.device))


def arm_parts(arm: str) -> tuple[str, int]:
    if arm == "native":
        return "native", 0
    mode, bits = arm.split(":", 1)
    if mode not in {"remove", "extract"} or len(bits) != 4 or set(bits) - {"0", "1"}:
        raise ValueError(f"malformed arm: {arm}")
    return mode, int(bits, 2)


def _selected(mask: int, site: int, head: int) -> bool:
    index = next(i for i, (_, term_site, term_head) in enumerate(TERMS)
                 if (term_site, term_head) == (site, head))
    return bool(mask & (1 << index))


@torch.no_grad()
def replay_site_arm(
    state: torch.Tensor,
    first_value: torch.Tensor,
    attention: torch.nn.Module,
    site: int,
    mode: str,
    subset: int,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Replay one attention site and apply exactly one registered subset arm."""

    batch, length, width = state.shape
    if width != D or first_value.shape != (batch, length, HEADS, HEAD_DIM):
        raise ValueError("attention replay interface changed")
    q = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(batch, length, HEADS, HEAD_DIM)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value.view_as(raw_value)
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_DIM,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_DIM
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    pattern = score1 * score2
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    pattern = pattern.masked_fill(~causal, 0)
    heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    for head in SITE_HEADS[site]:
        equality = induction.contract_induction_fetch(
            pattern[:, head], value[:, :, head], tokens,
        )
        chosen = _selected(subset, site, head)
        if mode == "remove":
            if chosen:
                heads[:, head] = heads[:, head] - equality
        elif mode == "extract":
            heads[:, head] = equality if chosen else 0
        else:
            raise ValueError(f"unknown analytical mode: {mode}")
    flattened = heads.transpose(1, 2).contiguous().view(batch, length, width)
    return _linear(flattened, attention.c_proj.weight)


def build_masks(rows: torch.Tensor, stored: Mapping[str, object]) -> dict[str, torch.Tensor]:
    all_positive = stored["all_positive"].clone()
    masks = {
        "matched_positive": stored["positive"].clone(),
        "matched_negative": stored["matched_negative"].clone(),
        "all_positive": all_positive,
        "off_target": stored["off_target"].clone(),
    }
    for name in ("near_positive", "far_positive", "one_predecessor_positive",
                 "multiple_predecessor_positive"):
        masks[name] = torch.zeros_like(all_positive)
    for row_index, row in enumerate(rows):
        for query in range(64, 256):
            if not bool(all_positive[row_index, query]):
                continue
            predecessors = torch.nonzero(row[:query] == row[query], as_tuple=False).flatten()
            distance = query - int(predecessors[-1])
            masks["near_positive" if distance <= 16 else "far_positive"][row_index, query] = True
            masks[
                "one_predecessor_positive" if len(predecessors) == 1
                else "multiple_predecessor_positive"
            ][row_index, query] = True
    masks["all"] = torch.zeros_like(all_positive)
    masks["all"][:, 64:256] = True
    return {name: masks[name] for name in CELLS}


EXPECTED_SUPPORT = {
    "matched_positive": (225, 121), "matched_negative": (225, 131),
    "all_positive": (3084, 191), "near_positive": (719, 160),
    "far_positive": (2365, 191), "one_predecessor_positive": (1366, 190),
    "multiple_predecessor_positive": (1718, 185), "off_target": (33555, 192),
    "all": (36864, 192),
}


def validate_inputs() -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, object]]:
    for path, digest in HASHES.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(ROW_RECEIPT.read_text())
    entry = receipt["entries"]["final_natural"]
    if receipt.get("status") != "frozen_before_any_v2_model_forward" \
            or entry.get("file_sha256") != HASHES[ROWS] \
            or receipt.get("roles", {}).get("final_natural") != "one_shot_final":
        raise RuntimeError("natural row authority changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("schema") != "induction_equality_tensor_final_ood_v2_role" \
            or payload.get("role") != "final_natural" \
            or list(payload["rows"].shape) != [DOCUMENTS, TOKENS + 1] \
            or len(payload["records"]) != DOCUMENTS:
        raise RuntimeError("natural row payload changed")
    masks = build_masks(payload["rows"], payload["copy_cells"])
    if list(masks) != list(CELLS):
        raise RuntimeError("task-condition order changed")
    support = {}
    for name, mask in masks.items():
        observed = (int(mask.sum()), int(mask.any(1).sum()))
        if observed != EXPECTED_SUPPORT[name]:
            raise RuntimeError(f"task-condition support changed: {name} {observed}")
        support[name] = {
            "tokens": observed[0], "documents": observed[1],
            "mask_sha256": tensor_sha256(mask),
        }
    if not torch.equal(masks["near_positive"] | masks["far_positive"], masks["all_positive"]) \
            or bool((masks["near_positive"] & masks["far_positive"]).any()) \
            or not torch.equal(
                masks["one_predecessor_positive"] | masks["multiple_predecessor_positive"],
                masks["all_positive"],
            ) \
            or bool((masks["one_predecessor_positive"] &
                     masks["multiple_predecessor_positive"]).any()):
        raise RuntimeError("derived task conditions do not form exact partitions")
    document_ids = tuple(str(record["document_id"]) for record in payload["records"])
    if len(set(document_ids)) != DOCUMENTS:
        raise RuntimeError("natural documents are not unique")
    metadata = {
        "document_ids_sha256": hashlib.sha256("\0".join(document_ids).encode()).hexdigest(),
        "support": support,
        "row_file_sha256": sha256(ROWS),
        "row_receipt_sha256": sha256(ROW_RECEIPT),
    }
    return payload, masks, metadata


def make_plan() -> runtime.CircuitPlan:
    native = runtime.ArmPlan.build("native", runtime.ArmKind.NATIVE)
    candidates = tuple(runtime.ArmPlan.build(
        arm, runtime.ArmKind.CANDIDATE,
        attention_replacements={site: f"{arm}:L{site}" for site in SITE_HEADS},
    ) for arm in ARMS[1:])
    return runtime.CircuitPlan("equality-term-subset-factorial-stage1", 18, (native, *candidates))


def _accumulate(
    arm_index: int,
    document_start: int,
    masks: Mapping[str, torch.Tensor],
    target_nll: torch.Tensor,
    kl: torch.Tensor,
    correct: torch.Tensor,
    loss_sums: torch.Tensor,
    kl_sums: torch.Tensor,
    correct_sums: torch.Tensor,
) -> None:
    for local_document in range(len(target_nll)):
        document = document_start + local_document
        for cell_index, cell in enumerate(CELLS):
            selected = masks[cell][document]
            loss_sums[arm_index, document, cell_index] = target_nll[local_document, selected].double().sum().cpu()
            kl_sums[arm_index, document, cell_index] = kl[local_document, selected].double().sum().cpu()
            correct_sums[arm_index, document, cell_index] = correct[local_document, selected].double().sum().cpu()


@torch.no_grad()
def collect(model: torch.nn.Module, payload: Mapping[str, object],
            masks: Mapping[str, torch.Tensor]) -> dict[str, object]:
    rows = payload["rows"]
    plan = make_plan()
    loss_sums = torch.zeros(len(ARMS), DOCUMENTS, len(CELLS), dtype=torch.float64)
    kl_sums = torch.zeros_like(loss_sums)
    correct_sums = torch.zeros_like(loss_sums)
    counts = torch.stack([
        torch.tensor([int(mask[row].sum()) for row in range(DOCUMENTS)], dtype=torch.float64)
        for mask in (masks[cell] for cell in CELLS)
    ], dim=1)
    outer = {arm: {"forwards": 0, "returns": 0, "documents": 0} for arm in ARMS}
    sites = {arm: [[0, 0, 0, 0] for _ in range(18)] for arm in ARMS}
    replay_max_abs = 0.0
    replay_relative_squared = 0.0
    device = next(model.parameters()).device
    peak_memory = 0
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        native_owner = runtime.CircuitForwardOwner(plan=plan, arm="native")
        native_logits = native_owner.run(model, tokens, require_production=True)
        native_logprob = F.log_softmax(native_logits.float(), dim=-1)
        native_probability = native_logprob.exp()
        native_nll = -native_logprob.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        native_correct = native_logits.argmax(-1) == targets
        zero_kl = torch.zeros_like(native_nll)
        _accumulate(0, start, masks, native_nll, zero_kl, native_correct,
                    loss_sums, kl_sums, correct_sums)
        closure = native_owner.closure
        outer["native"] = {
            "forwards": outer["native"]["forwards"] + closure.completed_outer_forwards,
            "returns": outer["native"]["returns"] + closure.outer_returns,
            "documents": outer["native"]["documents"] + closure.document_count,
        }
        for site, value in enumerate(closure.sites):
            counts_site = (value.native_attention_calls, value.replacement_attention_calls,
                           value.native_mlp_calls, value.replacement_mlp_calls)
            sites["native"][site] = [a + b for a, b in zip(sites["native"][site], counts_site)]
        for arm_index, arm in enumerate(ARMS[1:], start=1):
            mode, subset = arm_parts(arm)
            callbacks = {}
            for site in SITE_HEADS:
                attention = model.transformer.h[site].attn
                def callback(event, *, site=site, attention=attention, mode=mode,
                             subset=subset, tokens=tokens):
                    write = replay_site_arm(
                        event.state, event.first_value, attention, site, mode, subset, tokens,
                    )
                    return write, event.first_value
                callbacks[f"{arm}:L{site}"] = callback
            owner = runtime.CircuitForwardOwner(
                plan=plan, arm=arm, attention_replacements=callbacks,
            )
            arm_logits = owner.run(model, tokens, require_production=True)
            if arm == "remove:0000":
                difference = (arm_logits.float() - native_logits.float())
                replay_max_abs = max(replay_max_abs, float(difference.abs().max()))
                denominator = float(native_logits.float().square().sum())
                replay_relative_squared = max(
                    replay_relative_squared,
                    float(difference.square().sum()) / max(denominator, 1e-30),
                )
            arm_logprob = F.log_softmax(arm_logits.float(), dim=-1)
            arm_nll = -arm_logprob.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            point_kl = (native_probability * (native_logprob - arm_logprob)).sum(-1).clamp_min(0)
            arm_correct = arm_logits.argmax(-1) == targets
            _accumulate(arm_index, start, masks, arm_nll, point_kl, arm_correct,
                        loss_sums, kl_sums, correct_sums)
            closure = owner.closure
            outer[arm] = {
                "forwards": outer[arm]["forwards"] + closure.completed_outer_forwards,
                "returns": outer[arm]["returns"] + closure.outer_returns,
                "documents": outer[arm]["documents"] + closure.document_count,
            }
            for site, value in enumerate(closure.sites):
                counts_site = (value.native_attention_calls, value.replacement_attention_calls,
                               value.native_mlp_calls, value.replacement_mlp_calls)
                sites[arm][site] = [a + b for a, b in zip(sites[arm][site], counts_site)]
            del arm_logits, arm_logprob, arm_nll, point_kl
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        del native_logits, native_logprob, native_probability, native_nll, native_correct
    expected = DOCUMENTS // BATCH
    for arm in ARMS:
        if outer[arm] != {"forwards": expected, "returns": expected, "documents": DOCUMENTS}:
            raise RuntimeError(f"outer call census changed: {arm}")
        for site, observed in enumerate(sites[arm]):
            replaced = arm != "native" and site in SITE_HEADS
            wanted = [0, expected, expected, 0] if replaced else [expected, 0, expected, 0]
            if observed != wanted:
                raise RuntimeError(f"site call census changed: {arm} L{site} {observed}")
    return {
        "loss_sums": loss_sums, "kl_sums": kl_sums,
        "correct_sums": correct_sums, "counts": counts,
        "outer": outer, "sites": sites,
        "replay_max_abs": replay_max_abs,
        "replay_relative_squared": replay_relative_squared,
        "peak_gpu_memory_bytes": int(peak_memory),
    }


def _mobius(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    for bit in range(4):
        for mask in range(16):
            if mask & (1 << bit):
                output[..., mask] -= output[..., mask ^ (1 << bit)]
    return output


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    a, b = _rankdata(left.double()), _rankdata(right.double())
    a, b = a - a.mean(), b - b.mean()
    denominator = float(a.norm() * b.norm())
    return float((a @ b) / denominator) if denominator > 0 else 0.0


def shapley(dividends: torch.Tensor) -> torch.Tensor:
    answer = torch.zeros(4, dtype=torch.float64)
    for mask in range(1, 16):
        size = mask.bit_count()
        for bit in range(4):
            if mask & (1 << bit):
                answer[bit] += dividends[mask] / size
    return answer


def classify(point: float, low: float, high: float, floor: float = .006) -> str:
    if high <= -floor:
        return "redundant"
    if low >= floor:
        return "complementary"
    return "additive_or_unresolved"


def half_classify(value: float, floor: float = .006) -> str:
    if value <= -floor:
        return "redundant"
    if value >= floor:
        return "complementary"
    return "additive_or_unresolved"


def analyze(stats: Mapping[str, object]) -> dict[str, object]:
    loss = stats["loss_sums"].double()
    counts = stats["counts"].double()
    remove_indices = torch.tensor([ARMS.index(f"remove:{mask:04b}") for mask in SUBSETS])
    extract_indices = torch.tensor([ARMS.index(f"extract:{mask:04b}") for mask in SUBSETS])

    def effects(document_slice: slice | torch.Tensor | None = None):
        selected_loss = loss if document_slice is None else loss[:, document_slice]
        selected_counts = counts if document_slice is None else counts[document_slice]
        pooled_ce = selected_loss.sum(1) / selected_counts.sum(0).unsqueeze(0)
        extraction = pooled_ce[extract_indices[0]].unsqueeze(0) - pooled_ce[extract_indices]
        removal = pooled_ce[remove_indices] - pooled_ce[remove_indices[0]].unsqueeze(0)
        return pooled_ce, extraction.T.contiguous(), removal.T.contiguous()

    pooled_ce, extraction, removal = effects()
    dividends = _mobius(extraction)
    interaction_masks = tuple(range(1, 16))
    point_vector = dividends[:, interaction_masks].reshape(-1)
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED.encode()).digest()[:8], "little")
    )
    bootstrap_parts = []
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        n = min(500, BOOTSTRAP_DRAWS - start)
        draws = torch.randint(DOCUMENTS, (n, DOCUMENTS), generator=generator)
        weights = torch.zeros(n, DOCUMENTS, dtype=torch.float64)
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        cell_parts = []
        for cell in range(len(CELLS)):
            denominator = weights @ counts[:, cell]
            if bool((denominator <= 0).any()):
                raise RuntimeError("bootstrap task condition has zero support")
            ce = (weights @ loss[:, :, cell].T) / denominator.unsqueeze(1)
            recovered = ce[:, extract_indices[0]].unsqueeze(1) - ce[:, extract_indices]
            cell_parts.append(_mobius(recovered)[:, interaction_masks])
        bootstrap_parts.append(torch.cat(cell_parts, dim=1))
    boot = torch.cat(bootstrap_parts, dim=0)
    deviations = (boot - point_vector.unsqueeze(0)).abs().max(1).values.sort().values
    critical = float(deviations[math.ceil(.95 * BOOTSTRAP_DRAWS) - 1])
    low_vector, high_vector = point_vector - critical, point_vector + critical
    low = low_vector.view(len(CELLS), len(interaction_masks))
    high = high_vector.view(len(CELLS), len(interaction_masks))
    mask_to_column = {mask: mask - 1 for mask in interaction_masks}
    cell_to_index = {name: index for index, name in enumerate(CELLS)}

    halves = []
    for document_slice in (slice(0, 96), slice(96, 192)):
        _, half_extraction, half_removal = effects(document_slice)
        half_dividends = _mobius(half_extraction)
        halves.append({
            "extraction": half_extraction,
            "removal": half_removal,
            "dividends": half_dividends,
            "shapley": torch.stack([shapley(half_dividends[cell])
                                     for cell in range(len(CELLS))]),
        })

    all_positive = cell_to_index["all_positive"]
    pair_mask, early_mask, full_mask = 0b1100, 0b0011, 0b1111
    pair_point = float(dividends[all_positive, pair_mask])
    pair_low = float(low[all_positive, mask_to_column[pair_mask]])
    pair_high = float(high[all_positive, mask_to_column[pair_mask]])
    block_point = float(
        extraction[all_positive, full_mask] - extraction[all_positive, early_mask]
        - extraction[all_positive, pair_mask] + extraction[all_positive, 0]
    )
    # The block contrast has four cells, so use the registered order-two .006 floor.
    block_draw = []
    column_offset = all_positive * len(interaction_masks)
    # Reconstruct the contrast directly from bootstrap CE for an honest interval.
    # It equals the sum of all dividends that touch both blocks.
    cross_masks = [mask for mask in interaction_masks
                   if (mask & early_mask) and (mask & pair_mask)]
    cross_columns = [column_offset + mask_to_column[mask] for mask in cross_masks]
    block_boot = boot[:, cross_columns].sum(1)
    block_deviation = (block_boot - block_point).abs().sort().values
    block_critical = float(block_deviation[math.ceil(.95 * BOOTSTRAP_DRAWS) - 1])
    block_low, block_high = block_point - block_critical, block_point + block_critical

    pair_halves = [float(item["dividends"][all_positive, pair_mask]) for item in halves]
    block_halves = [float(sum(item["dividends"][all_positive, mask] for mask in cross_masks))
                    for item in halves]
    pair_class = classify(pair_point, pair_low, pair_high)
    block_class = classify(block_point, block_low, block_high)
    pair_half_classes = [half_classify(value) for value in pair_halves]
    block_half_classes = [half_classify(value) for value in block_halves]
    shapley_rho = spearman(
        halves[0]["shapley"][all_positive], halves[1]["shapley"][all_positive],
    )

    specializations = []
    for left_name, right_name in (
        ("near_positive", "far_positive"),
        ("one_predecessor_positive", "multiple_predecessor_positive"),
    ):
        left, right = cell_to_index[left_name], cell_to_index[right_name]
        singleton_difference = extraction[left, [1, 2, 4, 8]] - extraction[right, [1, 2, 4, 8]]
        winner = int(singleton_difference.abs().argmax())
        pair_difference = float(dividends[left, pair_mask] - dividends[right, pair_mask])
        half_singletons = [
            item["extraction"][left, [1, 2, 4, 8]]
            - item["extraction"][right, [1, 2, 4, 8]] for item in halves
        ]
        half_pairs = [float(item["dividends"][left, pair_mask]
                            - item["dividends"][right, pair_mask]) for item in halves]
        singleton_stable = (
            abs(float(singleton_difference[winner])) >= .012
            and all(math.copysign(1, float(value[winner]))
                    == math.copysign(1, float(singleton_difference[winner]))
                    for value in half_singletons)
            and spearman(half_singletons[0], half_singletons[1]) >= .70
        )
        pair_stable = (
            abs(pair_difference) >= .012
            and all(abs(value) >= .012 and math.copysign(1, value) == math.copysign(1, pair_difference)
                    for value in half_pairs)
        )
        specializations.append({
            "contrast": f"{left_name}_minus_{right_name}",
            "singleton_difference": singleton_difference.tolist(),
            "largest_singleton_term": TERMS[winner][0],
            "half_singleton_differences": [value.tolist() for value in half_singletons],
            "half_singleton_spearman": spearman(half_singletons[0], half_singletons[1]),
            "pair_interaction_difference": pair_difference,
            "half_pair_interaction_differences": half_pairs,
            "singleton_specialization_stable": singleton_stable,
            "pair_specialization_stable": pair_stable,
        })
    stable_specialization = any(
        item["singleton_specialization_stable"] or item["pair_specialization_stable"]
        for item in specializations
    )

    full_recovery = {
        cell: float(extraction[index, full_mask]) for index, cell in enumerate(CELLS)
    }
    dominated = (
        abs(full_recovery["all_positive"]) <= abs(full_recovery["matched_negative"])
        and abs(full_recovery["all_positive"]) <= abs(full_recovery["off_target"])
        and not stable_specialization
    )
    pair_stable = (
        pair_class != "additive_or_unresolved"
        and pair_half_classes[0] == pair_class == pair_half_classes[1]
        and all(abs(value) >= .006 for value in pair_halves)
    )
    grouping_eligible = pair_stable and shapley_rho >= .70 and not dominated
    pred_b = pair_class != "additive_or_unresolved"
    pred_c = block_class != "additive_or_unresolved"
    pred_d = stable_specialization
    pred_e = grouping_eligible

    pooled = {}
    for arm_index, arm in enumerate(ARMS):
        pooled[arm] = {}
        for cell_index, cell in enumerate(CELLS):
            n = float(counts[:, cell_index].sum())
            pooled[arm][cell] = {
                "tokens": int(n),
                "ce": float(loss[arm_index, :, cell_index].sum() / n),
                "native_to_arm_kl": float(stats["kl_sums"][arm_index, :, cell_index].sum() / n),
                "top1_accuracy": float(stats["correct_sums"][arm_index, :, cell_index].sum() / n),
            }
    pred_a = (
        float(stats["replay_relative_squared"]) <= 1e-12
        and pooled["remove:1111"]["matched_positive"]["ce"]
            > pooled["remove:0000"]["matched_positive"]["ce"]
        and full_recovery["matched_positive"] > 0
    )
    all_effects_below_floor = bool(
        (extraction[:, [1, 2, 4, 8]].abs() < .006).all()
        and (dividends[:, [3, 5, 6, 9, 10, 12]].abs() < .006).all()
    )
    sign_instability = (
        pair_class != "additive_or_unresolved" and not pair_stable
    ) or (
        block_class != "additive_or_unresolved"
        and not (block_half_classes[0] == block_class == block_half_classes[1])
    )
    strong_null = (not pred_a) or all_effects_below_floor or sign_instability or dominated

    interaction_reports = {}
    for cell_index, cell in enumerate(CELLS):
        interaction_reports[cell] = {}
        for mask in interaction_masks:
            interaction_reports[cell][f"{mask:04b}"] = {
                "terms": [TERMS[bit][0] for bit in range(4) if mask & (1 << bit)],
                "point_nat": float(dividends[cell_index, mask]),
                "simultaneous_low": float(low[cell_index, mask_to_column[mask]]),
                "simultaneous_high": float(high[cell_index, mask_to_column[mask]]),
            }
    return {
        "pooled_arm_cells": pooled,
        "extraction_recovery_nat": {
            cell: {f"{mask:04b}": float(extraction[cell_index, mask]) for mask in SUBSETS}
            for cell_index, cell in enumerate(CELLS)
        },
        "removal_damage_nat": {
            cell: {f"{mask:04b}": float(removal[cell_index, mask]) for mask in SUBSETS}
            for cell_index, cell in enumerate(CELLS)
        },
        "interactions": interaction_reports,
        "primary_pair": {
            "terms": ["L8H3", "L8H4"], "point_nat": pair_point,
            "simultaneous_low": pair_low, "simultaneous_high": pair_high,
            "classification": pair_class, "half_points_nat": pair_halves,
            "half_classifications": pair_half_classes, "stable": pair_stable,
        },
        "early_vs_layer8_block": {
            "early_terms": ["L5H5", "L7H3"], "layer8_terms": ["L8H3", "L8H4"],
            "point_nat": block_point, "simultaneous_low": block_low,
            "simultaneous_high": block_high, "classification": block_class,
            "half_points_nat": block_halves, "half_classifications": block_half_classes,
        },
        "shapley_all_positive": {
            "full": shapley(dividends[all_positive]).tolist(),
            "halves": [item["shapley"][all_positive].tolist() for item in halves],
            "half_spearman": shapley_rho,
        },
        "context_specialization": specializations,
        "full_set_recovery_by_condition": full_recovery,
        "dominated_by_negative_or_off_target": dominated,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "simultaneous_absolute_critical_nat": critical,
        },
        'pred_a_instrument_and_endpoint_liveness': pred_a,
        'pred_b_primary_pair_resolved': pred_b,
        'pred_c_cross_layer_composition_resolved': pred_c,
        'pred_d_context_specialization_stable': pred_d,
        'pred_e_natural_grouping_eligible_for_code_confirmation': pred_e,
        "strong_null": strong_null,
        "next_step": (
            "freeze_group_and_preregister_code_confirmation"
            if grouping_eligible else
            "decompose_equality_terms_below_head_grain_using_qk_value_or_downstream_readers"
        ),
    }


def main() -> None:
    started = time.time()
    payload, masks, metadata = validate_inputs()
    if len(ARMS) != 33 or len(set(ARMS)) != len(ARMS) \
            or [arm_parts(arm) for arm in ARMS[1:5]] != [
                ("remove", 0), ("remove", 1), ("remove", 2), ("remove", 3),
            ] or hashlib.sha256("\0".join(ARMS).encode()).hexdigest() != ARM_IDENTITY_SHA256:
        raise RuntimeError("subset arm enumeration changed")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        toy = torch.arange(16, dtype=torch.float64).square()
        if not torch.allclose(_mobius(toy).sum(), toy[-1]):
            raise RuntimeError("Möbius analysis identity failed")
        print(json.dumps({
            "status": "dry_run_passed", "rung": 457,
            "arms": len(ARMS), "subsets_per_background": len(SUBSETS),
            "documents": DOCUMENTS, "cells": list(CELLS), "metadata": metadata,
            "code_ood_loaded": False, "model_loaded": False,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung457 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, _ = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    measured = collect(model, payload, masks)
    analysis = analyze(measured)
    bundle = {
        "schema": "equality_term_subset_factorial_stage1_sufficient_statistics_v1",
        "arms": ARMS, "cells": CELLS, "terms": TERMS,
        "loss_sums": measured["loss_sums"], "kl_sums": measured["kl_sums"],
        "correct_sums": measured["correct_sums"], "counts": measured["counts"],
        "raw_rows_or_tokens_included": False, "logits_or_hidden_states_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 457,
        "claim_level": "natural_text_subset_identification_not_fresh_final_or_adoption",
        "terms": TERMS, "arms": ARMS, "cells": CELLS,
        "arm_identity_sha256": ARM_IDENTITY_SHA256,
        "input_identity": metadata,
        "code_ood_loaded": False,
        "sealed_attention0_confirmation_opened": False,
        "call_census": {"outer": measured["outer"], "sites": measured["sites"]},
        "replay": {
            "max_abs": measured["replay_max_abs"],
            "relative_squared": measured["replay_relative_squared"],
        },
        "execution_price": {
            "analytical_configurations": 32,
            "native_integrity_configuration": 1,
            "documents": DOCUMENTS,
            "outer_forwards": len(ARMS) * (DOCUMENTS // BATCH),
            "peak_gpu_memory_bytes": measured["peak_gpu_memory_bytes"],
            "deployed_parameters_saved": 0,
        },
        "analysis": analysis,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 457,
        "predictions": {key: value for key, value in analysis.items() if key.startswith("pred_")},
        "strong_null": analysis["strong_null"],
        "primary_pair": analysis["primary_pair"],
        "early_vs_layer8_block": analysis["early_vs_layer8_block"],
        "context_specialization": analysis["context_specialization"],
        "runtime_s": result["runtime_s"], "next_step": analysis["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
