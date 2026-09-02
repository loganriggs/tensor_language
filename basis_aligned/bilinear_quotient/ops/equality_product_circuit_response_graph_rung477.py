#!/usr/bin/env python3
"""RUNG477 -- per-product downstream circuit-response graph.

Registered before opening individual product responses on discovery circuit families:
  pred_a: exact frozen inputs, replay, contraction, support, and execution counts.
  pred_b: at least two MLPs contain many source/half-stable product terms.
  pred_c: a cross-MLP mutual-nearest graph beats circuit-permutation controls.
  pred_d: the graph survives leave-one-top-level-family checks.
  pred_e: both endpoint groups have stable task-selective aggregate responses.
Strong null: invalid instrument, too few stable terms, or no above-control graph.
Literal deployed price: zero parameters saved and zero added.
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

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_query_circuit_fingerprint_rung475 as census_parent
import equality_query_product_group_circuit_fingerprint_rung476 as parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_PRODUCT_CIRCUIT_RESPONSE_GRAPH_RUNG477_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_query_product_group_circuit_fingerprint_rung476_results.json"
PARENT_BUNDLE = ROOT / "equality_query_product_group_circuit_fingerprint_rung476_per_position.pt"
PARENT_SOURCE = ROOT / "ops/equality_query_product_group_circuit_fingerprint_rung476.py"
PRODUCT_SOURCE = ROOT / "ops/equality_mlp_product_term_group_rung467.py"
CENSUS = ROOT / "census_state_diverse.pt"
BATTERY = ROOT / "circuits/BATTERY.json"
OUT = ROOT / "equality_product_circuit_response_graph_rung477_results.json"
BUNDLE = ROOT / "equality_product_circuit_response_graph_rung477_bundle.pt"
SOURCES = parent.SOURCES
SITES = parent.SITES
MODULES = parent.MODULES
PAIRS = parent.PAIRS
PAIR_NAMES = parent.PAIR_NAMES
MASK_TYPES = ("member", "slice_control")
DISCOVERY_STOP = 500
HALF_STOP = 250
BATCH = 4
TOKENS = parent.TOKENS
HIDDEN = product_parent.HIDDEN
DISCOVERY_ROOTS = (0, 2, 4, 6, 8, 18)
VALIDATION_ROOTS = (1, 3, 5, 7, 11, 13, 23)
PERMUTATION_SEEDS = tuple(range(2026090200, 2026090216))
EXPECTED_FORWARDS = math.ceil(DISCOVERY_STOP / BATCH) * (3 + len(SOURCES))
HASHES = {
    PREREG: "365e5b9dcd5122a0ebc4aa224249e3eda5823148165f89ec14c8760dadba69c3",
    PARENT_RESULT: "107a4c9dd8f4c473cdc13fa1ac3719e955427ba5557ff57122e8149f10234e47",
    PARENT_BUNDLE: "6abcdec05d497e65eee34090b4dd4ad4a4cc91583ec2adf1b7bf6951fb9a751e",
    PARENT_SOURCE: "4eaf57248864fd64bf62c1d4d2ff21a6003e3299708e645194d58eeaa0b001c1",
    PRODUCT_SOURCE: "3665fc1b33ebb7bff78f78a9548d75219a43e3a0593e79bed6075a42a821bc8b",
    CENSUS: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    BATTERY: "86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(tag):
    return int(tag.split(".")[1])


def _batch_mask(flat_mask, start, stop):
    return flat_mask.view(parent.DOCUMENTS, TOKENS)[start:stop]


def expected_backwards(circuit_masks, discovery_tags):
    count = 0
    for start in range(0, DISCOVERY_STOP, BATCH):
        stop = min(start + BATCH, DISCOVERY_STOP)
        for tag in discovery_tags:
            for mask_type in MASK_TYPES:
                count += int(bool(_batch_mask(circuit_masks[tag][mask_type], start, stop).any()))
    return count * len(SOURCES)


def support_report(circuit_masks, tags, row_ranges):
    report = {}
    for start, stop in row_ranges:
        row_mask = torch.zeros(parent.DOCUMENTS, TOKENS, dtype=torch.bool)
        row_mask[start:stop] = True
        flat = row_mask.flatten()
        key = f"{start}:{stop}"
        report[key] = {mask_type: {
            tag: int((circuit_masks[tag][mask_type] & flat).sum()) for tag in tags
        } for mask_type in MASK_TYPES}
    return report


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 476 or result.get("pred_a_instrument") is not True \
            or any(result.get(key) is not False for key in (
                "pred_b_selected_group", "pred_c_half_stable",
                "pred_d_beats_controls", "pred_e_separates_complement",
            )) or result.get("strong_null") is not True:
        raise RuntimeError("rung476 registered verdict changed")
    rows, _, positive, circuit_masks, scale, metadata = census_parent.validate_inputs()
    tags = list(circuit_masks)
    discovery_tags = [tag for tag in tags if _root(tag) in DISCOVERY_ROOTS]
    validation_tags = [tag for tag in tags if _root(tag) in VALIDATION_ROOTS]
    if len(discovery_tags) != 32 or len(validation_tags) != 30 \
            or set(discovery_tags) | set(validation_tags) != set(tags) \
            or set(discovery_tags) & set(validation_tags):
        raise RuntimeError("frozen top-level circuit-family split changed")
    discovery_support = support_report(
        circuit_masks, discovery_tags, ((0, HALF_STOP), (HALF_STOP, DISCOVERY_STOP)),
    )
    if min(discovery_support[key]["member"][tag]
           for key in discovery_support for tag in discovery_tags) < 39:
        raise RuntimeError("discovery member support changed")
    if min(discovery_support[key]["slice_control"][tag]
           for key in discovery_support for tag in discovery_tags) < 439:
        raise RuntimeError("discovery matched-control support changed")
    backwards = expected_backwards(circuit_masks, discovery_tags)
    metadata = {
        **metadata,
        "rung476_result_sha256": sha256(PARENT_RESULT),
        "rung476_bundle_sha256": sha256(PARENT_BUNDLE),
        "discovery_documents": [0, DISCOVERY_STOP],
        "document_halves": [[0, HALF_STOP], [HALF_STOP, DISCOVERY_STOP]],
        "discovery_roots": list(DISCOVERY_ROOTS),
        "validation_roots_reserved": list(VALIDATION_ROOTS),
        "discovery_tags": discovery_tags,
        "validation_tags_reserved": validation_tags,
        "discovery_support": discovery_support,
        "expected_backwards": backwards,
    }
    return rows, positive, circuit_masks, scale, discovery_tags, validation_tags, metadata


def _record(audit_totals, key, audit, backwards=0):
    row = audit_totals.setdefault(key, {"forwards": 0, "backwards": 0})
    row["forwards"] += 1
    row["backwards"] += backwards


def _nll(logits, rows):
    targets = rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)


def collect_responses(model, rows, circuit_masks, scale, tags, audit_totals, replay):
    sums = torch.zeros(
        2, len(SOURCES), len(MASK_TYPES), len(SITES), HIDDEN, len(tags), dtype=torch.float64,
    )
    counts = torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    device = next(model.parameters()).device
    reconstruction, contraction_numerator, contraction_denominator, backwards = 0.0, 0.0, 0.0, 0
    for start in range(0, DISCOVERY_STOP, BATCH):
        stop = min(start + BATCH, DISCOVERY_STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung477:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung477:replay", audit, analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        with torch.no_grad():
            _, absent_products, _, audit, error = product_parent.run_term_forward(
                model, tokens, arm="base", capture_products=True,
            )
        _record(audit_totals, "rung477:absent", audit)
        reconstruction = max(reconstruction, error)
        half = int(start >= HALF_STOP)
        active = []
        for ci, tag in enumerate(tags):
            for ki, mask_type in enumerate(MASK_TYPES):
                selected = _batch_mask(circuit_masks[tag][mask_type], start, stop).to(device)
                observed = int(selected.sum())
                counts[half, ki, ci] += observed
                if observed:
                    active.append((ki, ci, selected))
        for si, source in enumerate(SOURCES):
            with torch.enable_grad():
                logits, products, writes, audit, error = product_parent.run_term_forward(
                    model, tokens, arm=source_parent.SOURCE_ARMS[source], scale=scale,
                    capture_products=True, gradient_writes=True,
                )
                _record(audit_totals, f"rung477:source:{source}", audit)
                reconstruction = max(reconstruction, error)
                nll = _nll(logits, batch_rows)
                for ai, (ki, ci, selected) in enumerate(active):
                    gradients = torch.autograd.grad(
                        nll[selected].sum(), tuple(writes[site] for site in SITES),
                        retain_graph=ai + 1 < len(active), allow_unused=False,
                    )
                    backwards += 1
                    for mi, (site, gradient) in enumerate(zip(SITES, gradients)):
                        module = model.transformer.h[MODULES[mi]].mlp
                        delta = (products[site] - absent_products[site]).float()
                        reader = torch.matmul(gradient.float(), module.Down.weight.float())
                        term_response = -(reader * delta).sum((0, 1))
                        direct_delta = torch.matmul(delta, module.Down.weight.float().T)
                        direct_response = -(gradient.float() * direct_delta).sum()
                        mismatch = term_response.sum() - direct_response
                        contraction_numerator += float(mismatch.square())
                        contraction_denominator += float(direct_response.square())
                        sums[half, si, ki, mi, :, ci] += term_response.double().cpu()
                del logits, products, writes, nll
        del absent_products
    contraction_error = contraction_numerator / max(contraction_denominator, 1e-30)
    return sums, counts, reconstruction, contraction_error, backwards


def response_profiles(sums, counts):
    means = sums / counts[:, None, :, None, None, :].clamp_min(1)
    contrast = means[:, :, 0] - means[:, :, 1]
    centered = contrast - contrast.mean(-1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
    normalized = centered / norms.clamp_min(1e-30)
    return means, contrast, normalized, norms.squeeze(-1)


def row_cosine(left, right):
    return (left * right).sum(-1) / (
        torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    ).clamp_min(1e-30)


def eligible_terms(normalized, norms):
    # normalized: [half, source, site, term, circuit]
    output = []
    for mi in range(len(SITES)):
        nonzero = (norms[:, :, mi] > 1e-12).all((0, 1))
        half_ok = torch.stack([
            row_cosine(normalized[0, si, mi], normalized[1, si, mi]) >= .50
            for si in range(len(SOURCES))
        ]).all(0)
        source_left = F.normalize(normalized[:, 0, mi].mean(0), dim=-1)
        source_right = F.normalize(normalized[:, 1, mi].mean(0), dim=-1)
        source_ok = row_cosine(source_left, source_right) >= .50
        output.append(nonzero & half_ok & source_ok)
    return torch.stack(output)


def pair_graph(normalized, eligible, left, right, *, permutations=None, device="cuda"):
    # Four views = two halves x two sources. Similarity selection is average-view;
    # admission is the minimum of the four view cosines.
    left_views = normalized[:, :, left].reshape(4, HIDDEN, -1).float().to(device)
    right_views = normalized[:, :, right].reshape(4, HIDDEN, -1).float().to(device)
    if permutations is not None:
        right_views = torch.stack([
            right_views[view, :, permutations[view]].contiguous() for view in range(4)
        ])
    similarity = sum(left_views[view] @ right_views[view].T for view in range(4)) / 4
    left_ok = eligible[left].to(device)
    right_ok = eligible[right].to(device)
    similarity[~left_ok] = -torch.inf
    similarity[:, ~right_ok] = -torch.inf
    best_right = similarity.argmax(1)
    best_left = similarity.argmax(0)
    left_indices = torch.arange(HIDDEN, device=device)
    mutual = left_ok & right_ok[best_right] & (best_left[best_right] == left_indices)
    chosen_left = left_indices[mutual]
    chosen_right = best_right[mutual]
    if chosen_left.numel():
        view_cosines = torch.stack([
            (left_views[view, chosen_left] * right_views[view, chosen_right]).sum(-1)
            for view in range(4)
        ], dim=1)
        keep = view_cosines.min(1).values >= .70
        chosen_left, chosen_right = chosen_left[keep], chosen_right[keep]
        view_cosines = view_cosines[keep]
    else:
        view_cosines = torch.empty(0, 4, device=device)
    return {
        "left_indices": chosen_left.cpu(), "right_indices": chosen_right.cpu(),
        "view_cosines": view_cosines.cpu(),
    }


def _graph_summary(graph):
    cosines = graph["view_cosines"]
    return {
        "count": int(len(graph["left_indices"])),
        "left_indices": graph["left_indices"].tolist(),
        "right_indices": graph["right_indices"].tolist(),
        "minimum_view_cosine_min": float(cosines.min()) if cosines.numel() else None,
        "minimum_view_cosine_median": float(cosines.min(1).values.median())
        if cosines.numel() else None,
    }


def _jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(len(left | right), 1)


def aggregate_group_report(means, contrast, left, right, graph):
    output = {}
    for mi, indices in ((left, graph["left_indices"]), (right, graph["right_indices"])):
        member = means[:, :, 0, mi, indices].sum(2)
        control = means[:, :, 1, mi, indices].sum(2)
        selective = contrast[:, :, mi, indices].sum(2)
        source_cosine = float(row_cosine(selective[:, 0].mean(0), selective[:, 1].mean(0)))
        half_cosines = [float(row_cosine(selective[0, si], selective[1, si]))
                        for si in range(len(SOURCES))]
        norm_ratios = (torch.linalg.vector_norm(member, dim=-1)
                       / torch.linalg.vector_norm(control, dim=-1).clamp_min(1e-30))
        output[SITES[mi]] = {
            "term_count": int(len(indices)), "source_cosine": source_cosine,
            "half_cosines": half_cosines, "member_control_norm_ratios": norm_ratios.tolist(),
            "passes": bool(source_cosine >= .80 and min(half_cosines) >= .70
                           and float(norm_ratios.min()) >= 1.5),
        }
    return output


def analyze(sums, counts):
    means, contrast, normalized, norms = response_profiles(sums, counts)
    eligible = eligible_terms(normalized, norms)
    graphs = []
    for pi, (left, right) in enumerate(PAIRS):
        graph = pair_graph(normalized, eligible, left, right)
        graphs.append({"pair": PAIR_NAMES[pi], "pair_index": pi, "graph": graph})
    graphs.sort(key=lambda row: (-len(row["graph"]["left_indices"]), row["pair_index"]))
    winner = graphs[0]
    left, right = PAIRS[winner["pair_index"]]
    null_counts = []
    for seed in PERMUTATION_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        permutations = torch.stack([torch.randperm(len(counts[0, 0]), generator=generator)
                                    for _ in range(4)])
        null_counts.append(len(pair_graph(
            normalized, eligible, left, right, permutations=permutations,
        )["left_indices"]))
    null_q95 = int(torch.quantile(torch.tensor(null_counts, dtype=torch.float64), .95,
                                  interpolation="higher"))
    full_left = winner["graph"]["left_indices"].tolist()
    full_right = winner["graph"]["right_indices"].tolist()
    leave_one = []
    tags = json.loads(BATTERY.read_text())["by_tag"]
    discovery_tags = [tag for tag in sorted(tags) if _root(tag) in DISCOVERY_ROOTS]
    for root in DISCOVERY_ROOTS:
        keep = torch.tensor([_root(tag) != root for tag in discovery_tags])
        reduced = contrast[..., keep]
        reduced = reduced - reduced.mean(-1, keepdim=True)
        reduced_norms = torch.linalg.vector_norm(reduced, dim=-1)
        reduced = reduced / reduced_norms[..., None].clamp_min(1e-30)
        reduced_eligible = eligible_terms(reduced, reduced_norms)
        local = []
        for pi, (pair_left, pair_right) in enumerate(PAIRS):
            graph = pair_graph(reduced, reduced_eligible, pair_left, pair_right)
            local.append((len(graph["left_indices"]), pi, graph))
        local.sort(key=lambda row: (-row[0], row[1]))
        _, local_pi, local_graph = local[0]
        leave_one.append({
            "omitted_root": root, "winning_pair": PAIR_NAMES[local_pi],
            "count": len(local_graph["left_indices"]),
            "left_endpoint_jaccard": _jaccard(full_left, local_graph["left_indices"].tolist())
            if local_pi == winner["pair_index"] else 0.0,
            "right_endpoint_jaccard": _jaccard(full_right, local_graph["right_indices"].tolist())
            if local_pi == winner["pair_index"] else 0.0,
        })
    aggregate = aggregate_group_report(means, contrast, left, right, winner["graph"])
    eligible_counts = {site: int(eligible[mi].sum()) for mi, site in enumerate(SITES)}
    pred_b = sum(count >= 230 for count in eligible_counts.values()) >= 2
    pred_c = bool(len(full_left) >= 32 and len(full_left) >= 4 * null_q95)
    pred_d = bool(
        sum(row["winning_pair"] == winner["pair"] for row in leave_one) >= 5
        and sum(row["left_endpoint_jaccard"] >= .50
                and row["right_endpoint_jaccard"] >= .50 for row in leave_one) >= 5
    )
    pred_e = bool(full_left and all(row["passes"] for row in aggregate.values()))
    weak_graph = bool(len(full_left) >= 16 and len(full_left) >= 2 * null_q95)
    return {
        "eligible_counts": eligible_counts,
        "graphs": [{**{key: value for key, value in row.items() if key != "graph"},
                    **_graph_summary(row["graph"])} for row in graphs],
        "proposed_pair": winner["pair"] if full_left else None,
        "permuted_graph_counts": null_counts,
        "permuted_graph_count_95pct": null_q95,
        "leave_one_family": leave_one,
        "aggregate_group_report": aggregate,
        "pred_b_stable_terms": pred_b, "pred_c_response_graph": pred_c,
        "pred_d_leave_family_stable": pred_d, "pred_e_selective_aggregate": pred_e,
        "weak_graph_survives_control": weak_graph,
    }


def main():
    started = time.time()
    rows, _, circuit_masks, scale, tags, validation_tags, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 477, "model_loaded": False,
            "product_response_outcomes_opened": False,
            "validation_family_product_responses_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_backwards": metadata["expected_backwards"],
            "discovery_tags": len(tags), "reserved_validation_tags": len(validation_tags),
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung477 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    sums, counts, reconstruction, contraction_error, backwards = collect_responses(
        model, rows, circuit_masks, scale, tags, audit_totals, replay,
    )
    analysis = analyze(sums, counts)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    observed_backwards = sum(row.get("backwards", 0) for row in audit_totals.values())
    # Backwards are recorded directly because each source forward contains many VJPs.
    observed_backwards += backwards
    member_min = int(counts[:, 0].min())
    control_min = int(counts[:, 1].min())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and contraction_error <= 1e-8 and torch.isfinite(sums).all()
        and member_min >= 39 and control_min >= 439
        and forwards == EXPECTED_FORWARDS
        and observed_backwards == metadata["expected_backwards"]
    )
    strong_null = bool(
        not pred_a
        or sum(count >= 230 for count in analysis["eligible_counts"].values()) < 2
        or not analysis["weak_graph_survives_control"]
    )
    torch.save({
        "schema": "rung477_discovery_product_circuit_response_v1",
        "response_sums": sums, "response_counts": counts,
        "sources": list(SOURCES), "mask_types": list(MASK_TYPES),
        "sites": list(SITES), "discovery_tags": tags,
        "validation_tags_or_responses_included": False,
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 477,
        "claim_level": "discovery_only_product_response_graph",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "validation_family_product_responses_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "validation_tags_or_responses_included": False,
                   "raw_tokens_logits_or_hidden_states_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "term_vs_write_contraction_relative_squared_max": contraction_error,
        "response_counts": counts.tolist(), "audit_totals": audit_totals,
        "execution_price": {"outer_forwards": forwards, "backwards": observed_backwards,
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_saved": 0, "deployed_parameters_added": 0},
        'pred_a_instrument': pred_a,
        'pred_b_stable_terms': analysis["pred_b_stable_terms"],
        'pred_c_response_graph': analysis["pred_c_response_graph"],
        'pred_d_leave_family_stable': analysis["pred_d_leave_family_stable"],
        'pred_e_selective_aggregate': analysis["pred_e_selective_aggregate"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": ("heldout_family_exact_group_removal" if pred_a and all(
            analysis[key] for key in ("pred_c_response_graph", "pred_d_leave_family_stable",
                                      "pred_e_selective_aggregate"))
            else "sparse_mixed_product_response_directions"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 477,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "contraction_error": contraction_error, "forwards": forwards,
                       "backwards": observed_backwards},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
