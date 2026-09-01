"""RUNG 418 -- EXACT ATTENTION0 CROSS-HEAD QK SHARED-HALF SCREEN.

The layer0 fold gives exact unit-RMS vocabulary factor tables q[t,h,b] and
k[t,h,b] for nine heads and two multiplicative score branches.  Existing work
already rejects one global token partition and shows the two branches inside a
head are usually complementary.  This screen asks the distinct question:
whether a complete or >=16-dimensional token-function subspace is shared by
one QK branch across >=3 heads while the companion branches stay different.

FIT/SELECT are token-id mod5 splits over all 50,257 real tokens.  Candidate
relations use gauge-invariant query/key column-space principal angles on FIT.
Two 128x128 maps are fitted source->target and scored on unseen token IDs via
the exact rotary score family.  Independent token-row permutations and Haar
subspaces are controls.  This is a folded-function screen, not compression,
mechanism identification, or adoption.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_cross_head_qk_shared_half_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
V = 50_257
N_HEAD = 9
HD = 128
D = 1152
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128)
ENTRIES = tuple((head, branch) for head in range(N_HEAD) for branch in (1, 2))
MAX_EVAL_RELATIONS = 8


def _entry_name(entry):
    return f"h{entry[0]}b{entry[1]}"


def _factor(factors, entry, side):
    return factors[entry[1]][0 if side == "q" else 1][:V, entry[0]].float()


def _basis(value, mask, *, centered):
    x = value[mask]
    if centered:
        x = x - x.mean(0, keepdim=True)
    gram = x.double().T @ x.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    floor = eigenvalues.max().clamp_min(1e-30) * 1e-12
    inverse = eigenvectors @ torch.diag(eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
    result = x @ inverse.float()
    orth_error = float((result.T @ result - torch.eye(HD, device=result.device)).abs().max())
    return result, orth_error


def _principal(left, right):
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    squared = singular.square()
    return {
        "projector_overlap": float(squared.mean()),
        "shared_dimensions_cos2_ge_0_50": int((squared >= .50).sum()),
        "top_principal_cos2": squared[:16].cpu().tolist(),
    }


def _percentile(values, q):
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q))


def _r2(target, prediction, baseline):
    sse = float((target.double() - prediction.double()).square().sum())
    sst = float((target.double() - baseline.double()).square().sum())
    return 1.0 - sse / max(sst, 1e-30)


def _pearson(left, right):
    a = left.double().flatten()
    b = right.double().flatten()
    a -= a.mean()
    b -= b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b) / denominator) if float(denominator) else 0.0


def _fit_map(source, target, train_mask, *, permutation=None):
    xs = source[train_mask]
    yt = target[train_mask]
    if permutation is not None:
        xs = xs[permutation]
    source_mean = xs.mean(0)
    target_mean = yt.mean(0)
    xc = xs - source_mean
    yc = yt - target_mean
    gram = xc.double().T @ xc.double()
    ridge = float(torch.trace(gram)) / HD * 1e-8
    mapping = torch.linalg.solve(
        gram + ridge * torch.eye(HD, device=gram.device, dtype=torch.float64),
        xc.double().T @ yc.double()).float()
    return {"source_mean": source_mean, "target_mean": target_mean,
            "mapping": mapping, "ridge": ridge}


def _apply_map(source, model):
    return model["target_mean"] + (source - model["source_mean"]) @ model["mapping"]


def _scores(q, k, query_ids, key_ids, offsets, rope_tables, apply_rot):
    cos, sin = rope_tables(max(OFFSETS) + 1, HD, q.device, q.dtype, "bf16")
    qrows = q[query_ids]
    krows = k[key_ids]
    columns = []
    for offset in offsets:
        qrot = apply_rot(qrows, cos[offset], sin[offset])
        columns.append((qrot * krows).sum(-1) / HD)
    return torch.stack(columns, 1)


def _relation_score(pair):
    return min(pair["q_centered"]["projector_overlap"],
               pair["k_centered"]["projector_overlap"])


def _components(edges):
    adjacency = {entry: set() for entry in ENTRIES}
    for edge in edges:
        left, right = tuple(edge["left"]), tuple(edge["right"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen, result = set(), []
    for root in ENTRIES:
        if root in seen or not adjacency[root]:
            continue
        stack, component = [root], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component
        result.append(component)
    return result


def _spanning_tree(component, edges):
    parent = {node: node for node in component}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    chosen = []
    for edge in sorted(edges, key=_relation_score, reverse=True):
        left, right = tuple(edge["left"]), tuple(edge["right"])
        if left not in component or right not in component:
            continue
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b
            chosen.append(edge)
    return chosen


@torch.no_grad()
def _fold_gate(model, factors, rows, scores_from_factors, reference_forward):
    tokens = rows[:1, :-1].to(model.transformer.wte.weight.device)
    captured = {}

    def capture(layer, score1, score2):
        if layer == 0:
            captured[1] = score1.detach()
            captured[2] = score2.detach()
        return score1, score2

    reference_forward(model, tokens, table_dtype="bf16", score_patch=capture)
    errors = {}
    for branch in (1, 2):
        folded = scores_from_factors(*factors[branch], tokens, HD, table_dtype="bf16")
        errors[str(branch)] = float((folded - captured[branch]).abs().max())
    return errors


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ENTRIES) == 18 and len(OFFSETS) == 8
        assert V == 50_257 and N_HEAD * HD == D
        print("ATTENTION0 CROSS-HEAD QK HALF | dry run: exact fold, 18 entries, controls frozen")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    from tier2_model import load_elriggs, rope_tables, apply_rot, reference_forward
    from tier2_folding import branch_factors, scores_from_factors
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    device = torch.device("cuda")
    # Keep the model/fold gate in float64 as in the established 1e-15 identity;
    # use float32 copies only for the large subspace screen.
    model, config = load_elriggs("bilin18", device=device, dtype=torch.float64)
    gate_factors = {branch: branch_factors(model, branch, dtype=torch.float64)
                    for branch in (1, 2)}
    factors = {branch: branch_factors(model, branch, dtype=torch.float32)
               for branch in (1, 2)}
    token_ids = torch.arange(V, device=device)
    train_mask = token_ids.remainder(5) != 4
    select_mask = ~train_mask
    train_indices = token_ids[train_mask]
    select_indices = token_ids[select_mask]
    receipt = json.loads(ROWS_RECEIPT.read_text())
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fold_errors = _fold_gate(
        model, gate_factors, select_rows, scores_from_factors, reference_forward)
    del gate_factors

    unit_errors = {}
    for entry in ENTRIES:
        for side in ("q", "k"):
            value = _factor(factors, entry, side)
            unit_errors[f"{_entry_name(entry)}_{side}"] = float(
                (value.square().mean(-1) - 1).abs().max())

    bases = {kind: {side: {} for side in ("q", "k")}
             for kind in ("raw", "centered", "haar")}
    permutations = {side: {} for side in ("q", "k")}
    orth_errors = []
    for entry_index, entry in enumerate(ENTRIES):
        for side_index, side in enumerate(("q", "k")):
            value = _factor(factors, entry, side)
            for centered, kind in ((False, "raw"), (True, "centered")):
                bases[kind][side][entry], error = _basis(value, train_mask, centered=centered)
                orth_errors.append(error)
            generator = torch.Generator(device="cpu").manual_seed(
                418_000 + entry_index * 10 + side_index)
            permutations[side][entry] = torch.randperm(
                len(train_indices), generator=generator, device="cpu").to(device)
            random = torch.randn(
                len(train_indices), HD, generator=generator, device="cpu").to(device)
            haar, error = _basis(random, torch.ones(len(random), dtype=torch.bool, device=device),
                                 centered=False)
            bases["haar"][side][entry] = haar
            orth_errors.append(error)

    pairs = []
    permutation_overlaps = {"q": [], "k": []}
    haar_overlaps = {"q": [], "k": []}
    for left_index, left in enumerate(ENTRIES):
        for right in ENTRIES[left_index + 1:]:
            if left[0] == right[0]:
                continue
            row = {"left": list(left), "right": list(right)}
            for side in ("q", "k"):
                row[f"{side}_raw"] = _principal(
                    bases["raw"][side][left], bases["raw"][side][right])
                row[f"{side}_centered"] = _principal(
                    bases["centered"][side][left], bases["centered"][side][right])
                permuted = _principal(
                    bases["centered"][side][left],
                    bases["centered"][side][right][permutations[side][right]])
                haar = _principal(bases["haar"][side][left], bases["haar"][side][right])
                row[f"{side}_permuted_overlap"] = permuted["projector_overlap"]
                row[f"{side}_haar_overlap"] = haar["projector_overlap"]
                permutation_overlaps[side].append(permuted["projector_overlap"])
                haar_overlaps[side].append(haar["projector_overlap"])
            pairs.append(row)

    null_threshold = {
        side: {"permutation_p99": _percentile(permutation_overlaps[side], .99),
               "haar_p99": _percentile(haar_overlaps[side], .99)}
        for side in ("q", "k")
    }
    eligible = []
    for pair in pairs:
        pair["eligible_edge"] = all(
            pair[f"{side}_centered"]["shared_dimensions_cos2_ge_0_50"] >= 16
            and pair[f"{side}_centered"]["projector_overlap"] >= .15
            and pair[f"{side}_centered"]["projector_overlap"]
                >= null_threshold[side]["permutation_p99"] + .08
            for side in ("q", "k"))
        if pair["eligible_edge"]:
            eligible.append(pair)

    components = _components(eligible)
    components.sort(key=lambda group: (len({entry[0] for entry in group}), len(group)), reverse=True)
    selected_component = components[0] if components else set()
    spanning = _spanning_tree(selected_component, eligible) if selected_component else []
    # Evaluate preregistered tree if it exists; otherwise evaluate the strongest pairs to diagnose the null.
    diagnostic_pairs = spanning if spanning else sorted(pairs, key=_relation_score, reverse=True)[:MAX_EVAL_RELATIONS]
    diagnostic_pairs = diagnostic_pairs[:MAX_EVAL_RELATIONS]

    # Frozen unseen-token pairs for exact score transfer.
    generator = torch.Generator(device="cpu").manual_seed(418_777)
    pair_count = min(65_536, len(select_indices) * 4)
    q_pick = torch.randint(len(select_indices), (pair_count,), generator=generator)
    k_pick = torch.randint(len(select_indices), (pair_count,), generator=generator)
    query_ids = select_indices[q_pick.to(device)]
    key_ids = select_indices[k_pick.to(device)]

    # Natural examples for the companion/product check.
    natural_q, natural_k, natural_delta = [], [], []
    for row in select_rows[:, :-1]:
        for query_position in range(64, row.shape[0]):
            for offset in OFFSETS:
                if query_position >= offset:
                    natural_q.append(int(row[query_position]))
                    natural_k.append(int(row[query_position - offset]))
                    natural_delta.append(offset)
    natural_q = torch.tensor(natural_q, device=device)
    natural_k = torch.tensor(natural_k, device=device)
    natural_delta = torch.tensor(natural_delta, device=device)

    evaluations = []
    for pair_index, pair in enumerate(diagnostic_pairs):
        left, right = tuple(pair["left"]), tuple(pair["right"])
        # Choose source direction on FIT by factor reconstruction.
        directions = []
        for source, target in ((left, right), (right, left)):
            models = {}
            fit_r2 = []
            for side in ("q", "k"):
                source_value = _factor(factors, source, side)
                target_value = _factor(factors, target, side)
                fitted = _fit_map(source_value, target_value, train_mask)
                prediction = _apply_map(source_value[train_mask], fitted)
                fit_r2.append(_r2(target_value[train_mask], prediction, fitted["target_mean"]))
                models[side] = fitted
            directions.append((sum(fit_r2), source, target, models, fit_r2))
        _, source, target, models, fit_r2 = max(directions, key=lambda item: item[0])

        control_models = {}
        select_permutations = {}
        for side_index, side in enumerate(("q", "k")):
            gen = torch.Generator(device="cpu").manual_seed(418_900 + pair_index * 10 + side_index)
            train_perm = torch.randperm(len(train_indices), generator=gen, device="cpu").to(device)
            select_permutations[side] = torch.randperm(
                len(select_indices), generator=gen, device="cpu").to(device)
            control_models[side] = _fit_map(
                _factor(factors, source, side), _factor(factors, target, side),
                train_mask, permutation=train_perm)

        target_true, target_predicted, target_control = {}, {}, {}
        factor_r2, control_factor_r2 = {}, {}
        for side in ("q", "k"):
            source_value = _factor(factors, source, side)
            target_value = _factor(factors, target, side)
            target_true[side] = target_value
            target_predicted[side] = _apply_map(source_value, models[side])
            permuted_select_source = source_value[select_mask][select_permutations[side]]
            control_select = _apply_map(permuted_select_source, control_models[side])
            factor_r2[side] = _r2(
                target_value[select_mask], target_predicted[side][select_mask],
                models[side]["target_mean"])
            control_factor_r2[side] = _r2(
                target_value[select_mask], control_select,
                control_models[side]["target_mean"])

        # Normalize mapped rows to the exact factor gauge before scoring.
        pred_q = F.rms_norm(target_predicted["q"], (HD,))
        pred_k = F.rms_norm(target_predicted["k"], (HD,))
        true_scores = _scores(target_true["q"], target_true["k"], query_ids, key_ids,
                              OFFSETS, rope_tables, apply_rot)
        predicted_scores = _scores(pred_q, pred_k, query_ids, key_ids,
                                   OFFSETS, rope_tables, apply_rot)
        # Controls are available only on SELECT rows; construct a compact SELECT-index table.
        control_q_select = _apply_map(
            _factor(factors, source, "q")[select_mask][select_permutations["q"]],
            control_models["q"])
        control_k_select = _apply_map(
            _factor(factors, source, "k")[select_mask][select_permutations["k"]],
            control_models["k"])
        lookup = torch.full((V,), -1, dtype=torch.long, device=device)
        lookup[select_indices] = torch.arange(len(select_indices), device=device)
        control_scores = _scores(
            F.rms_norm(control_q_select, (HD,)), F.rms_norm(control_k_select, (HD,)),
            lookup[query_ids], lookup[key_ids], OFFSETS, rope_tables, apply_rot)
        score_r2 = _r2(true_scores, predicted_scores, true_scores.mean())
        control_score_r2 = _r2(true_scores, control_scores, true_scores.mean())
        offset_half_r2 = {
            "powers_even": _r2(true_scores[:, ::2], predicted_scores[:, ::2],
                               true_scores[:, ::2].mean()),
            "powers_odd": _r2(true_scores[:, 1::2], predicted_scores[:, 1::2],
                              true_scores[:, 1::2].mean()),
        }

        source_companion = (source[0], 3 - source[1])
        target_companion = (target[0], 3 - target[1])
        # Fit the paired companion with the same direction and split.
        companion_models = {}
        for side in ("q", "k"):
            companion_models[side] = _fit_map(
                _factor(factors, source_companion, side),
                _factor(factors, target_companion, side), train_mask)
        comp_q = F.rms_norm(_apply_map(
            _factor(factors, source_companion, "q"), companion_models["q"]), (HD,))
        comp_k = F.rms_norm(_apply_map(
            _factor(factors, source_companion, "k"), companion_models["k"]), (HD,))
        companion_true = _scores(
            _factor(factors, target_companion, "q"),
            _factor(factors, target_companion, "k"), query_ids, key_ids,
            OFFSETS, rope_tables, apply_rot)
        companion_pred = _scores(comp_q, comp_k, query_ids, key_ids,
                                 OFFSETS, rope_tables, apply_rot)
        companion_r2 = _r2(companion_true, companion_pred, companion_true.mean())

        # Natural raw branch/product correlations, computed offset by offset.
        def natural(entry):
            q = _factor(factors, entry, "q")[natural_q]
            k = _factor(factors, entry, "k")[natural_k]
            cos, sin = rope_tables(max(OFFSETS) + 1, HD, device, q.dtype, "bf16")
            qrot = apply_rot(q, cos[natural_delta], sin[natural_delta])
            return (qrot * k).sum(-1) / HD

        source_selected_natural = natural(source)
        target_selected_natural = natural(target)
        source_companion_natural = natural(source_companion)
        target_companion_natural = natural(target_companion)
        branch_correlation = abs(_pearson(source_selected_natural, target_selected_natural))
        product_correlation = abs(_pearson(
            source_selected_natural * source_companion_natural,
            target_selected_natural * target_companion_natural))

        evaluations.append({
            "pair": {"source": list(source), "target": list(target),
                     "source_name": _entry_name(source), "target_name": _entry_name(target)},
            "eligible_edge": pair["eligible_edge"],
            "fit_factor_r2": dict(zip(("q", "k"), fit_r2)),
            "select_factor_r2": factor_r2,
            "control_select_factor_r2": control_factor_r2,
            "select_branch_score_r2": score_r2,
            "control_select_branch_score_r2": control_score_r2,
            "offset_half_score_r2": offset_half_r2,
            "companion_branch_score_r2": companion_r2,
            "natural_raw_selected_branch_correlation": branch_correlation,
            "natural_raw_complete_product_correlation": product_correlation,
            "selected_minus_product_correlation": branch_correlation - product_correlation,
        })

    selected_heads = {entry[0] for entry in selected_component}
    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and max(unit_errors.values()) <= 1e-6
        and int(train_mask.sum() + select_mask.sum()) == V
        and not bool((train_mask & select_mask).any())
        and len(OFFSETS) == 8 and len(natural_q) > 0
        and max(orth_errors) <= 2e-4)
    pred_b = len(selected_heads) >= 3 and len(spanning) > 0
    evaluated_tree = [evaluation for evaluation in evaluations if evaluation["eligible_edge"]]
    pred_c = bool(evaluated_tree) and all(
        min(value["select_factor_r2"].values()) >= .50
        and value["select_branch_score_r2"] >= .60
        and value["select_branch_score_r2"] - value["control_select_branch_score_r2"] >= .40
        and abs(value["offset_half_score_r2"]["powers_even"]
                - value["offset_half_score_r2"]["powers_odd"]) <= .15
        for value in evaluated_tree)
    pred_d = bool(evaluated_tree) and all(
        (value["companion_branch_score_r2"] <= .35
         or value["select_branch_score_r2"] - value["companion_branch_score_r2"] >= .25)
        and value["selected_minus_product_correlation"] >= .20
        for value in evaluated_tree)

    no_pair_even8 = all(
        pair["q_centered"]["shared_dimensions_cos2_ge_0_50"] < 8
        or pair["k_centered"]["shared_dimensions_cos2_ge_0_50"] < 8
        for pair in pairs)
    best_score_r2 = max((value["select_branch_score_r2"] for value in evaluations), default=-1.0)
    best_margin = max((value["select_branch_score_r2"]
                       - value["control_select_branch_score_r2"] for value in evaluations),
                      default=-1.0)
    all_companions_equal = bool(evaluations) and all(
        abs(value["select_branch_score_r2"] - value["companion_branch_score_r2"]) <= .10
        for value in evaluations)
    strong_null = (
        not pred_a or no_pair_even8 or best_score_r2 <= .25
        or best_margin <= .10 or all_companions_equal)

    result = {
        "status": "attention0_cross_head_qk_shared_half_complete",
        "rung": 418,
        "claim_level": "gauge_invariant_folded_function_screen_not_identification_or_compression",
        "definition": {
            "entry": "one architectural head and one of its two multiplicative QK score branches",
            "shared_dimension": "squared principal cosine >=.50 between centered vocabulary-function column subspaces",
            "projector_overlap": "mean squared principal cosine over 128 factor dimensions",
            "score": "exact unit-RMS q^T R_delta k /128 relative-offset branch score",
            "gauge_scope": "token-function subspaces and complete scores, never private factor coordinates",
        },
        "tokens": {"real": V, "FIT_mod_not4": int(train_mask.sum()),
                   "SELECT_mod4": int(select_mask.sum()), "FINAL_opened": 0},
        "offsets": list(OFFSETS),
        "documents": {"natural_SELECT": len(select_rows), "FINAL_opened": 0},
        "exactness": {"fold_max_abs_by_branch": fold_errors,
                      "factor_unit_rms_max_abs": max(unit_errors.values()),
                      "subspace_orthogonality_max_abs": max(orth_errors)},
        "null_thresholds": null_threshold,
        "pairwise": pairs,
        "eligible_edge_count": len(eligible),
        "selected_component_entries": [list(entry) for entry in sorted(selected_component)],
        "selected_component_distinct_heads": sorted(selected_heads),
        "selected_spanning_tree": [
            {"left": edge["left"], "right": edge["right"], "score": _relation_score(edge)}
            for edge in spanning],
        "evaluations": evaluations,
        "literal_price": {
            "maps_per_relation_values": 2 * HD * HD,
            "native_target_qk_projection_values": 2 * HD * D,
            "stored_deployed_values_this_screen": 0,
            "all_shared_source_costs_counted_for_adoption": False,
        },
        'pred_a_exact_auditable_fold': bool(pred_a),
        'pred_b_multihead_shared_half_subspace': bool(pred_b),
        'pred_c_unseen_token_and_score_transport': bool(pred_c),
        'pred_d_shared_half_different_companion': bool(pred_d),
        "null_no_cross_head_qk_half_vocabulary": bool(strong_null),
        "next_step": (
            "finite_natural_score_intervention_then_physical_ce"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "attention1_finite_response_producer_groups" if pred_a
            else "instrument_repair_only"),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "config": config,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "exactness": result["exactness"],
        "null_thresholds": null_threshold, "eligible_edge_count": len(eligible),
        "selected_heads": sorted(selected_heads), "evaluations": evaluations,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c, "pred_d": pred_d,
        "strong_null": strong_null, "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 CROSS-HEAD QK HALF DONE", flush=True)


if __name__ == "__main__":
    main()
