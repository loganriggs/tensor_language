"""RUNG 408 -- HELDOUT FOUR-STATE PREFIX-TOKEN ROUTER FEASIBILITY.

Use immutable rung407 I-active/Fisher per-position losses.  Fit one max-four-
leaf regression tree on prefix-observable token features from documents 0:192
and evaluate on documents 192:384 against p768 and the future-label oracle.
This is an off-policy feasibility screen, not a physical routed forward.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
import tiktoken


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
OUT = BQ / "mlp0_prefix_token_state_router_feasibility_results.json"
PARENT = BQ / "mlp0_p448_router_oracle_ceiling_results.json"
LOSS_ARTIFACT = BQ / "mlp0_p448_router_oracle_losses.pt"
CONFIRM_RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
CONFIRM_CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1/eval_384_source_documents.pt"
LOSS_TENSOR_SHA = "e6d92614ad4fbe5b6e63aa2939e7df6ecb197281c114aae9696baf3bc68ab082"
SOURCE_DOCUMENTS = 384
TRAIN_DOCUMENTS = 192
EVAL_DOCUMENTS = 192
WAVE_DOCUMENTS = 96
SCORING_START = 64
SCORING_STOP = 256
POSITIONS = SCORING_STOP - SCORING_START
REAL_V = 50257
TREE_SEED = 408
MAX_LEAVES = 4
MIN_LEAF = 2048
TWO_P448_VALUES = 14_599_296
P768_VALUES = 13_272_192
EXPECTED_NAMES = (
    "native", "covariance_p448", "T_active_p448", "I_active_p448",
    "TI_active_p448", "Fisher_p448", "covariance_p640", "covariance_p768")


def _tensor_sha256(value: torch.Tensor):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _morphology_table():
    encoding = tiktoken.get_encoding("gpt2")
    result = np.empty(REAL_V, dtype=np.int64)
    for token in range(REAL_V):
        value = encoding.decode_single_token_bytes(token)
        has_digit = any(48 <= byte <= 57 for byte in value)
        if has_digit:
            result[token] = 2
        elif len(value) >= 2 and value[0] == 32 and (
                65 <= value[1] <= 90 or 97 <= value[1] <= 122):
            result[token] = 0
        elif len(value) >= 1 and (65 <= value[0] <= 90 or 97 <= value[0] <= 122):
            result[token] = 1
        else:
            result[token] = 3
    return result


def _feature_at(row, absolute_position, morphology, log_frequency, max_log_frequency):
    current = int(row[absolute_position])
    previous = int(row[absolute_position - 1])
    prior = row[:absolute_position]
    matches = torch.nonzero(prior == current, as_tuple=False).flatten()
    repeated = len(matches) > 0
    distance = absolute_position - int(matches[-1]) if repeated else 257
    current_onehot = np.eye(4, dtype=np.float32)[morphology[current]]
    previous_onehot = np.eye(4, dtype=np.float32)[morphology[previous]]
    return np.concatenate((np.asarray([
        absolute_position / 255.0,
        float(repeated),
        math.log1p(distance) / math.log(258.0),
        log_frequency[current] / max_log_frequency,
    ], dtype=np.float32), current_onehot, previous_onehot))


def _features(rows, morphology):
    training_tokens = rows[:TRAIN_DOCUMENTS, :256].reshape(-1)
    frequency = torch.bincount(training_tokens, minlength=REAL_V).numpy()
    log_frequency = np.log1p(frequency).astype(np.float32)
    max_log_frequency = float(log_frequency.max())
    result = np.empty((SOURCE_DOCUMENTS, POSITIONS, 12), dtype=np.float32)
    repeat_state = np.empty((SOURCE_DOCUMENTS, POSITIONS), dtype=np.int64)
    for document in range(SOURCE_DOCUMENTS):
        row = rows[document]
        for offset, absolute_position in enumerate(range(SCORING_START, SCORING_STOP)):
            feature = _feature_at(
                row, absolute_position, morphology, log_frequency, max_log_frequency)
            result[document, offset] = feature
            repeated = bool(feature[1])
            if not repeated:
                repeat_state[document, offset] = 0
            else:
                normalized = float(feature[2])
                distance = round(math.exp(normalized * math.log(258.0)) - 1)
                repeat_state[document, offset] = 1 if distance <= 8 else 2 if distance <= 32 else 3
    return result, repeat_state, log_frequency, max_log_frequency


def _state_selector(train_advantage, train_states, eval_states):
    choices = {}
    for state in range(4):
        mask = train_states == state
        choices[state] = bool(float(train_advantage[mask].mean()) >= 0) if mask.any() else True
    return np.asarray([choices[int(state)] for state in eval_states], dtype=bool), choices


def _summary(selected_loss, native_loss, oracle_loss, constant_i_loss, p768_loss,
             selection_i, eval_documents=EVAL_DOCUMENTS):
    selected = selected_loss.reshape(eval_documents, POSITIONS)
    native = native_loss.reshape(eval_documents, POSITIONS)
    damage = float(selected.mean() - native.mean())
    constant_damage = float(constant_i_loss.mean() - native_loss.mean())
    oracle_damage = float(oracle_loss.mean() - native_loss.mean())
    available = constant_damage - oracle_damage
    wave_damage = []
    for wave in range(2):
        start, end = wave * WAVE_DOCUMENTS, (wave + 1) * WAVE_DOCUMENTS
        wave_damage.append(float(selected[start:end].mean() - native[start:end].mean()))
    return {
        "damage": damage,
        "gain_over_constant_I": constant_damage - damage,
        "oracle_gain_recovery_fraction": (
            (constant_damage - damage) / available if available > 0 else 0.0),
        "gain_over_p768": float(p768_loss.mean() - selected_loss.mean()),
        "expert_use_fraction": {
            "I_active_p448": float(selection_i.mean()),
            "Fisher_p448": float((~selection_i).mean()),
        },
        "wave_damage": wave_damage,
    }


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (PARENT, LOSS_ARTIFACT, CONFIRM_RECEIPT, CONFIRM_CACHE))
        assert TRAIN_DOCUMENTS + EVAL_DOCUMENTS == SOURCE_DOCUMENTS
        assert POSITIONS == 192 and MAX_LEAVES == 4 and MIN_LEAF == 2048
        assert TWO_P448_VALUES == 14_599_296 and P768_VALUES == 13_272_192
        print("MLP0 PREFIX TOKEN ROUTER | dry run: artifact, split, features, tree, bars valid")
        return

    started = time.time()
    parent = json.loads(PARENT.read_text())
    if not parent["null_router_oracle_has_no_adoption_headroom"]:
        raise RuntimeError("rung407 null does not license position-state feasibility")
    artifact = torch.load(LOSS_ARTIFACT, map_location="cpu", weights_only=True)
    losses = artifact["losses"].float()
    names = tuple(artifact["program_names"])
    receipt = json.loads(CONFIRM_RECEIPT.read_text())
    all_rows = torch.load(CONFIRM_CACHE, map_location="cpu")
    records = receipt["document_provenance"]["sets"]["eval"]
    chunk0 = [index for index, record in enumerate(records) if int(record["chunk_id"]) == 0]
    ordinals = [int(records[index]["source_document_ordinal"]) for index in chunk0]
    rows = all_rows[chunk0, :257].long().contiguous()
    authority_exact = (
        names == EXPECTED_NAMES
        and tuple(losses.shape) == (8, SOURCE_DOCUMENTS, POSITIONS)
        and _tensor_sha256(losses) == LOSS_TENSOR_SHA
        and tuple(rows.shape) == (SOURCE_DOCUMENTS, 257)
        and ordinals == list(range(SOURCE_DOCUMENTS))
        and all(receipt["disjointness_gates"].values()))

    morphology = _morphology_table()
    features, repeat_state, log_frequency, max_log_frequency = _features(rows, morphology)
    finite_features = bool(np.isfinite(features).all())
    train_x = features[:TRAIN_DOCUMENTS].reshape(-1, 12)
    eval_x = features[TRAIN_DOCUMENTS:].reshape(-1, 12)
    index = {name: position for position, name in enumerate(names)}
    train_i = losses[index["I_active_p448"], :TRAIN_DOCUMENTS].numpy().reshape(-1)
    train_f = losses[index["Fisher_p448"], :TRAIN_DOCUMENTS].numpy().reshape(-1)
    eval_native = losses[index["native"], TRAIN_DOCUMENTS:].numpy().reshape(-1)
    eval_i = losses[index["I_active_p448"], TRAIN_DOCUMENTS:].numpy().reshape(-1)
    eval_f = losses[index["Fisher_p448"], TRAIN_DOCUMENTS:].numpy().reshape(-1)
    eval_p768 = losses[index["covariance_p768"], TRAIN_DOCUMENTS:].numpy().reshape(-1)
    train_advantage = train_f - train_i
    eval_advantage = eval_f - eval_i
    eval_oracle = np.minimum(eval_i, eval_f)
    oracle_choice_i = eval_i <= eval_f

    tree = DecisionTreeRegressor(
        random_state=TREE_SEED, max_leaf_nodes=MAX_LEAVES,
        min_samples_leaf=MIN_LEAF)
    tree.fit(train_x, train_advantage)
    predicted_advantage = tree.predict(eval_x)
    tree_choice_i = predicted_advantage >= 0
    tree_loss = np.where(tree_choice_i, eval_i, eval_f)
    leaf_train = tree.apply(train_x)
    leaves, leaf_counts = np.unique(leaf_train, return_counts=True)
    tree_summary = _summary(
        tree_loss, eval_native, eval_oracle, eval_i, eval_p768, tree_choice_i)
    tree_summary.update({
        "selection_accuracy_vs_future_oracle": float((tree_choice_i == oracle_choice_i).mean()),
        "mean_regret_to_future_oracle": float((tree_loss - eval_oracle).mean()),
        "leaf_count": int(tree.get_n_leaves()),
        "node_count": int(tree.tree_.node_count),
        "leaf_train_support": {str(int(leaf)): int(count) for leaf, count in zip(leaves, leaf_counts)},
        "feature_indices": [int(value) for value in tree.tree_.feature.tolist()],
        "thresholds": [float(value) for value in tree.tree_.threshold.tolist()],
        "children_left": [int(value) for value in tree.tree_.children_left.tolist()],
        "children_right": [int(value) for value in tree.tree_.children_right.tolist()],
        "leaf_predictions": [float(value) for value in tree.tree_.value.reshape(-1).tolist()],
    })

    train_positions = np.tile(np.arange(POSITIONS), TRAIN_DOCUMENTS)
    eval_positions = np.tile(np.arange(POSITIONS), EVAL_DOCUMENTS)
    position_train_state = train_positions // 48
    position_eval_state = eval_positions // 48
    morphology_train_state = np.argmax(train_x[:, 4:8], axis=1)
    morphology_eval_state = np.argmax(eval_x[:, 4:8], axis=1)
    repeat_train_state = repeat_state[:TRAIN_DOCUMENTS].reshape(-1)
    repeat_eval_state = repeat_state[TRAIN_DOCUMENTS:].reshape(-1)
    controls = {}
    for name, train_state, eval_state in (
            ("position_quartile", position_train_state, position_eval_state),
            ("current_morphology4", morphology_train_state, morphology_eval_state),
            ("repeat_distance4", repeat_train_state, repeat_eval_state)):
        choice_i, choices = _state_selector(train_advantage, train_state, eval_state)
        selected = np.where(choice_i, eval_i, eval_f)
        controls[name] = {
            **_summary(selected, eval_native, eval_oracle, eval_i, eval_p768, choice_i),
            "state_choice_I": {str(key): value for key, value in choices.items()},
        }
    controls["constant_I"] = _summary(
        eval_i, eval_native, eval_oracle, eval_i, eval_p768,
        np.ones_like(eval_i, dtype=bool))

    # For sampled evaluation positions, change every strictly future token and
    # verify the feature at the current position is unchanged with the frozen
    # training-frequency table.
    future_perturb_max_abs = 0.0
    for document in (192, 223, 287, 383):
        for absolute_position in (64, 128, 192, 255):
            original = _feature_at(
                rows[document], absolute_position, morphology,
                log_frequency, max_log_frequency)
            changed = rows[document].clone()
            if absolute_position + 1 < len(changed):
                changed[absolute_position + 1:] = (
                    changed[absolute_position + 1:] + 7919) % REAL_V
            perturbed = _feature_at(
                changed, absolute_position, morphology,
                log_frequency, max_log_frequency)
            future_perturb_max_abs = max(
                future_perturb_max_abs, float(np.max(np.abs(original - perturbed))))

    native_eval_ce = float(eval_native.mean())
    i_damage = float(eval_i.mean() - native_eval_ce)
    fisher_damage = float(eval_f.mean() - native_eval_ce)
    p768_damage = float(eval_p768.mean() - native_eval_ce)
    oracle_damage = float(eval_oracle.mean() - native_eval_ce)
    oracle_gain = i_damage - oracle_damage
    feature_used = set(int(value) for value in tree.tree_.feature if value >= 0)
    internal_nodes = int(tree.tree_.node_count - tree.get_n_leaves())
    frequency_table_values = REAL_V if 3 in feature_used else 0
    tree_values = int(4 * internal_nodes + tree.get_n_leaves())
    router_price = int(TWO_P448_VALUES + frequency_table_values + tree_values)
    tree_live = (
        0 < tree_summary["expert_use_fraction"]["I_active_p448"] < 1
        and not np.array_equal(tree_loss, eval_i))
    pred_a = (
        authority_exact and finite_features and bool(np.isfinite(losses.numpy()).all())
        and future_perturb_max_abs == 0.0
        and tree.get_n_leaves() <= MAX_LEAVES
        and min(leaf_counts) >= MIN_LEAF and tree_live)
    pred_b = oracle_damage <= p768_damage - .0002
    pred_c = (
        tree_summary["oracle_gain_recovery_fraction"] >= .25
        and tree_summary["gain_over_constant_I"] >= .001
        and tree_summary["damage"] <= p768_damage - .0002)
    pred_d = (
        all(tree_summary["wave_damage"][wave]
            < float(eval_p768.reshape(EVAL_DOCUMENTS, POSITIONS)[
                wave * WAVE_DOCUMENTS:(wave + 1) * WAVE_DOCUMENTS].mean()
                - eval_native.reshape(EVAL_DOCUMENTS, POSITIONS)[
                    wave * WAVE_DOCUMENTS:(wave + 1) * WAVE_DOCUMENTS].mean())
            for wave in range(2))
        and min(tree_summary["expert_use_fraction"].values()) >= .10
        and min(leaf_counts) >= MIN_LEAF)
    strong_null = (
        not pred_a or not pred_b
        or tree_summary["gain_over_constant_I"] < .0002
        or tree_summary["damage"] >= p768_damage
        or min(tree_summary["expert_use_fraction"].values()) < .02)

    result = {
        "status": "mlp0_prefix_token_state_router_feasibility_complete",
        "rung": 408,
        "claim_level": "heldout_off_policy_prefix_state_feasibility_not_physical_router",
        "convention": "CE added above saved native; lower is better",
        "authority": {
            "parent": str(PARENT),
            "loss_artifact": str(LOSS_ARTIFACT),
            "loss_tensor_sha256": _tensor_sha256(losses),
            "expected_loss_tensor_sha256": LOSS_TENSOR_SHA,
            "loss_shape": list(losses.shape),
            "program_names": list(names),
            "row_cache": str(CONFIRM_CACHE),
            "source_documents": SOURCE_DOCUMENTS,
            "train_documents_half_open": [0, TRAIN_DOCUMENTS],
            "evaluation_documents_half_open": [TRAIN_DOCUMENTS, SOURCE_DOCUMENTS],
            "evaluation_waves": [[192, 288], [288, 384]],
            "scoring_positions_half_open": [SCORING_START, SCORING_STOP],
            "authority_exact": authority_exact,
            "FINAL_opened": 0,
        },
        "feature_definition": {
            "columns": [
                "position_over_255", "repeated", "log_repeat_distance",
                "training_log_frequency", "current_morphology4_onehot",
                "previous_morphology4_onehot"],
            "dimension": 12,
            "morphology_states": [
                "word_start_alpha", "continuation_alpha", "digit", "other"],
            "training_frequency_only": True,
            "future_perturbation_max_abs_feature_change": future_perturb_max_abs,
            "finite": finite_features,
        },
        "evaluation_reference": {
            "native_ce": native_eval_ce,
            "I_active_damage": i_damage,
            "Fisher_damage": fisher_damage,
            "covariance_p768_damage": p768_damage,
            "future_label_position_oracle_damage": oracle_damage,
            "I_to_oracle_available_gain": oracle_gain,
        },
        "tree": tree_summary,
        "controls": controls,
        "literal_price": {
            "two_p448_shared_down_bias_values": TWO_P448_VALUES,
            "tree_internal_and_leaf_values": tree_values,
            "training_frequency_table_values_if_used": frequency_table_values,
            "optimistic_total_values": router_price,
            "p768_values": P768_VALUES,
            "runtime_operations_and_prefix_cache_not_priced": True,
        },
        'pred_a_authority_split_and_state_instrument_are_exact': bool(pred_a),
        'pred_b_heldout_position_oracle_has_price_relevant_headroom': bool(pred_b),
        'pred_c_four_prefix_states_recover_enough_headroom': bool(pred_c),
        'pred_d_tree_gain_transports_and_uses_both_experts': bool(pred_d),
        "null_cheap_prefix_state_cannot_harvest_oracle": bool(strong_null),
        "next_object": (
            "physical_fixed_four_state_I_Fisher_router"
            if pred_a and pred_b and pred_c and pred_d and not strong_null else None),
        "physical_router_or_adoption_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("MLP0 PREFIX TOKEN STATE ROUTER FEASIBILITY DONE")


if __name__ == "__main__":
    main()
