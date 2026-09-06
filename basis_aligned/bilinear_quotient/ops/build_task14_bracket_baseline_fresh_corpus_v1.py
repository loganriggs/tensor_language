#!/usr/bin/env python3
"""Freeze all-row baseline coefficients and untouched Task14/bracket rows."""

# BQLANE: cpu
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import circuit_battery_task14 as task_base
import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as task_prior
import circuit_fast_screen_managed_runner as managed
from build_bracket_suffix_free_fresh_corpus_v1 import DELIMITERS, PAIRS, digest, encode, one
from run_task14_bracket_native_baseline_semantic_linear_feasibility_v1 import bracket_features, task_features

ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_feasibility_v1_result.json"
V4 = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
TASK_OUT = ROOT / "circuits/prior_art/task14_native_baseline_fresh_corpus_v1_rows.json"
BRACKET_OUT = ROOT / "circuits/prior_art/bracket_native_baseline_fresh_corpus_v1_rows.json"
COEFFICIENT_OUT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_v1_artifact.json"
EXPECTED = {
    FEASIBILITY: "62d9d6e302d60b5372a13b3fbf119c6dfb333375b775497886515d0134fe29ca",
    V4: "85c5cc0549421fc1575d96ce621d0677ea4b0cc2d154b2c0bf7af90f4148bd4c",
}
TASK_SCHEMA = "task14_native_baseline_fresh_corpus_rows_v1"
TASK_PAIRS = (("spy", "spies"), ("knight", "knights"), ("cook", "cooks"), ("guide", "guides"), ("fan", "fans"), ("sibling", "siblings"), ("girl", "girls"), ("lion", "lions"), ("tiger", "tigers"), ("bear", "bears"), ("rabbit", "rabbits"), ("rat", "rats"), ("snake", "snakes"), ("shark", "sharks"), ("whale", "whales"), ("bee", "bees"))
BRACKET_PREFIXES = ("The navigator", "A sculptor", "The librarian", "One gardener", "The astronomer", "A violinist")
BRACKET_WORDS = ("apricot", "basket", "candle", "desert", "feather", "garden", "hammer", "igloo", "jacket", "ladder", "meadow", "needle", "oyster", "pebble", "ribbon", "tunnel", "walnut", "yellow")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_rows() -> list[dict]:
    rows = []
    for group in range(16):
        recipient_number = 0 if group < 8 else 1
        within = group % 8
        direction = "singular_to_plural" if recipient_number == 0 else "plural_to_singular"
        lexical_group = (0 if recipient_number == 0 else 8) + (within + 3) % 8
        a1_group, a2_group = (group + 5) % 16, (group + 11) % 16
        attractor_state = ((within % 4) // 2, within % 2)
        for template_id, template in task_prior.TEMPLATES:
            subjects = {"recipient": TASK_PAIRS[group][recipient_number], "opposite_same_lemma": TASK_PAIRS[group][1 - recipient_number], "same_number_different_lemma": TASK_PAIRS[lexical_group][recipient_number]}
            endpoints = {}
            for role in task_prior.ROLES:
                number = 1 - recipient_number if role == "opposite_same_lemma" else recipient_number
                text = template.format(a1=TASK_PAIRS[a1_group][attractor_state[0]], a2=TASK_PAIRS[a2_group][attractor_state[1]], subject=subjects[role])
                ids = task_base.ENCODING.encode(text)
                if len(ids) != 9 or task_base.ENCODING.decode(ids) != text:
                    raise ValueError(f"Task14 tokenization changed: {text}")
                answer = 318 if number == 0 else 389
                endpoints[role] = {"text": text, "ids": ids, "subject": subjects[role], "subject_number": "singular" if number == 0 else "plural", "answer_id": answer, "foil_id": 389 if answer == 318 else 318}
            identity = [TASK_SCHEMA, group, template_id, attractor_state, endpoints]
            rows.append({"schema": TASK_SCHEMA, "task_id": "subject_verb.number_agreement", "phase": "PROSPECTIVE_UNOPENED", "group_number": group, "row_id": task_prior.canonical_sha256(identity), "template_id": template_id, "direction_id": direction, "attractor_state": list(attractor_state), "subject_position": 8, "endpoints": endpoints})
    prior_vocabulary, prior_prompts, prior_tokens = task_prior._prior_material()
    prior_vocabulary.update(word for pair in task_prior.NOUN_PAIRS for word in pair)
    forms = {word for pair in TASK_PAIRS for word in pair}
    if forms & prior_vocabulary or any(len(task_base.ENCODING.encode(" " + word)) != 1 for word in forms):
        raise ValueError("Task14 lexical novelty/tokenization failed")
    prompts = [endpoint["text"] for row in rows for endpoint in row["endpoints"].values()]
    tokens = [tuple(endpoint["ids"]) for row in rows for endpoint in row["endpoints"].values()]
    if len(rows) != 32 or sorted(Counter((row["direction_id"], row["template_id"]) for row in rows).values()) != [8] * 4 or len(set(prompts)) != 96 or set(prompts) & prior_prompts or set(tokens) & prior_tokens:
        raise ValueError("Task14 fresh authority balance/novelty failed")
    return rows


def bracket_rows() -> list[dict]:
    rng = random.Random(202609060111)
    rows = []
    for left_index, right_index in PAIRS:
        left, right = DELIMITERS[left_index], DELIMITERS[right_index]
        distractor = DELIMITERS[({0, 1, 2} - {left_index, right_index}).pop()]
        for replicate, prefix in enumerate(BRACKET_PREFIXES):
            w0, w1, w2, w3, w4 = rng.sample(BRACKET_WORDS, 5)
            starts = (f"Before filing the ledger, {prefix.lower()} sealed", f"Once the inventory was checked, {prefix.lower()} closed", f"During the final audit, {prefix.lower()} completed")
            common = f"{starts[replicate % 3]} {distractor['open']} the {w0} and the {w1} {distractor['close']}; afterward the appendix started"
            tail = f"the {w2}, the {w3}, and the {w4} with no closing mark"
            base, donor = f"{common} {left['open']} {tail}", f"{common} {right['open']} {tail}"
            base_ids, donor_ids = encode(base), encode(donor)
            differences = [index for index, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
            if len(base_ids) != len(donor_ids) or len(differences) != 1:
                raise ValueError("bracket pair is not a one-token edit")
            coordinates = {"family": "ledger_completed_distractor_pending_type_substitution", "pair": [left_index, right_index], "replicate": replicate, "prefix": prefix, "words": [w0, w1, w2, w3, w4]}
            rows.append({"row_id": digest(coordinates), "split": "PROSPECTIVE_NATIVE_BASELINE_V1", "family_id": "ledger_completed_distractor_pending_type_substitution", "program_role": "target", "base_text": base, "donor_text": donor, "base_ids": base_ids, "donor_ids": donor_ids, "base_answer": left["close"], "donor_answer": right["close"], "base_answer_id": one(left["close"]), "donor_answer_id": one(right["close"]), "evaluation_directions": ["base_to_donor", "donor_to_base"], "construction_checks": {"roundtrip": True, "equal_token_length": True, "single_token_difference": True, "completed_distractor_type": distractor["name"]}})
    counts = Counter((row["base_answer_id"], row["donor_answer_id"]) for row in rows)
    if len(rows) != 36 or len({row["row_id"] for row in rows}) != 36 or len(counts) != 6 or set(counts.values()) != {6}:
        raise ValueError("bracket fresh authority balance failed")
    return rows


def coefficients(result: dict) -> dict:
    task_records = result["task14_evidence"]
    bracket_records = result["bracket_evidence"]
    task_x = np.asarray([task_features(row) for row in task_records], dtype=np.float64)
    task_y = np.asarray([row["native_donorward_baseline_margin"] for row in task_records], dtype=np.float64)
    bracket_x = np.asarray([bracket_features(row) for row in bracket_records], dtype=np.float64)
    bracket_y = np.asarray([row["native_donorward_baseline_margin"] for row in bracket_records], dtype=np.float64)
    return {"task14": np.linalg.lstsq(task_x, task_y, rcond=None)[0].tolist(), "bracket": np.linalg.lstsq(bracket_x, bracket_y, rcond=None)[0].tolist()}


def main() -> None:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable source changed: {path}")
    if TASK_OUT.exists() or BRACKET_OUT.exists() or COEFFICIENT_OUT.exists():
        raise ValueError("refusing overwrite")
    feasibility = json.loads(FEASIBILITY.read_text())
    if feasibility["terminal"] != "feasibility_screen":
        raise ValueError("feasibility did not pass")
    task = task_rows(); bracket = bracket_rows(); fitted = coefficients(feasibility)
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    task_value = {"schema": TASK_SCHEMA, "status": "rows_frozen_outcomes_unopened", "created_utc": created, "row_count": 32, "endpoint_count": 96, "direction_template_counts": {f"{direction}.{template}": count for (direction, template), count in sorted(Counter((row["direction_id"], row["template_id"]) for row in task).items())}, "model_loaded": False, "model_forwards": 0, "outcomes_opened": [], "rows": task}
    bracket_counts = Counter((row["base_answer_id"], row["donor_answer_id"]) for row in bracket)
    bracket_value = {"schema": "bracket_native_baseline_fresh_corpus_rows_v1", "status": "rows_frozen_outcomes_unopened", "created_utc": created, "family_id": "ledger_completed_distractor_pending_type_substitution", "row_count": 36, "endpoint_count": 72, "ordered_pair_row_counts": {f"{a}->{b}": count for (a, b), count in sorted(bracket_counts.items())}, "model_loaded": False, "model_forwards": 0, "outcomes_opened": [], "rows": bracket}
    task_bytes = managed.atomic_create_json(TASK_OUT, task_value)
    bracket_bytes = managed.atomic_create_json(BRACKET_OUT, bracket_value)
    artifact = {"schema": "task14_bracket_native_baseline_semantic_linear_artifact_v1", "candidate_id": "cross_behavior.task14_bracket_native_baseline_semantic_linear_prospective_v2", "created_utc": created, "source_feasibility_sha256": EXPECTED[FEASIBILITY], "features": {"task14": ["intercept", "direction_sign", "has_E", "has_A", "has_U", "has_W"], "bracket": ["intercept", "recipient_is_1", "recipient_is_8", "donor_is_1", "donor_is_8"]}, "coefficients": fitted, "fits": 2, "task14_rows_sha256": hashlib.sha256(task_bytes).hexdigest(), "bracket_rows_sha256": hashlib.sha256(bracket_bytes).hexdigest(), "outcomes_opened": [], "terminal": "frozen_artifact"}
    artifact_bytes = managed.atomic_create_json(COEFFICIENT_OUT, artifact)
    print(json.dumps({"task14_rows_sha256": hashlib.sha256(task_bytes).hexdigest(), "bracket_rows_sha256": hashlib.sha256(bracket_bytes).hexdigest(), "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "task14_rows": 32, "bracket_rows": 36, "outcomes_opened": []}, sort_keys=True))


if __name__ == "__main__":
    main()
