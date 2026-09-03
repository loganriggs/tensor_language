#!/usr/bin/env python3
"""CPU-only design support for R582 cached-value downstream-use decomposition.

This module creates outcome-blind, source-matched successor/copy prompts and
implements the exact bilinear response algebra used by the preregistration.  It
does not import or load the model and cannot run a scientific outcome.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
PREREG = ROOT / "polynomial_causal" / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG582_PREREGISTRATION.md"
ROWS = BQ / "numbered_list_cached_value_downstream_use_rows_rung582.json"
RECEIPT = BQ / "numbered_list_cached_value_downstream_use_rows_rung582_receipt.json"
DRYRUN = BQ / "numbered_list_cached_value_downstream_use_rung582_dryrun.json"
R576_RESULT = BQ / "numbered_list_cached_value_weight_removal_rung576_results.json"
R579_AUDIT = BQ / "numbered_list_cached_value_weight_removal_rung579_audit.json"
R576_PREREG = ROOT / "polynomial_causal" / "NUMBERED_LIST_CACHED_VALUE_WEIGHT_REMOVAL_RUNG576_PREREGISTRATION.md"
AUTHORITIES = {
    R576_RESULT: "a6041c28cefc4f695f6e649210884774ed576bae80c14c031473d6b8c8ff2f73",
    R579_AUDIT: "03c03cf9fafe343584f323440d3eab4ab686a70fce44bc36d0fb2ccec945bf2d",
    R576_PREREG: "a776ebc1df29a6f3193d3315e190ec9494c95905596e450461c002378f8f59b6",
}
ENC = tiktoken.get_encoding("gpt2")
SEED = 582
BATCH = 24
SITES = (8, 10, 12, 14)
COMPONENT_ARMS = ("background_cross", "contrast_self", "joint_response")
NULL_ARMS = ("different_group_same_cell", "same_source_other_action")

NUMBER_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty",
}

SPLITS = {
    "FIT": {
        "count": 16, "seed": 58201, "sources": (8, 9),
        "words": ("acorn", "beacon", "cabin", "drum", "ember", "flask", "grove", "harp",
                  "inlet", "kettle", "meadow", "orchard", "pebble", "reef", "saddle", "violin"),
    },
    "SELECT": {
        "count": 8, "seed": 58202, "sources": (11, 12),
        "words": ("alcove", "bonnet", "cradle", "dune", "easel", "fountain", "granite", "hinge",
                  "igloo", "lantern", "mosaic", "oar", "pillow", "ridge", "silo", "vase"),
    },
    "FINAL_TEST": {
        "count": 8, "seed": 58203, "sources": (14, 15),
        "words": ("archway", "blossom", "compass", "delta", "elm", "fossil", "geyser", "hammock",
                  "inkwell", "lighthouse", "marble", "obelisk", "parchment", "raft", "summit", "velvet"),
    },
    "OOD": {
        "count": 8, "seed": 58204, "sources": (17, 18),
        "words": ("activation", "basis", "circuit", "decoder", "eigenvalue", "feature", "graph", "hessian",
                  "intervention", "jacobian", "matrix", "operator", "projector", "residual", "subspace", "vector"),
    },
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_authorities() -> dict[str, str]:
    for path, expected in AUTHORITIES.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"frozen R576/R579 authority changed: {path}")
    result = json.loads(R576_RESULT.read_text())
    audit = json.loads(R579_AUDIT.read_text())
    if not (result["pred_a_exact_weight_compilation"] is True
            and result["fit_report"]["list_necessity_pass"] is True
            and result["fit_report"]["active_copy_controls_pass"] is False
            and result["decision"] == "removal_or_selectivity_null"):
        raise RuntimeError("R576 no longer records the prerequisite held/null boundary")
    if not (audit["all_checks_pass"] is True
            and audit["source_result_sha256"] == AUTHORITIES[R576_RESULT]):
        raise RuntimeError("R579 no longer independently audits the pinned R576 result")
    return {str(path): expected for path, expected in AUTHORITIES.items()}


def _encode(text: str) -> list[int]:
    ids = ENC.encode(text)
    if ENC.decode(ids) != text:
        raise RuntimeError("tokenization does not round-trip")
    return ids


def _one_token(text: str) -> int:
    ids = _encode(text)
    if len(ids) != 1:
        raise RuntimeError(f"registered value is not one token: {text!r} -> {ids}")
    return ids[0]


def _pack(prefix: str, source_text: str, suffix: str, answer: str | None,
          *, structural_answer: str | None = None,
          arithmetic_answer: str | None = None) -> dict:
    text = prefix + source_text + suffix
    ids = _encode(text)
    source_position = len(_encode(prefix))
    source_id = _one_token(source_text)
    if ids[source_position] != source_id:
        raise RuntimeError("source token merged across a prompt boundary")
    result = {
        "text": text,
        "ids": ids,
        "source_position": source_position,
        "source_id": source_id,
        "query_position": len(ids) - 1,
        "answer": answer,
        "answer_id": None if answer is None else _one_token(answer),
    }
    if structural_answer is not None:
        result["structural_answer"] = structural_answer
        result["structural_answer_id"] = _one_token(structural_answer)
    if arithmetic_answer is not None:
        result["arithmetic_answer"] = arithmetic_answer
        result["arithmetic_answer_id"] = _one_token(arithmetic_answer)
    return result


def _list_prompt(values: tuple[int, int, int], words: tuple[str, str, str]) -> tuple[str, str, str]:
    prefix = "".join(f"{value}. {word}\n" for value, word in zip(values[:2], words[:2], strict=True))
    return prefix, str(values[2]), f". {words[2]}\n"


def _sequence_prompt(values: tuple[int, int, int], word: str, representation: str,
                     variant: int) -> tuple[str, str, str]:
    render = (lambda value: str(value)) if representation == "digit" else (lambda value: NUMBER_WORD[value])
    a, b, c = (render(value) for value in values)
    lead = f"The {word} number sequence is" if variant == 0 else f"For the {word}, the numbers are"
    return f"{lead} {a}, {b},", f" {c}", ","


def _condition(group_id: str, split: str, representation: str, source_level: int,
               source_value: int, condition: str, values: tuple[int, int, int],
               words: tuple[str, str, str], variant: int) -> dict:
    action = "copy" if condition in {"factorial_copy", "surface_copy"} else (
        "successor" if condition in {"factorial_successor", "surface_successor", "relation_break"} else "step_two")
    shown_words = tuple(reversed(words)) if condition.startswith("surface_") else words
    shown_variant = 1 - variant if condition.startswith("surface_") else variant
    if representation == "list":
        prefix, source, suffix = _list_prompt(values, shown_words)
        answer = str(source_value if action == "copy" else source_value + 1)
        structural, arithmetic = str(source_value + 1), str(source_value + 2)
    else:
        # All three group words remain visible so independently generated groups
        # cannot collapse to the same sequence prompt merely by sharing one noun.
        word_phrase = " ".join(shown_words)
        prefix, source, suffix = _sequence_prompt(values, word_phrase, representation, shown_variant)
        render = (lambda value: f" {value}") if representation == "digit" else (lambda value: " " + NUMBER_WORD[value])
        answer = render(source_value if action == "copy" else source_value + 1)
        structural, arithmetic = render(source_value + 1), render(source_value + 2)
    packed = _pack(prefix, source, suffix, None if action == "step_two" else answer,
                   structural_answer=structural if action == "step_two" else None,
                   arithmetic_answer=arithmetic if action == "step_two" else None)
    identity = {
        "rung": 582, "group_id": group_id, "representation": representation,
        "source_level": source_level, "condition": condition,
    }
    packed.update({
        "row_id": canonical_hash(identity), "group_id": group_id, "split": split,
        "representation": representation, "source_level": source_level,
        "source_value": source_value, "condition": condition, "action": action,
        "values": list(values), "model_outcome_opened": False,
    })
    return packed


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for split, spec in SPLITS.items():
        rng = random.Random(spec["seed"])
        seen_words: set[tuple[str, str, str]] = set()
        for group_index in range(spec["count"]):
            while True:
                words = tuple(rng.sample(spec["words"], 3))
                if words not in seen_words and tuple(reversed(words)) not in seen_words:
                    seen_words.add(words)
                    seen_words.add(tuple(reversed(words)))
                    break
            variant = rng.randrange(2)
            group_id = canonical_hash({"rung": 582, "split": split, "index": group_index, "words": words})
            for representation in ("list", "digit", "word"):
                for source_level, source_value in enumerate(spec["sources"]):
                    conditions = {
                        "factorial_copy": (source_value, source_value, source_value),
                        "factorial_successor": (source_value - 2, source_value - 1, source_value),
                        "surface_copy": (source_value, source_value, source_value),
                        "surface_successor": (source_value - 2, source_value - 1, source_value),
                        "relation_break": (source_value - 3, source_value - 1, source_value),
                        "step_two": (source_value - 4, source_value - 2, source_value),
                    }
                    for condition, values in conditions.items():
                        rows.append(_condition(group_id, split, representation, source_level,
                                               source_value, condition, values, words, variant))
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    expected_per_group = 3 * 2 * 6
    counts = collections.Counter(str(row["split"]) for row in rows)
    groups = collections.defaultdict(list)
    prompt_ids, row_ids = set(), set()
    for row in rows:
        groups[str(row["group_id"])].append(row)
        prompt = tuple(int(value) for value in row["ids"])
        if prompt in prompt_ids or row["row_id"] in row_ids:
            raise RuntimeError("duplicate prompt or row ID")
        prompt_ids.add(prompt); row_ids.add(row["row_id"])
        if row["model_outcome_opened"] is not False:
            raise RuntimeError("builder contains a model outcome")
        ids = list(row["ids"])
        if ids[int(row["source_position"])] != int(row["source_id"]):
            raise RuntimeError("source position is wrong")
        if int(row["query_position"]) != len(ids) - 1:
            raise RuntimeError("query is not final")
    if set(groups) and any(len(items) != expected_per_group for items in groups.values()):
        raise RuntimeError("incomplete semantic group")
    group_splits = {group: {str(row["split"]) for row in items} for group, items in groups.items()}
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise RuntimeError("semantic group crosses splits")
    for items in groups.values():
        cells = {(row["representation"], row["source_level"], row["condition"]) for row in items}
        expected = {(rep, source, condition) for rep in ("list", "digit", "word")
                    for source in (0, 1) for condition in (
                        "factorial_copy", "factorial_successor", "surface_copy",
                        "surface_successor", "relation_break", "step_two")}
        if cells != expected:
            raise RuntimeError("group does not contain the frozen source/action cells")
        lookup = {(row["representation"], row["source_level"], row["condition"]): row for row in items}
        for rep in ("list", "digit", "word"):
            for source in (0, 1):
                copy = lookup[(rep, source, "factorial_copy")]
                successor = lookup[(rep, source, "factorial_successor")]
                if copy["source_id"] != successor["source_id"] or copy["source_value"] != successor["source_value"]:
                    raise RuntimeError("copy/successor action contrast changed the final source")
    expected_counts = {split: spec["count"] * expected_per_group for split, spec in SPLITS.items()}
    if dict(counts) != expected_counts:
        raise RuntimeError(f"unexpected split counts: {dict(counts)}")
    return {"rows": len(rows), "groups": len(groups), "split_rows": expected_counts,
            "group_disjoint": True, "source_matched_action_cells": True}


def bilinear_response(left: np.ndarray, right: np.ndarray, down: np.ndarray,
                      state_without: np.ndarray, state_with: np.ndarray) -> dict[str, np.ndarray]:
    """Exact finite response of D[(Lx)*(Rx)] split into cross and self terms.

    ``state_without`` and ``state_with`` are already-normalized states from the
    paired source-deleted and source-present prefix trajectories.  Combining the
    two ordered cross terms makes the result invariant to swapping L and R.
    """
    left = np.asarray(left); right = np.asarray(right); down = np.asarray(down)
    x0 = np.asarray(state_without); x1 = np.asarray(state_with)
    delta = x1 - x0
    l0, r0 = x0 @ left.T, x0 @ right.T
    ld, rd = delta @ left.T, delta @ right.T
    cross = (ld * r0 + l0 * rd) @ down.T
    self_term = (ld * rd) @ down.T
    joint = cross + self_term
    direct = ((x1 @ left.T) * (x1 @ right.T) - l0 * r0) @ down.T
    scale = max(float(np.sum(direct.astype(np.float64) ** 2)), 1e-30)
    relative_error = float(np.sum((joint - direct).astype(np.float64) ** 2) / scale)
    return {"background_cross": cross, "contrast_self": self_term,
            "joint_response": joint, "direct_response": direct,
            "relative_squared_error": np.asarray(relative_error)}


def two_factor_mobius(native: float, remove_cross: float, remove_self: float,
                      remove_joint: float) -> dict[str, float]:
    """Möbius coefficients for two removals, using larger-is-better outcome Y."""
    return {
        "cross": float(remove_cross - native),
        "self": float(remove_self - native),
        "cross_x_self": float(remove_joint - remove_cross - remove_self + native),
    }


def deterministic_group_bootstrap(values: Mapping[str, float], *, cell_id: str,
                                  replicates: int = 2000) -> np.ndarray:
    """Deterministic group bootstrap whose every draw is content-addressed."""
    keys = sorted(values)
    if not keys:
        raise ValueError("bootstrap requires at least one group")
    output = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        sample = []
        for k in range(len(keys)):
            payload = f"r582-group-bootstrap-v1:{cell_id}:{b}:{k}".encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(keys)
            sample.append(float(values[keys[index]]))
        output[b] = float(np.mean(sample))
    return output


def action_gap_records(records: Iterable[Mapping[str, object]], metric: str) -> dict[tuple[str, int, str], float]:
    """Successor-minus-copy effect at fixed group, format, source, and surface."""
    cells: dict[tuple[str, str, int, str], float] = {}
    for row in records:
        condition = str(row["condition"])
        if condition not in {"factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}:
            continue
        surface = "surface" if condition.startswith("surface_") else "factorial"
        action = "copy" if condition.endswith("copy") else "successor"
        key = (str(row["group_id"]), str(row["representation"]), int(row["source_level"]), surface)
        cell = (*key, action)
        if cell in cells:
            raise ValueError("duplicate action cell")
        cells[cell] = float(row[metric])
    bases = {(group, rep, source, surface) for group, rep, source, surface, _ in cells}
    output = {}
    for group, rep, source, surface in bases:
        succ = (*((group, rep, source, surface)), "successor")
        copy = (*((group, rep, source, surface)), "copy")
        if succ not in cells or copy not in cells:
            raise ValueError("unmatched source/action cell")
        output[(group, source, f"{rep}:{surface}")] = cells[succ] - cells[copy]
    return output


def deterministic_null_maps(rows: Sequence[Mapping[str, object]], split: str) -> dict[str, dict[str, str]]:
    """Return fully matched donor row IDs for both frozen null interventions."""
    selected = [row for row in rows if row["split"] == split and row["condition"] in {
        "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]
    if not selected:
        raise ValueError(f"split has no null-eligible rows: {split}")
    by_id = {str(row["row_id"]): row for row in selected}
    if len(by_id) != len(selected):
        raise ValueError("null-eligible row IDs are not unique")
    strata = collections.defaultdict(list)
    for row in selected:
        key = (str(row["representation"]), int(row["source_level"]), str(row["condition"]))
        strata[key].append(row)
    different: dict[str, str] = {}
    for key, items in strata.items():
        if len(items) < 2:
            raise ValueError(f"different-group null stratum has fewer than two rows: {key}")
        ordered = sorted(items, key=lambda row: canonical_hash({
            "seed": SEED, "null": "different_group_same_cell", "row_id": row["row_id"]}))
        for index, row in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            if row["group_id"] == donor["group_id"]:
                raise RuntimeError("different-group null failed to derange groups")
            different[str(row["row_id"])] = str(donor["row_id"])
    lookup = {(str(row["group_id"]), str(row["representation"]), int(row["source_level"]),
               str(row["condition"])): str(row["row_id"]) for row in selected}
    other_action: dict[str, str] = {}
    for row in selected:
        condition = str(row["condition"])
        partner = condition.replace("copy", "successor") if condition.endswith("copy") else condition.replace("successor", "copy")
        key = (str(row["group_id"]), str(row["representation"]), int(row["source_level"]), partner)
        if key not in lookup:
            raise ValueError(f"same-source other-action partner missing: {key}")
        donor_id = lookup[key]
        donor = by_id[donor_id]
        if donor["source_id"] != row["source_id"] or donor["source_value"] != row["source_value"]:
            raise RuntimeError("other-action null changed the final source")
        other_action[str(row["row_id"])] = donor_id
    return {"different_group_same_cell": different, "same_source_other_action": other_action}


def batch_count(rows: Sequence[Mapping[str, object]], split: str) -> int:
    lengths = collections.Counter(len(row["ids"]) for row in rows if row["split"] == split)
    return sum(math.ceil(count / BATCH) for count in lengths.values())


def dryrun_document(rows: Sequence[Mapping[str, object]]) -> dict:
    authorities = validate_authorities()
    validation = validate_rows(rows)
    fit_batches, select_batches = batch_count(rows, "FIT"), batch_count(rows, "SELECT")
    null_counts = {split: {name: len(mapping) for name, mapping in deterministic_null_maps(rows, split).items()}
                   for split in ("FIT", "SELECT")}
    # Each split first captures source-present and source-deleted prefixes. FIT
    # then evaluates 3 components at 4 sites and 2 nulls at the selected site;
    # SELECT evaluates only the selected site's 3 components and 2 nulls.
    fit_arms = 2 + len(SITES) * len(COMPONENT_ARMS) + len(NULL_ARMS)
    select_arms = 2 + len(COMPONENT_ARMS) + len(NULL_ARMS)
    maximum = fit_batches * fit_arms + select_batches * select_arms
    return {
        "rung": 582, "stage": "cpu_only_preregistration_dry_run",
        "authority_sha256": authorities,
        "validation": validation, "sites": list(SITES),
        "component_arms": list(COMPONENT_ARMS), "null_arms": list(NULL_ARMS),
        "fit_batches": fit_batches, "select_batches": select_batches,
        "null_donor_counts": null_counts,
        "fit_arm_forwards_per_batch": fit_arms,
        "select_arm_forwards_per_batch": select_arms,
        "maximum_model_forwards_if_eventually_executed": maximum,
        "model_forwards": 0, "model_backwards": 0, "model_loaded": False,
        "model_weights_updated": False, "opened_splits": [],
        "FINAL_TEST_or_OOD_opened": False,
    }


def main() -> None:
    rows = build_rows()
    dryrun = dryrun_document(rows)
    document = {
        "schema": "r582.cached_value_downstream_use.rows.v1", "rung": 582,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "outcomes_opened": [], "rows": rows,
    }
    ROWS.write_text(json.dumps(document, indent=1) + "\n")
    receipt = {
        "schema": "r582.cached_value_downstream_use.receipt.v1", "rung": 582,
        **dryrun["validation"], "rows_sha256": file_sha256(ROWS),
        "split_policy": {split: {"groups": spec["count"], "seed": spec["seed"],
                                  "sources": list(spec["sources"])} for split, spec in SPLITS.items()},
        "model_loaded": False, "model_forwards": 0, "outcomes_opened": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    dryrun["rows_sha256"] = file_sha256(ROWS)
    dryrun["receipt_sha256"] = file_sha256(RECEIPT)
    if PREREG.exists():
        dryrun["preregistration_sha256"] = file_sha256(PREREG)
    DRYRUN.write_text(json.dumps(dryrun, indent=1) + "\n")
    print(json.dumps(dryrun, indent=1))


if __name__ == "__main__":
    main()
