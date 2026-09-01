#!/usr/bin/env python3
"""Rung 440: physically separated archive-feasibility extractor.

Run in this order:

  python simplicity_archive_feasibility.py features
  python simplicity_archive_feasibility.py labels --feature-sha <printed sha256>
  python simplicity_archive_feasibility.py audit

This script does not fit a predictor. See SIMPLICITY_ARCHIVE_FEASIBILITY_PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
HERE = Path(__file__).resolve().parent
FEATURE_PATH = HERE / "simplicity_archive_features_v1.json"
LABEL_PATH = HERE / "simplicity_archive_labels_v1.json"
AUDIT_PATH = HERE / "simplicity_archive_feasibility_results.json"
RUNG_MIN = 300
RUNG_MAX = 436
EXCLUDED_NAME_FRAGMENTS = (
    "invalid",
    "diagnostic",
    "initial",
    "first_invalid",
    "semantic_hash_repaired",
    "orthogonality_diagnostic",
)
STRUCTURAL_TOKENS = (
    "price",
    "byte",
    "scalar",
    "parameter",
    "rank",
    "width",
    "atom",
    "edge",
    "interface",
    "operation",
    "runtime",
)
FORBIDDEN_FEATURE_KEY_TOKENS = (
    "pred_",
    "null",
    "ce",
    "loss",
    "error",
    "fidelity",
    "accuracy",
    "cosine",
    "correlation",
    "certificate",
    "damage",
)


def key_has_forbidden_token(key: str) -> bool:
    lower = key.lower()
    if lower.startswith("pred_"):
        return True
    parts = set(lower.replace("-", "_").split("_"))
    return any(token in parts for token in FORBIDDEN_FEATURE_KEY_TOKENS if token != "pred_")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def registered_predicates(value: dict[str, Any]) -> dict[str, bool]:
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and key.startswith("pred_") and isinstance(item, bool)
    }


def exclusion_reason(path: Path, value: dict[str, Any]) -> str | None:
    lower_name = path.name.lower()
    for fragment in EXCLUDED_NAME_FRAGMENTS:
        if fragment in lower_name:
            return f"excluded_name_fragment:{fragment}"
    status = str(value.get("status", "")).lower()
    if "invalid" in status:
        return "status_contains_invalid"
    return None


def eligible_receipts() -> tuple[dict[int, list[tuple[Path, dict[str, Any]]]], list[dict[str, Any]]]:
    by_rung: dict[int, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    for path in sorted(BQ.glob("*_results.json")):
        value = load_json(path)
        if value is None:
            continue
        rung = value.get("rung")
        if not isinstance(rung, int) or not RUNG_MIN <= rung <= RUNG_MAX:
            continue
        if len(registered_predicates(value)) < 3:
            continue
        reason = exclusion_reason(path, value)
        if reason is not None:
            exclusions.append({"rung": rung, "receipt": path.name, "reason": reason})
            continue
        by_rung[rung].append((path, value))
    return by_rung, exclusions


def canonical_receipts() -> tuple[list[tuple[int, Path, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_rung, exclusions = eligible_receipts()
    canonical: list[tuple[int, Path, dict[str, Any]]] = []
    ambiguous: list[dict[str, Any]] = []
    for rung, rows in sorted(by_rung.items()):
        if len(rows) != 1:
            ambiguous.append(
                {
                    "rung": rung,
                    "receipts": [path.name for path, _ in rows],
                    "reason": "multiple_nonexcluded_receipts_for_rung",
                }
            )
            continue
        path, value = rows[0]
        canonical.append((rung, path, value))
    return canonical, exclusions, ambiguous


def git_first_add_timestamps(paths: list[Path]) -> dict[Path, int]:
    """Read add-times in one Git walk; one subprocess per receipt is prohibitively slow."""
    rels = [path.relative_to(ROOT) for path in paths]
    proc = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=@@%ct", "--name-only", "--", *map(str, rels)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    current_stamp: int | None = None
    output: dict[Path, int] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("@@") and line[2:].isdigit():
            current_stamp = int(line[2:])
            continue
        if current_stamp is None or not line.strip():
            continue
        absolute = ROOT / line.strip()
        if absolute in paths:
            output[absolute] = min(current_stamp, output.get(absolute, current_stamp))
    return output


def classify_module(name: str) -> str:
    n = name.lower()
    if "attention0" in n or "attn0" in n:
        return "attention0"
    if "mlp0" in n or "early_mlp" in n:
        return "mlp0_or_early_mlp"
    if "mlp" in n:
        return "other_mlp"
    if "qk" in n or "attention" in n or "attn" in n:
        return "other_attention"
    if "vocab" in n or "embedding" in n or "token" in n:
        return "vocabulary"
    if "state" in n or "hankel" in n:
        return "predictive_state"
    if "mixed" in n or "whole_program" in n or "frontier" in n:
        return "whole_program"
    return "other"


def classify_grammar(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("sparse", "code", "vocab", "dictionary", "atom")):
        return "sparse_or_dictionary"
    if any(x in n for x in ("pca", "svd", "tucker", "rank", "lowrank", "low_rank")):
        return "low_rank"
    if any(x in n for x in ("quadratic", "bilinear", "polynomial")):
        return "polynomial"
    if any(x in n for x in ("hankel", "state", "continuous")):
        return "state_or_continuous"
    if any(x in n for x in ("mixed", "composite", "physical", "frontier")):
        return "composite_program"
    if any(x in n for x in ("causal", "intervention", "knockout", "ablation")):
        return "causal_probe"
    return "other"


def find_source(receipt: Path) -> Path | None:
    stem = receipt.name.removesuffix("_results.json")
    candidates = [
        BQ / "ops" / f"{stem}.py",
        BQ / f"{stem}.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def ast_counts(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    nodes = list(ast.walk(tree))
    return {
        "nodes": len(nodes),
        "calls": sum(isinstance(x, ast.Call) for x in nodes),
        "functions": sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) for x in nodes),
        "loops": sum(isinstance(x, (ast.For, ast.AsyncFor, ast.While)) for x in nodes),
        "branches": sum(isinstance(x, (ast.If, ast.IfExp, ast.Match)) for x in nodes),
    }


def finite_numbers(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(finite_numbers(item))
    elif isinstance(value, list):
        for item in value:
            out.extend(finite_numbers(item))
    return out


def structural_summaries(value: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for key, item in value.items():
        lower = key.lower()
        if not any(token in lower for token in STRUCTURAL_TOKENS):
            continue
        if key_has_forbidden_token(lower):
            continue
        numbers = finite_numbers(item)
        if numbers:
            summaries[key] = {
                "count": len(numbers),
                "minimum": min(numbers),
                "maximum": max(numbers),
                "sum_abs": sum(abs(x) for x in numbers),
            }
    return summaries


def explicit_arm_names(value: dict[str, Any]) -> list[str]:
    arms = value.get("arms")
    if not isinstance(arms, dict):
        return []
    return sorted(str(key) for key in arms)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def forbidden_feature_keys(value: Any) -> list[str]:
    return sorted(
        {
            key
            for key in walk_keys(value)
            if key_has_forbidden_token(key)
        }
    )


def command_features() -> None:
    canonical, exclusions, ambiguous = canonical_receipts()
    chronology = git_first_add_timestamps([receipt for _, receipt, _ in canonical])
    rows = []
    for rung, receipt, value in canonical:
        source = find_source(receipt)
        summaries = structural_summaries(value)
        arm_names = explicit_arm_names(value)
        receipt_rel = receipt.relative_to(ROOT)
        source_rel = source.relative_to(ROOT) if source else None
        rows.append(
            {
                "join_key": f"r{rung}:{sha256(receipt)[:16]}",
                "rung": rung,
                "receipt_path": str(receipt_rel),
                "receipt_sha256": sha256(receipt),
                "receipt_first_add_unix": chronology.get(receipt),
                "source_path": str(source_rel) if source_rel else None,
                "source_sha256": sha256(source) if source else None,
                "module_family": classify_module(receipt.name),
                "grammar_family": classify_grammar(receipt.name),
                "source_ast": ast_counts(source),
                "declared_arm_names": arm_names,
                "declared_arm_count": len(arm_names),
                "structural_summaries": summaries,
                "has_machine_readable_price": any(
                    any(token in key.lower() for token in ("price", "byte", "scalar", "parameter"))
                    for key in summaries
                ),
            }
        )
    output = {
        "schema": "simplicity_archive_features_v1",
        "rung_interval_inclusive": [RUNG_MIN, RUNG_MAX],
        "rows": rows,
        "excluded_receipts": exclusions,
        "ambiguous_rungs": ambiguous,
    }
    forbidden = forbidden_feature_keys(output)
    if forbidden:
        raise RuntimeError(f"forbidden feature keys: {forbidden}")
    write_json(FEATURE_PATH, output)
    print(json.dumps({"feature_path": str(FEATURE_PATH), "feature_sha256": sha256(FEATURE_PATH), "rows": len(rows)}))


INSTRUMENT_TOKENS = (
    "instrument",
    "exact",
    "identity",
    "bill",
    "hash",
    "dtype",
    "config",
    "replay",
    "physical",
)
CATEGORY_TOKENS = {
    "ood_transport": ("ood", "fresh", "transfer", "transport", "general", "stable", "stability", "corpus"),
    "extraction_identification": ("extract", "identif", "semantic", "certificate", "census", "circuit"),
    "removal_intervention": ("remov", "intervention", "edit", "signed", "causal", "knockout", "ablation", "collateral"),
    "composition": ("compos", "additive", "interaction", "tax", "joint"),
}


def predicate_categories(key: str) -> tuple[bool, list[str]]:
    lower = key.lower()
    instrument = any(token in lower for token in INSTRUMENT_TOKENS)
    categories = [name for name, tokens in CATEGORY_TOKENS.items() if any(token in lower for token in tokens)]
    return instrument, categories


def command_labels(feature_sha: str) -> None:
    if not FEATURE_PATH.is_file() or sha256(FEATURE_PATH) != feature_sha:
        raise RuntimeError("feature file missing or changed after the supplied frozen hash")
    features = json.loads(FEATURE_PATH.read_text())
    feature_by_rung = {row["rung"]: row for row in features["rows"]}
    canonical, _, _ = canonical_receipts()
    rows = []
    for rung, receipt, value in canonical:
        feature = feature_by_rung.get(rung)
        if feature is None or feature["receipt_sha256"] != sha256(receipt):
            raise RuntimeError(f"feature/label provenance mismatch for rung {rung}")
        predicates = registered_predicates(value)
        pred_records = {}
        for key, result in sorted(predicates.items()):
            instrument, categories = predicate_categories(key)
            pred_records[key] = {"value": result, "instrument": instrument, "categories": categories}
        nulls = {
            key: item
            for key, item in value.items()
            if isinstance(key, str) and "null" in key.lower() and isinstance(item, bool)
        }
        pred_records.update(
            {
                key: {"value": result, "instrument": False, "categories": []}
                for key, result in sorted(nulls.items())
            }
        )
        rows.append({"join_key": feature["join_key"], "registered_labels": pred_records})
    output = {
        "schema": "simplicity_archive_labels_v1",
        "frozen_feature_sha256": feature_sha,
        "rows": rows,
    }
    write_json(LABEL_PATH, output)
    print(json.dumps({"label_path": str(LABEL_PATH), "label_sha256": sha256(LABEL_PATH), "rows": len(rows)}))


def command_audit() -> None:
    features = json.loads(FEATURE_PATH.read_text())
    labels = json.loads(LABEL_PATH.read_text())
    feature_sha = sha256(FEATURE_PATH)
    if labels.get("frozen_feature_sha256") != feature_sha:
        raise RuntimeError("label file does not bind the current feature file")
    frows = features["rows"]
    lrows = labels["rows"]
    f_by_join = {row["join_key"]: row for row in frows}
    l_by_join = {row["join_key"]: row for row in lrows}
    join_match = set(f_by_join) == set(l_by_join)
    forbidden = forbidden_feature_keys(features)
    hashes_complete = all(row["receipt_sha256"] for row in frows)
    chronology_complete = all(row["receipt_first_add_unix"] is not None for row in frows)
    reasons_complete = all(row.get("reason") for row in features["excluded_receipts"] + features["ambiguous_rungs"])
    source_linked = sum(row["source_path"] is not None for row in frows)
    arm_mapped = sum(row["declared_arm_count"] > 0 for row in frows)
    priced = sum(row["has_machine_readable_price"] for row in frows)
    module_families = sorted({row["module_family"] for row in frows})
    grammar_families = sorted({row["grammar_family"] for row in frows})
    category_counts: Counter[str] = Counter()
    category_cells: dict[str, set[str]] = defaultdict(set)
    instrument_labels = 0
    total_predicate_labels = 0
    for join_key, lrow in l_by_join.items():
        frow = f_by_join[join_key]
        cell = f"{frow['module_family']}::{frow['grammar_family']}"
        for key, record in lrow["registered_labels"].items():
            if not key.startswith("pred_"):
                continue
            total_predicate_labels += 1
            if record["instrument"]:
                instrument_labels += 1
                continue
            for category in record["categories"]:
                category_counts[category] += 1
                category_cells[category].add(cell)
    category_pass = {
        category: category_counts[category] >= 25 and len(category_cells[category]) >= 3
        for category in CATEGORY_TOKENS
    }
    n = len(frows)
    unique_structurally_eligible_rungs = n + len(features["ambiguous_rungs"])
    ambiguity_fraction = len(features["ambiguous_rungs"]) / max(1, unique_structurally_eligible_rungs)
    pred_a = bool(join_match and not forbidden and hashes_complete and chronology_complete and reasons_complete)
    pred_b = bool(
        n >= 80
        and len(module_families) >= 5
        and len(grammar_families) >= 4
        and source_linked / max(1, n) >= 0.90
        and chronology_complete
    )
    pred_c = bool(arm_mapped / max(1, n) >= 0.70 and priced / max(1, n) >= 0.70)
    pred_d = sum(category_pass.values()) >= 3
    strong_null = bool(n < 50 or ambiguity_fraction > 0.20 or sum(category_pass.values()) < 2)
    if pred_a and pred_b and pred_c and pred_d:
        routing = "license_rung441_historical_held_family_backtest"
    elif pred_a and pred_b and (not pred_c or not pred_d):
        routing = "do_not_fit_build_hand_reviewed_candidate_arm_manifest_and_common_consequence_schema"
    elif not pred_a:
        routing = "instrument_repair_only_same_cutoff_and_bars"
    else:
        routing = "archive_insufficient_design_prospective_candidate_bank"
    result = {
        "status": "complete",
        "rung": 440,
        "claim_level": "research_instrument_feasibility",
        "archive_cutoff": [RUNG_MIN, RUNG_MAX],
        "feature_path": str(FEATURE_PATH.relative_to(ROOT)),
        "feature_sha256": feature_sha,
        "label_path": str(LABEL_PATH.relative_to(ROOT)),
        "label_sha256": sha256(LABEL_PATH),
        "canonical_receipts": n,
        "excluded_receipts": len(features["excluded_receipts"]),
        "ambiguous_rungs": len(features["ambiguous_rungs"]),
        "ambiguity_fraction": ambiguity_fraction,
        "source_linkage_count": source_linked,
        "source_linkage_fraction": source_linked / max(1, n),
        "chronology_complete": chronology_complete,
        "module_families": module_families,
        "grammar_families": grammar_families,
        "explicit_arm_map_count": arm_mapped,
        "explicit_arm_map_fraction": arm_mapped / max(1, n),
        "machine_readable_price_count": priced,
        "machine_readable_price_fraction": priced / max(1, n),
        "registered_predicate_labels": total_predicate_labels,
        "instrument_labels": instrument_labels,
        "consequence_category_label_counts": dict(category_counts),
        "consequence_category_cell_counts": {key: len(category_cells[key]) for key in CATEGORY_TOKENS},
        "consequence_category_pass": category_pass,
        "forbidden_feature_keys": forbidden,
        "join_keys_match": join_match,
        "pred_a_separation_and_provenance": pred_a,
        "pred_b_archive_volume": pred_b,
        "pred_c_candidate_granularity_and_price": pred_c,
        "pred_d_consequence_coverage": pred_d,
        "strong_null_archive_not_viable": strong_null,
        "routing": routing,
        "literal_deployed_model_values": 0,
        "native_model_calls": 0,
    }
    write_json(AUDIT_PATH, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features")
    labels = sub.add_parser("labels")
    labels.add_argument("--feature-sha", required=True)
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.command == "features":
        command_features()
    elif args.command == "labels":
        command_labels(args.feature_sha)
    else:
        command_audit()


if __name__ == "__main__":
    main()
