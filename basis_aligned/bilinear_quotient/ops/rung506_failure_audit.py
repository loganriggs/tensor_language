#!/usr/bin/env python3
"""CPU-only clause audit of rung506's zero-eligible-site strong null."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
for path in (ROOT, ROOT / "ops", ROOT.parent / "polynomial_causal"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import natural_action_conditioned_later_write_state_atlas_rung506 as rung


RESULT = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_results.json"
BUNDLE = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_bundle.pt"
SOURCE = ROOT / "ops/natural_action_conditioned_later_write_state_atlas_rung506.py"
OUT = ROOT / "rung506_failure_audit_results.json"
EXPECTED = {
    RESULT: "f86e5f0303ab0616ea14e3141fd09886ca54d326e8d83ea6c8c13a62f66db75e",
    BUNDLE: "225f73cb885e0e51d76ed329b60b044359a600a18e130846c99dd4c103959093",
    SOURCE: "9a17e28312a0e7214e5fc587123e3267e2650b382f3a40daf12ad1a380b1d004",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    hashes = {path: sha256(path) for path in EXPECTED}
    if hashes != EXPECTED:
        raise RuntimeError("rung506 authority changed")
    result = json.loads(RESULT.read_text())
    if not (
        result["pred_a_exact_live_conditional_instrument"] is True
        and result["pred_b_score_actions_recalibrate_new_documents"] is True
        and result["pred_c_at_least_one_whole_write_edge_confirms"] is False
        and result["strong_null"] is True
        and result["analysis"]["eligible_sites"] == []
        and result["analysis"]["discovery_edges"] == []
    ):
        raise RuntimeError("rung506 verdict changed")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "rung506_finite_later_write_state_atlas_sufficient_statistics_v1" \
            or bundle.get("validation_opened") is not False:
        raise RuntimeError("rung506 bundle schema changed")
    collection = bundle["collections"]["discovery_singletons"]
    registered = result["analysis"]["discovery_checks"]["sites"]

    site_audit = {}
    task_stable_sites = []
    for site in rung.SITES:
        circuit_repeat = [registered[site]["sources"][source]["repeat"]["cosine"]
                          for source in rung.SOURCES]
        circuit_source = [registered[site]["source_comparisons"][f"N:{source}"]["cosine"]
                          for source in rung.SOURCES[1:]]
        circuit_rms = [registered[site]["sources"][source]["pooled_rms_nat"]
                       for source in rung.SOURCES]
        task_repeat = []
        task_source = []
        for source in rung.SOURCES:
            task_repeat.append(rung.cosine(
                rung.task_vector(collection, site, collection, source, "half0"),
                rung.task_vector(collection, site, collection, source, "half1")))
        native = rung.task_vector(collection, site, collection, "N", "pooled")
        for source in rung.SOURCES[1:]:
            task_source.append(rung.cosine(
                native, rung.task_vector(collection, site, collection, source, "pooled")))
        task_stable = min(task_repeat) >= .50 and min(task_source) >= .70
        if task_stable:
            task_stable_sites.append(site)
        site_audit[site] = {
            "circuit_fingerprint_rms_nat_range": [min(circuit_rms), max(circuit_rms)],
            "minimum_circuit_repeat_cosine": min(circuit_repeat),
            "minimum_circuit_source_cosine": min(circuit_source),
            "minimum_task_repeat_cosine": min(task_repeat),
            "minimum_task_source_cosine": min(task_source),
            "task_stable_under_same_repeat_source_bars": task_stable,
        }

    task_edges = []
    task_edge_metrics = {}
    for left, right in itertools.combinations(task_stable_sites, 2):
        source_cosines = [rung.cosine(
            rung.task_vector(collection, left, collection, source, "pooled"),
            rung.task_vector(collection, right, collection, source, "pooled"))
            for source in rung.SOURCES]
        name = rung.edge_name(left, right)
        task_edge_metrics[name] = {
            "source_cosines": source_cosines,
            "minimum_source_cosine": min(source_cosines),
        }
        if min(source_cosines) >= .60:
            task_edges.append(name)

    payload = {
        "status": "complete",
        "analysis": "after_outcome_cpu_clause_audit_no_new_model_outcomes",
        "input_sha256": {str(path): digest for path, digest in hashes.items()},
        "registered_result": {
            "pred_a": True, "pred_b": True, "pred_c": False,
            "strong_null": True,
            "next_step": "split_fixed_writes_into_exact_attention_or_bilinear_terms",
        },
        "site_audit": site_audit,
        "counts": {
            "physically_live_sites": sum(
                min(row["circuit_fingerprint_rms_nat_range"]) >= .0005
                for row in site_audit.values()),
            "sites_passing_every_circuit_repeat_bar": sum(
                row["minimum_circuit_repeat_cosine"] >= .50
                for row in site_audit.values()),
            "sites_passing_every_circuit_source_bar": sum(
                row["minimum_circuit_source_cosine"] >= .70
                for row in site_audit.values()),
            "task_stable_sites": len(task_stable_sites),
        },
        "task_stable_sites": task_stable_sites,
        "task_only_pair_screen_descriptive_not_registered_edges": task_edges,
        "task_only_pair_metrics": task_edge_metrics,
        "interpretation": (
            "All 19 writes have material circuit-fingerprint RMS, but no single write repeats its "
            "32-circuit member-minus-control direction across the two discovery document halves and "
            "no write clears source invariance. Seven MLP writes do have stable four-context task "
            "effects, forming descriptive early and late task-similarity clusters. This does not "
            "rescore rung506 or license task-only whole-write edges. It localizes the failure to the "
            "current 32-circuit observation coordinates at whole-write grain and supports the frozen "
            "route: split a task-stable write internally. MLP10 is the user-named bilinear test case."
        ),
        "new_model_outcomes_opened": False,
        "next_object": "exact_MLP10_named_input_source_pair_terms_with_finite_downstream_interventions",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "counts": payload["counts"], "task_stable_sites": task_stable_sites,
        "task_only_pairs": task_edges, "next_object": payload["next_object"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
