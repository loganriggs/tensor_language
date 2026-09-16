#!/usr/bin/env python3
"""Rebuild and audit the extracted-circuit graph registry."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basis_aligned.polynomial_causal import build_circuit_graph_registry_v1 as builder

HERE = Path(__file__).resolve().parent
OUT = HERE / "CIRCUIT_GRAPH_REGISTRY_V1_RESULT.json"
REQUIRED_VERIFIED = {
    "subject_number_l11h3_sparse_graph_v1",
    "equality_l5h5_m4_input_port_factor_graph_v1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rebuilt = builder.build_registry(run_verifiers=True)
    stored = json.loads(builder.JSON_OUT.read_text())
    markdown = builder.render_markdown(rebuilt)
    package_dirs = {path.name for path in builder.PACKAGES.iterdir() if path.is_dir()}
    stored_packages = {record["package"] for record in stored["packages"]}
    unknowns_preserved = all(
        value in (True, False, None)
        for record in rebuilt["packages"] for value in record["four_traits"].values()
    ) and all(
        record["four_traits"][trait] is None
        for record in rebuilt["packages"] for trait in builder.TRAITS
        if trait not in (json.loads((builder.ROOT / record["manifest"]).read_text())
                         if record["manifest"] else {}).get("four_traits", {})
    )
    predictions = {
        "pred_a_complete_directory_inventory": package_dirs == stored_packages
            and rebuilt["counts"]["packages"] == len(package_dirs),
        "pred_b_generated_artifacts_reproduce": rebuilt == stored
            and builder.MARKDOWN_OUT.read_text() == markdown,
        "pred_c_two_four_trait_verifiers_pass": REQUIRED_VERIFIED <= set(rebuilt["verified_four_trait_packages"])
            and rebuilt["counts"]["four_trait_declared_unverified"] == 0,
        "pred_d_dependency_graph_is_acyclic": rebuilt["counts"]["dependency_cycles"] == 0,
        "pred_e_missing_evidence_remains_unknown": unknowns_preserved,
    }
    result = {
        "schema": "circuit_graph_registry_v1_result",
        "terminal": "audited_circuit_graph_registry" if all(predictions.values()) else "invalid",
        "predictions": predictions,
        "counts": rebuilt["counts"],
        "verified_four_trait_packages": rebuilt["verified_four_trait_packages"],
        "files": {
            "builder": {"path": str((HERE / "build_circuit_graph_registry_v1.py").relative_to(ROOT)),
                        "sha256": sha(HERE / "build_circuit_graph_registry_v1.py")},
            "tests": {"path": str((HERE / "test_build_circuit_graph_registry_v1.py").relative_to(ROOT)),
                      "sha256": sha(HERE / "test_build_circuit_graph_registry_v1.py")},
            "registry_json": {"path": str(builder.JSON_OUT.relative_to(ROOT)), "sha256": sha(builder.JSON_OUT)},
            "registry_markdown": {"path": str(builder.MARKDOWN_OUT.relative_to(ROOT)), "sha256": sha(builder.MARKDOWN_OUT)},
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(Path(__file__))},
        },
        "scope": "Filesystem-complete inventory and live verifier audit of extracted-circuit packages. Missing metadata is unknown; only packages with all four declared traits and a passing hash/evidence verifier are certified.",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(predictions.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
