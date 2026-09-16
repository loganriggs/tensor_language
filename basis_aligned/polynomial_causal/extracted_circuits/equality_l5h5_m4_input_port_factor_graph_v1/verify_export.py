#!/usr/bin/env python3
"""Verify the equality M4 normalized-input graph and its four-trait evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
POLY = PACKAGE.parents[1]
EXPORT = POLY / "EQUALITY_L5H5_M4_INPUT_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    export = json.loads(EXPORT.read_text())
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    if export["terminal"] != "exported_equality_l5h5_m4_input_port_factor_graph_ood":
        raise ValueError("export terminal changed")
    if not all(export["four_traits"].values()) or export["four_traits"] != manifest["four_traits"]:
        raise ValueError("four-trait declaration mismatch")
    for name, digest in export["files"].items():
        if sha(PACKAGE / name) != digest:
            raise ValueError(f"package file changed: {name}")

    source_path = POLY / export["source_result"]
    if sha(source_path) != export["source_result_sha256"]:
        raise ValueError("source evidence changed")
    source = json.loads(source_path.read_text())
    expected_terminal = "equality_l5h5_m4_input_port_factor_graph_extracted_ood"
    if source["terminal"] != expected_terminal or not all(source["predictions"].values()):
        raise ValueError("source evidence no longer passes")
    predicates = source["predictions"]
    trait_evidence = {
        "ood_prediction": predicates["pred_b_ood_behavioral_equivalence"],
        "extraction": predicates["pred_e_normalized_input_extraction_boundary"],
        "selective_removal": predicates["pred_c_node_removal_equivalence"]
            and predicates["pred_d_selectivity_and_roll_equivalence"],
        "composition_reuse": predicates["pred_a_bitwise_product_corner_and_score_closure"],
    }
    if trait_evidence != export["four_traits"]:
        raise ValueError("source predicates do not entail four traits")

    source_graph = source["graph"]
    if source_graph["dynamic_input_ports"] != manifest["dynamic_input_ports"] \
            or source_graph["nodes"] != manifest["nodes"] or source_graph["learned_parameters"] != 0:
        raise ValueError("manifest/source graph mismatch")
    if manifest["external_native_activation_inputs"] != 5 \
            or manifest["deterministic_context_inputs"] != 2 \
            or manifest["native_product_inputs"] != 0 \
            or manifest["residual_corner_inputs"] != 0 \
            or manifest["raw_qk_activation_inputs"] != 0 \
            or manifest["oracle_score_inputs"] != 0:
        raise ValueError("extraction boundary changed")

    print(json.dumps({
        "terminal": "equality_l5h5_m4_input_port_four_traits_verified",
        "four_traits": trait_evidence,
        "external_native_activation_inputs": 5,
        "deterministic_context_inputs": 2,
        "package_files": len(export["files"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
