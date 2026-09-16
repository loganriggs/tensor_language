#!/usr/bin/env python3
"""Verify package hashes and four-trait evidence for the exported graph."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
POLY = PACKAGE.parents[1]
EXPORT = POLY / "SUBJECT_NUMBER_SPARSE_GRAPH_EXPORT_V1_RESULT.json"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    export = json.loads(EXPORT.read_text()); manifest = json.loads((PACKAGE / "manifest.json").read_text())
    if export["terminal"] != "exported_subject_number_sparse_graph_four_traits" \
            or not all(export["four_traits"].values()) or export["four_traits"] != manifest["four_traits"]:
        raise ValueError("four-trait export status failed")
    for name, digest in export["files"].items():
        if sha(PACKAGE / name) != digest: raise ValueError(f"package file changed: {name}")
    for claim, evidence in export["evidence"].items():
        path = POLY / evidence["file"]
        if sha(path) != evidence["sha256"]: raise ValueError(f"evidence changed: {claim}")
        result = json.loads(path.read_text())
        if result["terminal"] != evidence["terminal"] or not all(result["predictions"].values()):
            raise ValueError(f"evidence no longer passes: {claim}")
    extraction = json.loads((POLY / export["evidence"]["token_extraction"]["file"]).read_text())
    if extraction["component"]["masks"] != [edge["mask"] for edge in manifest["edges"]] \
            or extraction["component"]["terms"] != [edge["term"] for edge in manifest["edges"]] \
            or extraction["component"]["external_activation_inputs"] != 0:
        raise ValueError("manifest/executor mismatch")
    print(json.dumps({"terminal": "subject_number_sparse_graph_export_verified",
                      "package_files": len(export["files"]), "evidence_files": len(export["evidence"]),
                      "four_traits": export["four_traits"]}, indent=2, sort_keys=True))

if __name__ == "__main__": main()
