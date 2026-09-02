#!/usr/bin/env python3
"""CPU-only comparison of rung 466 code and rung 505 natural causal vectors."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
CODE = ROOT / "equality_correction_group_factorial_rung466_results.json"
NATURAL = ROOT / "equality_score_gauged_downstream_program_rung505_results.json"
OUT = ROOT / "rung505_cross_corpus_component_audit_results.json"
EXPECTED = {
    "code": "d04acf3637834830f8ee7bd73eaa8a6c435386816ef54fce1d8451b0597132fe",
    "natural": "3720a2feb24fc5ec4554d858a00a576a1fcd44f0e789d2b728e66483d7d8d1a1",
}
MASKS = {"T": "7", "G": "24", "ALL": "31"}
SITES = {"m8": "1", "m9": "2", "m12": "4", "a14": "8", "m17": "16"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    hashes = {"code": sha256(CODE), "natural": sha256(NATURAL)}
    if hashes != EXPECTED:
        raise RuntimeError(f"input hash mismatch: {hashes}")

    code = json.loads(CODE.read_text())["analysis"]["pooled"]
    natural = json.loads(NATURAL.read_text())["analysis"]["pooled"]
    code_vectors = code["subset_vectors"]
    natural_vectors = natural["subset_vectors"]
    code_interactions = code["cross_group_interaction_vectors"]
    natural_interactions = natural["interaction_vectors"]

    component_cosines: dict[str, dict[str, float]] = {}
    for code_source in ("N", "H"):
        for natural_source in ("N", "P", "Z7", "Z8"):
            pair = f"{code_source}:{natural_source}"
            component_cosines[pair] = {
                name: cosine(
                    code_vectors[code_source][mask], natural_vectors[natural_source][mask]
                )
                for name, mask in MASKS.items()
            }
            component_cosines[pair]["I"] = cosine(
                code_interactions[code_source], natural_interactions[natural_source]
            )

    site_cosines: dict[str, dict[str, float]] = {}
    for code_source in ("N", "H"):
        for natural_source in ("N", "P", "Z7", "Z8"):
            pair = f"{code_source}:{natural_source}"
            site_cosines[pair] = {
                site: cosine(
                    code_vectors[code_source][mask], natural_vectors[natural_source][mask]
                )
                for site, mask in SITES.items()
            }

    def ranges(table: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        names = next(iter(table.values())).keys()
        return {
            name: {
                "minimum": min(row[name] for row in table.values()),
                "maximum": max(row[name] for row in table.values()),
            }
            for name in names
        }

    result = {
        "status": "complete",
        "analysis": "after_outcome_cpu_cross_corpus_component_audit_no_new_model_outcomes",
        "input_sha256": hashes,
        "component_cosines": component_cosines,
        "component_cosine_ranges": ranges(component_cosines),
        "site_cosines": site_cosines,
        "site_cosine_ranges": ranges(site_cosines),
        "interpretation": (
            "The fixed task group T and the five-site union do not preserve their causal context "
            "directions from code to natural text. G and the T-by-G finite interaction are much "
            "more stable, but this is an after-outcome component audit and does not rescue the "
            "registered cross-corpus program claim. The next prospective object should be defined "
            "by natural-text causal responses rather than by the code-selected five-site grouping."
        ),
        "new_model_outcomes_opened": False,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "component_cosine_ranges": result["component_cosine_ranges"],
        "site_cosine_ranges": result["site_cosine_ranges"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
