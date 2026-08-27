#!/usr/bin/env python3
"""CPU-only audit of the generic conditions for the shared V/O gauge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPOSITORY = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
REVISION = "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
EXPECTED_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
OUTPUT = HERE / "shared_value_gauge_audit_results.json"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = Path(hf_hub_download(REPOSITORY, "pytorch_model.bin",
                                revision=REVISION, local_files_only=True))
    if file_hash(path) != EXPECTED_SHA256:
        raise ValueError("checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    mixing = [float(state[f"transformer.h.{layer}.attn.lamb"])
              for layer in range(18)]
    values = state["transformer.h.0.attn.c_v.weight"].double().view(9, 128, 1152)
    heads = []
    for head, value in enumerate(values):
        singular = torch.linalg.svdvals(value)
        heads.append({"head": head, "minimum_singular_value": float(singular[-1]),
                      "maximum_singular_value": float(singular[0]),
                      "relative_minimum_singular_value": float(singular[-1]/singular[0]),
                      "rank_at_relative_tolerance_1e_7":
                          int((singular > 1e-7*singular[0]).sum())})
    result = {"schema_version": 1, "audit_id": "bilin18.shared-value-gauge-genericity.v1",
              "repository": REPOSITORY, "revision": REVISION,
              "checkpoint_sha256": EXPECTED_SHA256,
              "attention_value_mixing_coefficients": mixing,
              "all_layers_have_nonzero_shared_value_edge": all(value != 0 for value in mixing),
              "layer0_shared_value_heads": heads,
              "all_shared_value_heads_full_row_rank":
                  all(row["rank_at_relative_tolerance_1e_7"] == 128 for row in heads),
              "minimum_relative_singular_value": min(
                  row["relative_minimum_singular_value"] for row in heads),
              "conclusion": "one GL(128) value/output gauge per head is tied across all 18 layers"}
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(f"nonzero edges={result['all_layers_have_nonzero_shared_value_edge']}; "
          f"full rank={result['all_shared_value_heads_full_row_rank']}; "
          f"min relative singular={result['minimum_relative_singular_value']:.6f}")


if __name__ == "__main__":
    main()
