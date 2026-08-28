#!/usr/bin/env python3
"""Compare signed-square and product codecs on frozen native MLP4 prefixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from . import mlp4_bilinear_residual_codec as product_codec
from . import mlp4_signed_square_codec as square_codec
from . import bilinear_tensor_invariants as invariants

HERE = Path(__file__).resolve().parent
BYTES = HERE / "mlp4_z4_candidate_bytes.pt"
INVENTORY = HERE / "mlp4_z4_candidate_inventory.json"
OUTPUT_BYTES = HERE / "mlp4_signed_square_candidate_bytes.pt"
OUTPUT = HERE / "mlp4_signed_square_audit.json"
SOURCES = ("mlp4.rmsnorm_input",)
WIDTHS = (1152,)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    artifact = torch.load(BYTES, map_location="cpu", weights_only=False)
    inventory = json.loads(INVENTORY.read_text())
    encoded = {}; rows = []
    for source in inventory["candidates"]:
        if source["family"] != "native_product":
            continue
        candidate_id = source["candidate_id"]
        product_bytes = artifact["encoded"][candidate_id]
        decoded = product_codec.decode(product_bytes)
        square_bytes, price = square_codec.encode(
            decoded["A"], decoded["B"], decoded["C"], decoded["bias"],
            source["step"], SOURCES, WIDTHS)
        encoded[candidate_id] = square_bytes
        square = square_codec.decode(square_bytes)
        square_A = torch.cat((square["U"], square["V"]), 1)
        square_B = square_A
        square_C = torch.cat((square["C"], -square["C"]), 0)
        relative_error = invariants.relative_tensor_frobenius_error(
            decoded["A"], decoded["B"], decoded["C"],
            square_A, square_B, square_C)
        absolute_error = invariants.tensor_frobenius_error(
            decoded["A"], decoded["B"], decoded["C"],
            square_A, square_B, square_C)
        composition_bound = invariants.rms_sphere_residual_lipschitz_bound(
            decoded["A"], decoded["B"], decoded["C"],
            square_A, square_B, square_C)
        product_bits = source["conditional_known_gauge_bits"]
        rows.append({"candidate_id": candidate_id, "components": source["capacity"],
                     "product_codec_bits": product_bits,
                     "signed_square_codec_bits": price["conditional_known_gauge_bits"],
                     "signed_square_minus_product_bits":
                         price["conditional_known_gauge_bits"]-product_bits,
                     "signed_square_ratio":
                         price["conditional_known_gauge_bits"]/product_bits,
                     "relative_coefficient_tensor_frobenius_error": relative_error,
                     "coefficient_tensor_frobenius_error": absolute_error,
                     "rms_sphere_residual_lipschitz_upper_bound": composition_bound,
                     "product_hash": source["canonical_bytes_hash"],
                     "signed_square_hash": price["canonical_bytes_hash"]})
    torch.save({"schema_version": 1, "source_candidate_bytes_sha256": sha(BYTES),
                "encoded": encoded}, OUTPUT_BYTES)
    result = {
        "schema_version": 1,
        "audit_id": "bilin18.mlp4.native-signed-square-codec.v1",
        "source_candidate_bytes_sha256": sha(BYTES),
        "source_inventory_sha256": sha(INVENTORY),
        "signed_square_candidate_bytes": OUTPUT_BYTES.name,
        "signed_square_candidate_bytes_sha256": sha(OUTPUT_BYTES),
        "same_quantization_step_as_frozen_product_codec": True,
        "distortion_metric": "exact relative Frobenius error of quadratic coefficient tensor",
        "composition_bound": "||delta_error|| <= rms_sphere_residual_lipschitz_upper_bound * ||delta_z||",
        "behavioral_roster_changed": False,
        "validation_opened": False,
        "global_cp_nonuniqueness_quotiented": False,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    for row in rows:
        print(row["candidate_id"], row["product_codec_bits"],
              row["signed_square_codec_bits"],
              f"ratio={row['signed_square_ratio']:.4f}")


if __name__ == "__main__":
    main()
