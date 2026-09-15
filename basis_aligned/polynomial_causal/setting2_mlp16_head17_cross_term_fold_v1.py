"""No-fit exact fold of MLP16 x head17.2 through MLP17 and 12 token readers."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from native_suffix_mlp16_producer_v1 import digest


P = Path(__file__).resolve().parent
OUT = P / "SETTING2_MLP16_HEAD17_CROSS_TERM_FOLD_V1_RESULT.json"


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(2)
    assert not OUT.exists()
    binding_path = P / "NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json"
    binding = json.loads(binding_path.read_text())["files"]
    assert all(digest(path) == expected for path, expected in binding.items())
    checkpoint = next(path for path in binding if path.endswith("/pytorch_model.bin"))

    ports_path = P / "NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt"
    upstream_path = P / "NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json"
    upstream = json.loads(upstream_path.read_text())
    assert digest(ports_path) == upstream["cache_sha256"]
    ports = torch.load(ports_path, weights_only=True, map_location="cpu")
    endpoint_path = P / "NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt"
    endpoints = torch.load(endpoint_path, weights_only=True, map_location="cpu")

    row_path = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
    rows = json.loads(row_path.read_text())["rows"]
    pairs = sorted(set((row["uk_id"], row["us_id"]) for row in rows))
    ids = [token_id for pair in pairs for token_id in pair]
    assert len(pairs) == 6 and len(set(ids)) == 12

    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
    left, right, down = [
        state[f"transformer.h.17.mlp.{name}.weight"].double()
        for name in ("Left", "Right", "Down")
    ]
    readers = state["lm_head.weight"][ids].double()
    output_slice = state["transformer.h.17.attn.c_proj.weight"][:, 2 * 128 : 3 * 128].double()

    producer = ports["block17_lambdas"][0].double() * ports["ports"]["mlp16_output"].double()
    head_value = ports["ports"]["head_values"][:, 2 * 128 : 3 * 128].double()
    attention = head_value @ output_slice.T
    pre = endpoints["ports"]["pre"].double()

    folded_down = readers @ down
    lp, rp = producer @ left.T, producer @ right.T
    la, ra = attention @ left.T, attention @ right.T
    ordered_pa = (lp * ra) @ folded_down.T
    ordered_ap = (la * rp) @ folded_down.T
    folded = ordered_pa + ordered_ap

    def q(x: torch.Tensor) -> torch.Tensor:
        return ((x @ left.T) * (x @ right.T)) @ down.T

    direct = (q(producer + attention) - q(producer) - q(attention) + q(torch.zeros_like(producer))) @ readers.T
    exact_error = relative_error(folded, direct)
    full = q(pre) @ readers.T
    live_fraction = float(folded.square().mean().sqrt() / full.square().mean().sqrt().clamp_min(1e-30))
    pa_rms = float(ordered_pa.square().mean().sqrt())
    ap_rms = float(ordered_ap.square().mean().sqrt())
    ordering_ratio = min(pa_rms, ap_rms) / max(pa_rms, ap_rms, 1e-30)
    cancellation_ratio = float(folded.norm() / (ordered_pa.norm() + ordered_ap.norm()).clamp_min(1e-30))

    direct_tensor_scalars = 12 * 1152 * 128
    folded_materialized_scalars = 12 * 4608 + 2 * 4608 * 128
    result = {
        "pred_a_exact_instrument": exact_error <= 1e-10,
        "pred_b_live_selected_interaction": live_fraction >= 0.02,
        "pred_c_both_orderings_matter": ordering_ratio >= 0.05,
        "pred_d_conditional_representation_smaller": folded_materialized_scalars < direct_tensor_scalars,
        "metrics": {
            "exact_relative_error": exact_error,
            "cross_term_to_full_reader_rms": live_fraction,
            "ordered_p_attention_rms": pa_rms,
            "ordered_attention_p_rms": ap_rms,
            "ordered_smaller_to_larger_rms": ordering_ratio,
            "sum_to_sum_of_norms": cancellation_ratio,
            "endpoint_count": len(pre),
            "reader_count": len(ids),
        },
        "price": {
            "direct_mixed_tensor_scalars": direct_tensor_scalars,
            "conditional_materialized_fold_scalars": folded_materialized_scalars,
            "conditional_fraction": folded_materialized_scalars / direct_tensor_scalars,
            "folded_down_scalars": 12 * 4608,
            "two_head_projection_compositions_scalars": 2 * 4608 * 128,
            "native_left_right_down_output_and_reader_weights_still_charged": True,
            "model_forwards": 0,
            "fits": 0,
        },
        "shapes": {
            "producer": list(producer.shape),
            "head_value": list(head_value.shape),
            "attention_write": list(attention.shape),
            "folded_down": list(folded_down.shape),
            "selected_output": list(folded.shape),
        },
        "token_pairs": pairs,
        "hashes": {
            "binding": digest(binding_path),
            "ports": digest(ports_path),
            "endpoints": digest(endpoint_path),
            "rows": digest(row_path),
            "script": digest(__file__),
        },
        "scope": "Exact cached native MLP16 x head17.2 numerator term through twelve token readers. Conditional price only; other paths, normalizers, bias, direct residual, final RMS/softcap, causal identification, fresh/OOD transfer, and full-model savings remain unclaimed.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
