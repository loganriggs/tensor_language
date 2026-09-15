"""Exact no-fit six-term MLP17 source census through 12 token readers."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from native_suffix_mlp16_producer_v1 import digest


P = Path(__file__).resolve().parent
OUT = P / "SETTING2_SELECTED_READER_SIX_TERM_CENSUS_V1_RESULT.json"


def rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
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
    upstream = json.loads((P / "NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json").read_text())
    assert digest(ports_path) == upstream["cache_sha256"]
    ports = torch.load(ports_path, weights_only=True, map_location="cpu")
    endpoint_path = P / "NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt"
    endpoint = torch.load(endpoint_path, weights_only=True, map_location="cpu")
    row_path = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
    rows = json.loads(row_path.read_text())["rows"]
    pairs = sorted(set((row["uk_id"], row["us_id"]) for row in rows))
    token_ids = [token for pair in pairs for token in pair]
    assert len(pairs) == 6 and len(set(token_ids)) == 12

    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
    left, right, down = [
        state[f"transformer.h.17.mlp.{name}.weight"].double()
        for name in ("Left", "Right", "Down")
    ]
    readers = state["lm_head.weight"][token_ids].double()
    out_slice = state["transformer.h.17.attn.c_proj.weight"][:, 2 * 128 : 3 * 128].double()
    pre = endpoint["ports"]["pre"].double()
    p = ports["block17_lambdas"][0].double() * ports["ports"]["mlp16_output"].double()
    h = ports["ports"]["head_values"][:, 2 * 128 : 3 * 128].double()
    a = h @ out_slice.T
    g = pre - p - a
    sources = {"g": g, "p": p, "a": a}
    folded_down = readers @ down

    l = {name: value @ left.T for name, value in sources.items()}
    r = {name: value @ right.T for name, value in sources.items()}
    terms = {}
    names = (("gg", "g", "g"), ("pp", "p", "p"), ("aa", "a", "a"),
             ("gp", "g", "p"), ("ga", "g", "a"), ("pa", "p", "a"))
    for label, first, second in names:
        product = l[first] * r[second]
        if first != second:
            product = product + l[second] * r[first]
        terms[label] = product @ folded_down.T

    direct = ((pre @ left.T) * (pre @ right.T)) @ down.T @ readers.T
    summed = sum(terms.values())
    delta = direct[1::2] - direct[0::2]
    delta_terms = {name: value[1::2] - value[0::2] for name, value in terms.items()}
    delta_sum = sum(delta_terms.values())
    denom = delta.square().sum().clamp_min(1e-30)
    reports = {}
    for name, value in delta_terms.items():
        ratio = float(value.norm() / delta.norm().clamp_min(1e-30))
        reports[name] = {
            "change_rms": float(value.square().mean().sqrt()),
            "change_norm_ratio": ratio,
            "aligned_fraction": float((value * delta).sum() / denom),
            "live": ratio >= 0.05,
            "per_reader_norm_ratios": [
                float(value[:, j].norm() / delta[:, j].norm().clamp_min(1e-30))
                for j in range(delta.shape[1])
            ],
        }
    ranking = sorted(reports, key=lambda name: reports[name]["change_norm_ratio"], reverse=True)
    next_candidates = [name for name in ranking if name != "pa" and reports[name]["live"]]
    absolute_error = rel(summed, direct)
    change_error = rel(delta_sum, delta)
    result = {
        "pred_a_absolute_exact": absolute_error <= 1e-10,
        "pred_b_change_exact": change_error <= 1e-10,
        "metrics": {
            "absolute_sum_relative_error": absolute_error,
            "change_sum_relative_error": change_error,
            "full_change_rms": float(delta.square().mean().sqrt()),
            "cancellation_ratio": float(delta.norm() / sum(v.norm() for v in delta_terms.values()).clamp_min(1e-30)),
            "pair_count": len(delta),
            "reader_count": len(token_ids),
        },
        "term_reports": reports,
        "frozen_change_norm_ranking": ranking,
        "next_candidate": next_candidates[0] if next_candidates else None,
        "shapes": {name: list(value.shape) for name, value in sources.items()},
        "token_pairs": pairs,
        "price": {"model_forwards": 0, "fits": 0, "explicit_terms": 6,
                  "ordered_elementwise_products": 9 * 4608,
                  "native_weights_and_all_omitted_background_still_charged": True},
        "hashes": {"binding": digest(binding_path), "ports": digest(ports_path),
                   "endpoints": digest(endpoint_path), "rows": digest(row_path),
                   "script": digest(__file__)},
        "scope": "Exact cached MLP17 numerator census for background g, scaled MLP16 p, and head17.2 a through twelve fixed token readers. Descriptive term selection only; no normalized logit, behavior, causal, fresh/OOD, or compression-adoption claim.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
