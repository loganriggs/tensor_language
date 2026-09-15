"""Exact four-source MLP17 census: earlier residual, MLP16, other attention, H17.2."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from native_suffix_mlp16_producer_v1 import digest


P = Path(__file__).resolve().parent
OUT = P / "SETTING2_EARLIER_RESIDUAL_ATTENTION_TERM_CENSUS_V1_RESULT.json"


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
    token_pairs = sorted(set((row["uk_id"], row["us_id"]) for row in rows))
    token_ids = [token for pair in token_pairs for token in pair]
    assert len(token_pairs) == 6 and len(set(token_ids)) == 12

    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
    left, right, down = [state[f"transformer.h.17.mlp.{name}.weight"].double()
                         for name in ("Left", "Right", "Down")]
    readers = state["lm_head.weight"][token_ids].double()
    out_slice = state["transformer.h.17.attn.c_proj.weight"][:, 2 * 128 : 3 * 128].double()
    pre = endpoint["ports"]["pre"].double()
    incoming = ports["ports"]["residual"].double()
    p = ports["block17_lambdas"][0].double() * ports["ports"]["mlp16_output"].double()
    a = ports["ports"]["head_values"][:, 2 * 128 : 3 * 128].double() @ out_slice.T
    sources = {"e": incoming - p, "p": p, "o": pre - incoming - a, "a": a}
    folded_down = readers @ down
    lhs = {name: value @ left.T for name, value in sources.items()}
    rhs = {name: value @ right.T for name, value in sources.items()}
    order = tuple(sources)
    terms = {}
    for i, first in enumerate(order):
        for second in order[i:]:
            label = first + second
            product = lhs[first] * rhs[second]
            if first != second:
                product = product + lhs[second] * rhs[first]
            terms[label] = product @ folded_down.T

    direct = ((pre @ left.T) * (pre @ right.T)) @ down.T @ readers.T
    summed = sum(terms.values())
    delta = direct[1::2] - direct[0::2]
    delta_terms = {name: value[1::2] - value[0::2] for name, value in terms.items()}
    reports = {}
    denom = delta.square().sum().clamp_min(1e-30)
    for name, value in delta_terms.items():
        ratio = float(value.norm() / delta.norm().clamp_min(1e-30))
        reports[name] = {
            "change_norm_ratio": ratio,
            "aligned_fraction": float((value * delta).sum() / denom),
            "live": ratio >= 0.05,
            "per_reader_norm_ratios": [float(value[:, j].norm() / delta[:, j].norm().clamp_min(1e-30))
                                         for j in range(delta.shape[1])],
        }

    prior_path = P / "SETTING2_SELECTED_READER_SIX_TERM_CENSUS_V1_RESULT.json"
    prior = json.loads(prior_path.read_text())
    g = sources["e"] + sources["o"]
    prior_sources = {"g": g, "p": sources["p"], "a": sources["a"]}
    pl = {name: value @ left.T for name, value in prior_sources.items()}
    pr = {name: value @ right.T for name, value in prior_sources.items()}
    prior_terms = {}
    for label, first, second in (("gg", "g", "g"), ("pp", "p", "p"), ("aa", "a", "a"),
                                 ("gp", "g", "p"), ("ga", "g", "a"), ("pa", "p", "a")):
        product = pl[first] * pr[second]
        if first != second:
            product = product + pl[second] * pr[first]
        prior_terms[label] = product @ folded_down.T
    aggregation = {"gg": terms["ee"] + terms["oo"] + terms["eo"],
                   "gp": terms["ep"] + terms["po"],
                   "ga": terms["ea"] + terms["oa"],
                   "pp": terms["pp"], "pa": terms["pa"], "aa": terms["aa"]}
    aggregation_errors = {name: rel(aggregation[name], prior_terms[name]) for name in prior_terms}
    ranking = sorted(reports, key=lambda name: reports[name]["change_norm_ratio"], reverse=True)
    absolute_error = rel(summed, direct)
    change_error = rel(sum(delta_terms.values()), delta)
    result = {
        "pred_a_absolute_exact": absolute_error <= 1e-10,
        "pred_b_change_exact": change_error <= 1e-10,
        "pred_c_prior_aggregation_exact": max(aggregation_errors.values()) <= 1e-10,
        "metrics": {"absolute_sum_relative_error": absolute_error,
                    "change_sum_relative_error": change_error,
                    "max_prior_aggregation_relative_error": max(aggregation_errors.values()),
                    "prior_aggregation_relative_errors": aggregation_errors,
                    "cancellation_ratio": float(delta.norm() / sum(v.norm() for v in delta_terms.values()).clamp_min(1e-30)),
                    "pair_count": len(delta), "reader_count": len(token_ids)},
        "term_reports": reports,
        "frozen_change_norm_ranking": ranking,
        "next_candidate": next((name for name in ranking if reports[name]["live"]), None),
        "shapes": {name: list(value.shape) for name, value in sources.items()},
        "token_pairs": token_pairs,
        "price": {"model_forwards": 0, "fits": 0, "explicit_terms": 10,
                  "ordered_elementwise_products": 16 * 4608,
                  "native_weights_and_source_generation_still_charged": True},
        "hashes": {"binding": digest(binding_path), "ports": digest(ports_path),
                   "endpoints": digest(endpoint_path), "rows": digest(row_path),
                   "prior_census": digest(prior_path), "script": digest(__file__)},
        "scope": "Exact cached four-source MLP17 numerator census through twelve fixed readers. Descriptive source splitting only; no normalized-logit, behavior, causal, fresh/OOD, or adoption claim.",
    }
    assert prior["pred_a_absolute_exact"] and prior["pred_b_change_exact"]
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
