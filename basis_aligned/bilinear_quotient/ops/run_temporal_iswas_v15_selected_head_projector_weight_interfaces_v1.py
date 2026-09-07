#!/usr/bin/env python3
"""Exact-weight interface diagnostic for selected v15 head-response projectors."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_exact_map_finiteness_and_price pred_b_reader_rankings_are_fold_stable pred_c_known_l15_readers_are_enriched pred_d_sources_converge_on_shared_downstream_interfaces pred_e_known_serial_value_writers_are_enriched
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import fastload
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_selected_head_projector_weight_interfaces_v1.json"
DAS = ROOT / "circuits/followups/temporal_iswas_v15_head_response_target_feasible_regularized_das_v1_result.json"
FASTLOAD = ROOT / "ops/fastload.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_selected_head_projector_weight_interfaces_v1_result.json"
CANDIDATE_ID = "temporal_iswas.v15_selected_head_projector_weight_interfaces_v1"
EXPECTED = {
    "prior": "e19a47cd65ca88459f9e5313f899ed760a97fcde691bae3b58b7648d22df2d9a",
    "das": "0e3ee2641a8f7c38ebad5cb0b73cf71461e2cd9bd96cdcb30a581e2f8d7bd30b",
    "fastload": "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
    "checkpoint": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
}
SOURCES = ((8, 1), (9, 1), (9, 4), (11, 3))
PRICE = {"model_forwards": 0, "example_evaluations": 0, "fit_updates": 0,
         "model_updates": 0, "transformer_backwards": 0, "checkpoint_loads": 1}
PREDICTION_KEYS = (
    "pred_a_authority_exact_map_finiteness_and_price",
    "pred_b_reader_rankings_are_fold_stable",
    "pred_c_known_l15_readers_are_enriched",
    "pred_d_sources_converge_on_shared_downstream_interfaces",
    "pred_e_known_serial_value_writers_are_enriched",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalized_score(torch, matrix, vector) -> float:
    denominator = float(torch.linalg.matrix_norm(matrix)) * float(vector.norm())
    return float((matrix @ vector).norm()) / denominator if denominator else 0.0


def writer_score(torch, covector, matrix) -> float:
    denominator = float(covector.norm()) * float(torch.linalg.matrix_norm(matrix))
    return float(covector @ matrix).norm() / denominator if denominator else 0.0


def ranked(rows, key="score"):
    values = sorted(rows, key=lambda row: (-row[key], row["label"]))
    denominator = max(1, len(values) - 1)
    for position, row in enumerate(values):
        row["percentile"] = (len(values) - 1 - position) / denominator
    return values


def spearman(left, right) -> float:
    labels = sorted(set(left) & set(right))
    if len(labels) < 2:
        return float("nan")
    rank_left = {row["label"]: i for i, row in enumerate(left)}
    rank_right = {row["label"]: i for i, row in enumerate(right)}
    a = [rank_left[label] for label in labels]
    b = [rank_right[label] for label in labels]
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else float("nan")


def downstream(torch, model, source_layer, write):
    interfaces, modules = [], []
    for layer in range(source_layer, 18):
        block = model.transformer.h[layer]
        if layer > source_layer:
            for head in range(9):
                factors = {}
                for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
                    matrix = getattr(block.attn, name).weight.detach().float()[head * 128:(head + 1) * 128]
                    score = normalized_score(torch, matrix, write)
                    factor = name[2:]
                    factors[factor] = score
                    interfaces.append({"label": f"L{layer}H{head}:{factor}", "score": score})
                modules.append({"label": f"L{layer}H{head}", "score": max(factors.values()),
                                "factors": factors})
        left = normalized_score(torch, block.mlp.Left.weight.detach().float(), write)
        right = normalized_score(torch, block.mlp.Right.weight.detach().float(), write)
        interfaces.extend(({"label": f"MLP{layer}:left", "score": left},
                           {"label": f"MLP{layer}:right", "score": right}))
        modules.append({"label": f"MLP{layer}", "score": max(left, right),
                        "left": left, "right": right})
    return ranked(interfaces), ranked(modules)


def upstream(torch, model, source_layer, value_covector):
    rows = []
    for layer in range(source_layer):
        block = model.transformer.h[layer]
        for head in range(9):
            matrix = block.attn.c_proj.weight.detach().float()[:, head * 128:(head + 1) * 128]
            rows.append({"label": f"L{layer}H{head}", "score": writer_score(torch, value_covector, matrix)})
        rows.append({"label": f"MLP{layer}", "score": writer_score(
            torch, value_covector, block.mlp.Down.weight.detach().float())})
    return ranked(rows)


def main():
    config, checkpoint, _module = fastload._paths()
    observed = {"prior": sha(PRIOR), "das": sha(DAS), "fastload": sha(FASTLOAD)}
    if observed != {key: EXPECTED[key] for key in observed}:
        raise RuntimeError(f"weight-interface authority changed: {observed}")
    prior, das = json.loads(PRIOR.read_text()), json.loads(DAS.read_text())
    if (prior.get("candidate_id") != CANDIDATE_ID or das.get("terminal") != "regularization_does_not_beat_dim"
            or das["selected"]["configuration"]["rank"] != 1):
        raise RuntimeError("selected projector disposition changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "sources": len(SOURCES),
              "folds": 2, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(checkpoint) != EXPECTED["checkpoint"]:
        raise RuntimeError("checkpoint weights changed")
    if config != {"vocab_size": 50304, "n_layer": 18, "n_head": 9, "n_embd": 1152,
                  "squared_mlp": False, "bilinear": True, "expansion_factor": 4,
                  "gated": False, "squared_attn": True, "bilinear_attn": True}:
        raise RuntimeError("model configuration changed")
    started_utc, started = utc_now(), time.perf_counter()
    model = fastload.load_model_fast().eval()
    import torch
    records = {}
    mapping_error = gauge_error = 0.0
    for fold in ("0", "1"):
        records[fold] = {}
        for layer, head in SOURCES:
            label = f"L{layer}H{head}"
            basis = torch.tensor(das["selected"]["projectors"][fold][label], dtype=torch.float32)
            if tuple(basis.shape) != (128, 1):
                raise RuntimeError(f"projector shape changed for {fold}/{label}")
            unit = basis[:, 0]
            output = model.transformer.h[layer].attn.c_proj.weight.detach().float()
            write = output[:, head * 128:(head + 1) * 128] @ unit
            embedded = torch.zeros(1152)
            embedded[head * 128:(head + 1) * 128] = unit
            mapping_error = max(mapping_error, float((write - output @ embedded).abs().max()))
            value = model.transformer.h[layer].attn.c_v.weight.detach().float()[head * 128:(head + 1) * 128]
            value_covector = value.T @ unit
            interfaces, modules = downstream(torch, model, layer, write)
            writers = upstream(torch, model, layer, value_covector)
            neg_interfaces, neg_modules = downstream(torch, model, layer, -write)
            neg_writers = upstream(torch, model, layer, -value_covector)
            gauge_error = max(gauge_error,
                max(abs(a["score"] - b["score"]) for a, b in zip(interfaces, neg_interfaces)),
                max(abs(a["score"] - b["score"]) for a, b in zip(modules, neg_modules)),
                max((abs(a["score"] - b["score"]) for a, b in zip(writers, neg_writers)), default=0.0))
            records[fold][label] = {"write_norm": float(write.norm()),
                "value_pullback_norm": float(value_covector.norm()),
                "downstream_interfaces": interfaces, "downstream_modules": modules,
                "upstream_value_writers": writers}
    fold_stability = {source: spearman(records["0"][source]["downstream_modules"],
                                       records["1"][source]["downstream_modules"])
                      for source in records["0"]}
    known_reader = {}
    shared = {}
    for fold in ("0", "1"):
        known_reader[fold] = {}
        tops = []
        for source, record in records[fold].items():
            by_label = {row["label"]: row for row in record["downstream_modules"]}
            known_reader[fold][source] = {label: by_label[label]["percentile"]
                                          for label in ("L15H1", "L15H5") if label in by_label}
            tops.extend(row["label"] for row in record["downstream_interfaces"][:10])
        counts = Counter(tops)
        shared[fold] = sorted(({"label": label, "source_top10_count": count}
                               for label, count in counts.items() if count >= 2),
                              key=lambda row: (-row["source_top10_count"], row["label"]))
    known_writer = {}
    for fold in ("0", "1"):
        known_writer[fold] = {}
        for source in ("L9H1", "L9H4", "L11H3"):
            by_label = {row["label"]: row for row in records[fold][source]["upstream_value_writers"]}
            labels = ("L8H1",) if source.startswith("L9") else ("L9H1", "L9H4")
            known_writer[fold][source] = {label: by_label[label]["percentile"] for label in labels}
    finite = all(math.isfinite(value) for value in [mapping_error, gauge_error, *fold_stability.values()])
    pred_a = observed == {key: EXPECTED[key] for key in observed} and mapping_error <= 1e-7 \
        and gauge_error <= 1e-12 and finite
    pred_b = all(value >= .80 for value in fold_stability.values())
    pred_c = all(sum(max(scores.values(), default=0.0) >= .75 for scores in known_reader[fold].values()) >= 3
                 for fold in ("0", "1"))
    pred_d = all(any(row["source_top10_count"] >= 3 for row in shared[fold]) for fold in ("0", "1"))
    pred_e = all(
        max(known_writer[fold]["L9H1"]["L8H1"], known_writer[fold]["L9H4"]["L8H1"]) >= .75
        and max(known_writer[fold]["L11H3"].values()) >= .75 for fold in ("0", "1"))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (pred_a, pred_b, pred_c, pred_d, pred_e))))
    terminal = "invalid" if not pred_a else "shared_weight_interface_diagnostic" if all(predictions.values()) \
        else "diffuse_or_unstable_weight_interfaces"
    result = {"schema": "temporal_iswas_v15_selected_head_projector_weight_interfaces_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"wo_embedding_max_abs_error": mapping_error,
            "sign_gauge_score_max_abs_error": gauge_error, "finite": finite},
        "fold_stability": fold_stability, "known_reader_percentiles": known_reader,
        "shared_top10_interfaces": shared, "known_value_writer_percentiles": known_writer,
        "records": records, "scope": "diagnostic_weight_geometry_not_causal_identification",
        "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("instrument", "fold_stability",
        "known_reader_percentiles", "shared_top10_interfaces", "known_value_writer_percentiles",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
