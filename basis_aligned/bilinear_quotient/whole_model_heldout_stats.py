"""Pure, CPU-only statistics for the whole-model held-out replication."""

from collections import defaultdict

import torch


def pooled_ce(loss_sums, counts, indices=None):
    """Token-weighted CE, optionally after a document-bootstrap row resample."""
    losses = torch.as_tensor(loss_sums, dtype=torch.float64)
    weights = torch.as_tensor(counts, dtype=torch.float64)
    if indices is not None:
        losses = losses[indices]
        weights = weights[indices]
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("covered-token count must be finite and positive")
    value = losses.sum() / denominator
    if not torch.isfinite(value):
        raise ValueError("pooled CE must be finite")
    return value


def ceiling(constant, live, arm):
    stake = constant - live
    if not torch.isfinite(stake) or stake <= 0:
        raise ValueError("joint constant stake must be finite and positive")
    return (constant - arm) / stake


def arm_gains(values):
    return {
        "attn": values["attn_upgraded"] - values["simple"],
        "mlp": values["mlp_upgraded"] - values["simple"],
        "both": values["both"] - values["simple"],
    }


def gain_structure_holds(gains, reference_joint_gain=0.0410):
    """Prospective gate: real positive gains, ordering, additivity, and retention."""
    return (
        gains["attn"] > 0
        and gains["mlp"] > 0
        and gains["both"] > 0
        and gains["attn"] > gains["mlp"]
        and gains["both"] > max(gains["attn"], gains["mlp"])
        and abs((gains["attn"] + gains["mlp"]) - gains["both"]) <= 0.01
        and gains["both"] >= 0.5 * reference_joint_gain
    )


def document_cluster_bootstrap(records, document_ids, draws=2000, seed=1699):
    """Paired bootstrap of ceilings/gains at the source-document cluster unit.

    ``records`` maps arm name to row-level ``loss_sums`` and ``counts``.  Rows from
    the same FineWeb document are always resampled together.  This preserves the
    pooled-token estimand while respecting within-document dependence.
    """
    if not records or set(records) < {"live", "constant", "simple",
                                     "attn_upgraded", "mlp_upgraded", "both"}:
        raise ValueError("all live/constant/program records are required")
    n_rows = len(document_ids)
    if n_rows == 0 or any(len(v["loss_sums"]) != n_rows or len(v["counts"]) != n_rows
                          for v in records.values()):
        raise ValueError("document ids and row statistics must align")

    grouped = defaultdict(list)
    for row, document_id in enumerate(document_ids):
        grouped[document_id].append(row)
    documents = sorted(grouped)
    groups = [torch.tensor(grouped[d], dtype=torch.long) for d in documents]
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(len(groups), (draws, len(groups)), generator=generator)
    metrics = {name: [] for name in ("simple", "attn_upgraded", "mlp_upgraded", "both")}
    gain_samples = {name: [] for name in (
        "attn", "mlp", "both", "both_minus_attn", "both_minus_mlp", "interaction"
    )}

    for sample in samples:
        indices = torch.cat([groups[int(i)] for i in sample])
        ces = {name: pooled_ce(v["loss_sums"], v["counts"], indices)
               for name, v in records.items()}
        vals = {name: ceiling(ces["constant"], ces["live"], ces[name])
                for name in metrics}
        gains = arm_gains(vals)
        for name, value in vals.items():
            metrics[name].append(value)
        for name in ("attn", "mlp", "both"):
            gain_samples[name].append(gains[name])
        gain_samples["both_minus_attn"].append(vals["both"] - vals["attn_upgraded"])
        gain_samples["both_minus_mlp"].append(vals["both"] - vals["mlp_upgraded"])
        gain_samples["interaction"].append(
            gains["both"] - gains["attn"] - gains["mlp"]
        )

    def summarize(values):
        values = torch.stack(values)
        return {
            "mean": float(values.mean()),
            "ci95": [float(torch.quantile(values, q)) for q in (0.025, 0.975)],
        }

    return {
        "unit": "FineWeb source document",
        "n_documents": len(documents),
        "n_rows": n_rows,
        "draws": draws,
        "seed": seed,
        "ceilings": {name: summarize(values) for name, values in metrics.items()},
        "gains": {name: summarize(values) for name, values in gain_samples.items()},
    }
