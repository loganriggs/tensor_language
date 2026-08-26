"""Question-channel causal polynomial with a held-out joint-intervention test.

Three grouped cuts are varied: W=top writer outputs, R=all downstream reader
inputs, F=the final residual channel. Discovery sees masks 000..110. Mask 111 is
the preregistered held-out joint. Additive and pairwise models fitted on discovery
effects predict the 111 effect on a disjoint row range.

Registered predictions:
  A. In the clean-frozen RMS arm, pairwise normalized error on the unseen 111
     true-target raw-logit effect is <= .25 and <= .70 x additive error.
  B. Pairwise prediction improves at least 25% when clean RMS gauges are frozen
     versus recomputed live.
  C. At least two-thirds of discovery interaction coefficients with magnitude
     >= .05 replicate in sign and retain >= half magnitude on the held-out rows.

This script is GPU-ready but is not placed on a runner queue automatically.
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
QK = HERE.parent / "qk_mdl"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(QK))

from data import fineweb_rows
from mobius import fit_effect_model, mobius_coefficients, normalized_error, predict_effect
from tier2_model import load_elriggs

D = 1152
T = 256
NH = 9
HD = 128
NL = 18
SITE = 11
BATCH = 8
FIT_ROWS = 96
EVAL_ROWS = 240
DISCOVERY_SKIP = 7000
HELDOUT_SKIP = 11000
WRITERS = {"attn9", "attn10", "mlp9", "mlp10"}
READERS = {f"attn{i}" for i in range(SITE, NL)} | {
    f"mlp{i}" for i in range(SITE, NL)
}
OUT = HERE / "question_channel_ledger_results.json"

DEV = "cuda"
ENC = tiktoken.get_encoding("gpt2")
MODEL = None
CFG = None
BLOCKS = None
ARE = None

V2 = None
MEANS = {}
MASK_V = None


def token_mask(pattern):
    result = torch.zeros(50257, dtype=torch.bool)
    for token in range(50257):
        if re.match(pattern, ENC.decode([token])):
            result[token] = True
    return result


def rms(x, key, mode, gauges):
    if mode == "live":
        return F.rms_norm(x, (x.shape[-1],))
    if mode == "record":
        eps = torch.finfo(x.dtype).eps
        gauges[key] = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        return F.rms_norm(x, (x.shape[-1],))
    if mode == "frozen":
        return x * gauges[key].to(dtype=x.dtype)
    raise ValueError(f"unknown RMS mode {mode}")


def edit_span(x, mean, strength):
    if strength == 0:
        return x
    xf = x.float()
    coeff = xf @ V2
    return (xf - strength * (coeff - mean) @ V2.T).to(x.dtype)


def add_projection(acc, name, value):
    flat = value.float().reshape(-1, D)
    acc[name]["sum"] += (flat @ V2).sum(0)
    acc[name]["n"] += flat.shape[0]


@torch.no_grad()
def forward(idx, strengths=(0.0, 0.0, 0.0), rms_mode="live", gauges=None,
            collect=None):
    """Manual exact forward. strengths=(writer, reader, final) cut fractions."""
    if gauges is None:
        gauges = {}
    writer_strength, reader_strength, final_strength = strengths
    batch, length = idx.shape
    x = rms(MODEL.transformer.wte(idx), "embed", rms_mode, gauges)
    x0 = x
    v1 = None
    causal = torch.tril(torch.ones(length, length, device=DEV, dtype=torch.bool))

    for layer, block in enumerate(BLOCKS):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attn = block.attn
        h = rms(x, f"attn_in_{layer}", rms_mode, gauges)
        if collect is not None and f"attn{layer}" in READERS:
            add_projection(collect, f"attn{layer}", h)
        if reader_strength and f"attn{layer}" in READERS:
            h = edit_span(h, MEANS[f"attn{layer}"], reader_strength)

        def qk(linear, branch):
            pre = linear(h).view(batch, length, NH, HD)
            z = rms(pre, f"{branch}_{layer}", rms_mode, gauges)
            cos, sin = attn.rotary(pre)
            return ARE(z, cos, sin)

        q = qk(attn.c_q, "q")
        k = qk(attn.c_k, "k")
        q2 = qk(attn.c_q2, "q2")
        k2 = qk(attn.c_k2, "k2")
        score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / HD
        score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HD
        pattern = (score1 * score2).masked_fill(~causal, 0.0)
        value = attn.c_v(h).view(batch, length, NH, HD)
        if v1 is None:
            v1 = value
        value = (1 - attn.lamb) * value + attn.lamb * v1.view_as(value)
        mixed = torch.einsum("bhqk,bkhd->bqhd", pattern.to(value.dtype), value)
        attn_out = attn.c_proj(mixed.reshape(batch, length, D))
        name = f"attn{layer}"
        if collect is not None and name in WRITERS:
            add_projection(collect, name, attn_out)
        if writer_strength and name in WRITERS:
            attn_out = edit_span(attn_out, MEANS[name], writer_strength)
        x = x + attn_out

        z = rms(x, f"mlp_in_{layer}", rms_mode, gauges)
        if collect is not None and f"mlp{layer}" in READERS:
            add_projection(collect, f"mlp{layer}", z)
        if reader_strength and f"mlp{layer}" in READERS:
            z = edit_span(z, MEANS[f"mlp{layer}"], reader_strength)
        mlp_out = block.mlp(z)
        name = f"mlp{layer}"
        if collect is not None and name in WRITERS:
            add_projection(collect, name, mlp_out)
        if writer_strength and name in WRITERS:
            mlp_out = edit_span(mlp_out, MEANS[name], writer_strength)
        x = x + mlp_out

    if collect is not None:
        add_projection(collect, "final", x)
    if final_strength:
        x = edit_span(x, MEANS["final"], final_strength)
    final = rms(x, "final", rms_mode, gauges)
    raw = MODEL.lm_head(final)
    return raw, 30.0 * torch.tanh(raw / 30.0), gauges


def build_slice():
    global V2
    unembed = MODEL.lm_head.weight.float().to(DEV)[:50257]
    direction = unembed[MASK_V.to(DEV)].mean(0)
    direction = direction / direction.norm()
    mlp = BLOCKS[SITE].mlp
    output_weight = direction @ mlp.Down.weight.float()
    qform = mlp.Left.weight.float().T @ (
        output_weight[:, None] * mlp.Right.weight.float()
    )
    symmetric = 0.5 * (qform + qform.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    order = eigenvalues.abs().argsort(descending=True)[:2]
    V2 = eigenvectors[:, order].contiguous()
    return eigenvalues[order]


@torch.no_grad()
def fit_means(rows):
    global MEANS
    names = sorted(WRITERS | READERS | {"final"})
    acc = {name: {"sum": torch.zeros(2, device=DEV), "n": 0} for name in names}
    for start in range(0, len(rows), BATCH):
        forward(rows[start:start + BATCH, :-1].to(DEV), collect=acc)
    MEANS = {name: item["sum"] / item["n"] for name, item in acc.items()}


def empty_arm():
    return defaultdict(float)


def accumulate(arm, raw, capped, clean_capped, targets):
    ce = F.cross_entropy(capped.reshape(-1, capped.shape[-1]), targets.reshape(-1),
                         reduction="none").view_as(targets)
    clean_prob = clean_capped.float().softmax(-1)
    kl = (clean_prob * (clean_capped.float().log_softmax(-1)
                        - capped.float().log_softmax(-1))).sum(-1)
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, :64] = False
    question = MASK_V.to(DEV)[targets] & valid
    background = valid & ~question
    true_raw = raw.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    for name, tensor, mask in (
        ("question_ce", ce, question),
        ("background_ce", ce, background),
        ("all_ce", ce, valid),
        ("question_true_raw", true_raw, question),
        ("all_kl", kl, valid),
    ):
        arm[name + "_sum"] += float(tensor[mask].sum())
        arm[name + "_n"] += int(mask.sum())


@torch.no_grad()
def evaluate(rows, gauge_mode):
    totals = {mask: empty_arm() for mask in range(8)}
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH].to(DEV)
        idx, targets = batch[:, :-1], batch[:, 1:]
        if gauge_mode == "frozen":
            clean_raw, clean_capped, gauges = forward(idx, rms_mode="record")
        else:
            clean_raw, clean_capped, gauges = forward(idx, rms_mode="live")
        accumulate(totals[0], clean_raw, clean_capped, clean_capped, targets)
        for mask in range(1, 8):
            strengths = tuple(float((mask >> bit) & 1) for bit in range(3))
            raw, capped, _ = forward(idx, strengths, gauge_mode, gauges)
            accumulate(totals[mask], raw, capped, clean_capped, targets)
    result = {}
    for mask, arm in totals.items():
        result[mask] = {
            key[:-4]: arm[key] / max(arm[key[:-4] + "_n"], 1)
            for key in arm if key.endswith("_sum")
        }
    return result


def effect_table(arms, metric):
    base = arms[0][metric]
    return {mask: arms[mask][metric] - base for mask in range(8)}


def compare_models(discovery, heldout, metric):
    disc_effects = effect_table(discovery, metric)
    held_effects = effect_table(heldout, metric)
    alpha = lambda mask: tuple((mask >> bit) & 1 for bit in range(3))
    additive_masks = (0, 1, 2, 4)
    pairwise_masks = tuple(range(7))
    additive = fit_effect_model([(alpha(m), disc_effects[m]) for m in additive_masks], 1)
    pairwise = fit_effect_model([(alpha(m), disc_effects[m]) for m in pairwise_masks], 2)
    additive_pred = predict_effect(additive, (1, 1, 1))
    pairwise_pred = predict_effect(pairwise, (1, 1, 1))
    actual = held_effects[7]
    disc_mobius = mobius_coefficients(disc_effects, 3)
    held_mobius = mobius_coefficients(held_effects, 3)
    interactions = []
    for mask in (3, 5, 6, 7):
        a, b = disc_mobius[mask], held_mobius[mask]
        if abs(a) >= 0.05:
            interactions.append({"mask": mask, "discovery": a, "heldout": b,
                                 "replicates": a * b > 0 and abs(b) >= 0.5 * abs(a)})
    return {
        "actual_heldout_111_effect": actual,
        "additive_prediction": additive_pred,
        "pairwise_prediction": pairwise_pred,
        "additive_normalized_error": normalized_error(additive_pred, actual),
        "pairwise_normalized_error": normalized_error(pairwise_pred, actual),
        "interaction_replication": interactions,
    }


@torch.no_grad()
def main():
    global ARE, BLOCKS, CFG, MASK_V, MODEL
    started = time.time()
    MODEL, CFG = load_elriggs("bilin18")
    BLOCKS = MODEL.transformer.h
    ARE = sys.modules[type(BLOCKS[0].attn).__module__].apply_rotary_emb
    MASK_V = token_mask(r"^\?$| \?$")
    eigenvalues = build_slice()
    fit_rows = fineweb_rows(FIT_ROWS, skip=80)[:, :T + 1].contiguous()
    discovery_rows = fineweb_rows(EVAL_ROWS, skip=DISCOVERY_SKIP)[:, :T + 1].contiguous()
    heldout_rows = fineweb_rows(EVAL_ROWS, skip=HELDOUT_SKIP)[:, :T + 1].contiguous()
    fit_means(fit_rows)

    output = {"config": {"fit_rows": FIT_ROWS, "eval_rows_per_split": EVAL_ROWS,
                          "discovery_skip": DISCOVERY_SKIP, "heldout_skip": HELDOUT_SKIP,
                          "variables": ["writer_cut", "reader_cut", "final_cut"]},
              "slice_eigenvalues": [float(x) for x in eigenvalues]}
    for mode in ("live", "frozen"):
        discovery = evaluate(discovery_rows, mode)
        heldout = evaluate(heldout_rows, mode)
        output[mode] = {
            "discovery": {str(k): v for k, v in discovery.items()},
            "heldout": {str(k): v for k, v in heldout.items()},
            "comparisons": {metric: compare_models(discovery, heldout, metric)
                            for metric in ("question_true_raw", "question_ce",
                                           "background_ce", "all_kl")},
        }

    frozen = output["frozen"]["comparisons"]["question_true_raw"]
    live = output["live"]["comparisons"]["question_true_raw"]
    pred_a = (frozen["pairwise_normalized_error"] <= 0.25
              and frozen["pairwise_normalized_error"]
              <= 0.70 * frozen["additive_normalized_error"])
    pred_b = (frozen["pairwise_normalized_error"]
              <= 0.75 * live["pairwise_normalized_error"])
    reps = frozen["interaction_replication"]
    pred_c = bool(reps) and sum(x["replicates"] for x in reps) / len(reps) >= 2 / 3
    output["predictions"] = {"pairwise_beats_additive": pred_a,
                             "frozen_beats_live": pred_b,
                             "interactions_replicate": pred_c}
    output["runtime_s"] = time.time() - started
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["predictions"], indent=2), flush=True)
    print(f"wrote {OUT} in {output['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
