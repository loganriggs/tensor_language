"""Empirical prefix/continuation Hankel-rank audit for bilin18.

Forms cross-products of prefix and suffix fragments. Two scalar series are scored:
the model log-probability of the suffix and the pre-softcap question-channel score
after the suffix. Matrix completion holds out 20% of prefix/suffix pairs.

Registered predictions:
  A. Some rank <= 8 reduces held-out RMSE by >= 30% versus row+column effects.
  B. The selected rank95 agrees within 2 between two disjoint row ranges.
  C. The question matrix has lower rank95 than the full suffix-logprob matrix.

Cross-concatenated sequences are counterfactual. Their CE is reported as an OOD
diagnostic, and no language-model fidelity claim is made if it exceeds natural CE by
more than 1 nat/token.
"""

import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
QK = HERE.parent / "qk_mdl"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(QK))

from data import fineweb_rows
from hankel import complete_low_rank, heldout_rmse, row_column_baseline, spectrum
from tier2_model import load_elriggs, reference_forward

PREFIXES = 48
SUFFIXES = 48
PREFIX_LEN = 64
SUFFIX_LEN = 8
BATCH = 8
SPLITS = {"discovery": 7000, "heldout": 11000}
OUT = HERE / "hankel_rank_audit_results.json"
DEV = "cuda"


def make_grid(rows):
    prefixes = rows[:PREFIXES, :PREFIX_LEN]
    suffixes = rows[PREFIXES:PREFIXES + SUFFIXES,
                    PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
    sequences = torch.cat([
        torch.cat([prefixes[i].expand(SUFFIXES, -1), suffixes], dim=1)
        for i in range(PREFIXES)
    ])
    return sequences


@torch.no_grad()
def score_grid(model, sequences, question_tokens):
    sequence_score = []
    question_score = []
    splice_ce = []
    for start in range(0, len(sequences), BATCH):
        seq = sequences[start:start + BATCH].to(DEV)
        capped = reference_forward(model, seq[:, :-1])
        logp = capped.float().log_softmax(-1)
        targets = seq[:, 1:]
        token_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        suffix_lp = token_lp[:, PREFIX_LEN - 1:].sum(-1)
        sequence_score.append(suffix_lp.cpu())
        splice_ce.append((-token_lp[:, PREFIX_LEN - 1:]).mean(-1).cpu())
        pre_softcap = 30.0 * torch.atanh((capped.float() / 30.0).clamp(-0.999999, 0.999999))
        question_score.append(pre_softcap[:, -1, question_tokens].mean(-1).cpu())
    return {
        "suffix_logprob": torch.cat(sequence_score).view(PREFIXES, SUFFIXES),
        "question_raw": torch.cat(question_score).view(PREFIXES, SUFFIXES),
        "splice_ce": float(torch.cat(splice_ce).mean()),
    }


@torch.no_grad()
def natural_suffix_ce(model, rows):
    values = []
    sequences = rows[:PREFIXES, :PREFIX_LEN + SUFFIX_LEN]
    for start in range(0, len(sequences), BATCH):
        seq = sequences[start:start + BATCH].to(DEV)
        capped = reference_forward(model, seq[:, :-1])
        logp = capped.float().log_softmax(-1)
        token_lp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
        values.append((-token_lp[:, PREFIX_LEN - 1:]).mean(-1).cpu())
    return float(torch.cat(values).mean())


def matrix_report(matrix, seed):
    gen = torch.Generator().manual_seed(seed)
    observed = torch.rand(matrix.shape, generator=gen) > 0.20
    baseline = row_column_baseline(matrix, observed)
    baseline_rmse = heldout_rmse(baseline, matrix, observed)
    curves = {}
    for rank in (1, 2, 4, 8, 16):
        completed = complete_low_rank(matrix, observed, rank)
        curves[str(rank)] = heldout_rmse(completed, matrix, observed)
    spec = spectrum(matrix)
    return {"rank90": spec["rank90"], "rank95": spec["rank95"],
            "stable_rank": spec["stable_rank"],
            "top_singular_values": [float(x) for x in spec["singular_values"][:16]],
            "row_column_rmse": baseline_rmse, "rank_rmse": curves,
            "best_improvement": 1 - min(curves.values()) / baseline_rmse}


@torch.no_grad()
def main():
    started = time.time()
    model, _ = load_elriggs("bilin18")
    encoding = __import__("tiktoken").get_encoding("gpt2")
    question_tokens = [token for token in range(50257)
                       if re.match(r"^\?$| \?$", encoding.decode([token]))]
    output = {"config": {"prefixes": PREFIXES, "suffixes": SUFFIXES,
                          "prefix_len": PREFIX_LEN, "suffix_len": SUFFIX_LEN}}
    for split_index, (name, skip) in enumerate(SPLITS.items()):
        rows = fineweb_rows(PREFIXES + SUFFIXES, skip=skip)
        scored = score_grid(model, make_grid(rows), question_tokens)
        natural_ce = natural_suffix_ce(model, rows)
        output[name] = {"splice_ce": scored["splice_ce"],
                        "natural_ce": natural_ce,
                        "splice_ce_excess": scored["splice_ce"] - natural_ce,
                        "suffix_logprob": matrix_report(scored["suffix_logprob"], 10 + split_index),
                        "question_raw": matrix_report(scored["question_raw"], 20 + split_index)}

    improvements = [output[split][metric]["best_improvement"]
                    for split in SPLITS for metric in ("suffix_logprob", "question_raw")]
    pred_a = max(improvements) >= 0.30
    pred_b = all(abs(output["discovery"][metric]["rank95"]
                     - output["heldout"][metric]["rank95"]) <= 2
                 for metric in ("suffix_logprob", "question_raw"))
    pred_c = output["discovery"]["question_raw"]["rank95"] \
        < output["discovery"]["suffix_logprob"]["rank95"]
    output["predictions"] = {"low_rank_beats_additive": pred_a,
                             "rank_replicates": pred_b,
                             "question_lower_rank": pred_c}
    output["runtime_s"] = time.time() - started
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["predictions"], indent=2), flush=True)
    print(f"wrote {OUT} in {output['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
