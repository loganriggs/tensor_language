"""RUNG 300 -- JOINT FACTORIZATION OF UNTIED INPUT/OUTPUT VOCABULARY MAPS.

Question
--------
Can the stored input embedding E and output head U share one token-indexed code,
rather than paying for two independent 50,304 x 1,152 matrices?

The executable shared-code family keeps E exactly (it remains the live input
embedding) and predicts the output head by

    U_hat_s = E M + (U - E M) V_s V_s^T,

where M is the full-population ridge least-squares map from E to U and V_s are
the top right singular vectors of its residual.  Deployment stores E, M,
P_s=(U-EM)V_s, and V_s.  Logits are evaluated without materializing U_hat:

    h U_hat^T = (h M^T) E^T + (h V_s) P_s^T.

This is compared to the price-matched independent output SVD

    U_ind_r = (U Q_r) Q_r^T,

while E is again exact.  The independent rank r is the greatest rank whose
literal price does not exceed the corresponding shared-code arm.  Therefore a
shared win cannot be attributed to a larger scalar budget.

Frozen arms and populations
---------------------------
Residual ranks s in {0,128,256,512}; every arm at or below the exact 25%-saving
vocabulary ceiling is scored.  Predictive populations are 16 frozen FineWeb
rows from `.rowcache/fineweb_n96_skip1200.pt` and 16 deterministic WikiText-2
raw-test rows after token skip 1024.  No text, token frequency, hidden state, or
label enters either factor fit.  All 256 next-token positions per row are
scored.  A separate frozen FineWeb fit cache supplies token-frequency bins only
for post-fit diagnostics (unseen, count 1-2, 3-9, >=10).

Frozen predictions
------------------
pred_a_shared_code_predictive:
    Some shared arm saving at least 25% of the native vocabulary storage has
    FineWeb CE damage <=0.05 and WikiText CE damage <=0.08.
pred_b_shared_beats_price_matched_independent:
    At some common arm, shared-code CE damage is at least 20% smaller than its
    independent baseline on BOTH FineWeb and WikiText, with nonnegative
    baseline damage on both populations.
pred_c_rare_token_stability:
    For the best FineWeb shared arm satisfying the 25%-saving price, mean CE
    damage on unseen-or-count<=2 targets is at most twice the damage on
    count>=10 targets plus 0.02 nats, on both corpora.

Null / decision
---------------
The route is a predictive null if every 25%-saving shared arm damages FineWeb
by >=0.20 nats, or if shared never beats its price-matched independent arm on
both corpora.  Weight reconstruction is reported but cannot identify a useful
program by itself.  A positive screen is not adoption: it still needs the full
census, certificates, composition, and interventions.

Literal price
-------------
Native vocabulary storage is 2*50304*1152 = 115,900,416 scalars.  Shared rank s:
50304*1152 + 1152^2 + s*(50304+1152).  Independent rank r:
50304*1152 + r*(50304+1152).  Padding rows are retained and priced because they
participate in the output normalization even though labels are <50,257.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "joint_vocab_shared_code_screen_results.json"
DEV = "cuda"
D = 1152
W = 50304
V_REAL = 50257
RANKS = (0, 128, 256, 512)
N_ROWS = 16
RIDGE_REL = 1e-6


def _load_rows(path: Path, n: int = N_ROWS) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    rows = value["rows"] if isinstance(value, dict) else value
    assert rows.ndim == 2 and rows.shape[1] >= 257
    return rows[:n, :257].long().contiguous()


def _wikitext_rows(n: int = N_ROWS, width: int = 257, skip: int = 1024) -> tuple[torch.Tensor, str]:
    from datasets import load_dataset
    import tiktoken

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tiktoken.get_encoding("gpt2").encode_ordinary(text)
    stop = skip + n * width
    assert len(tokens) >= stop
    return torch.tensor(tokens[skip:stop], dtype=torch.long).reshape(n, width), str(dataset._fingerprint)


@torch.no_grad()
def _capture_hidden(model: torch.nn.Module, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_cpu = []
    target_cpu = []
    for start in range(0, len(rows), 2):
        batch = rows[start:start + 2]
        index = batch[:, :-1].to(DEV)
        target = batch[:, 1:].contiguous()
        x = F.rms_norm(model.transformer.wte(index), (D,))
        x0 = x
        value0 = None
        for block in model.transformer.h:
            x, value0 = block(x, value0, x0)
        hidden_cpu.append(F.rms_norm(x, (D,)).half().cpu().reshape(-1, D))
        target_cpu.append(target.reshape(-1))
    return torch.cat(hidden_cpu), torch.cat(target_cpu)


def _frequency_codes(target: torch.Tensor, fit_counts: torch.Tensor) -> torch.Tensor:
    count = fit_counts[target]
    return torch.where(
        count == 0,
        0,
        torch.where(count <= 2, 1, torch.where(count <= 9, 2, 3)),
    )


@torch.no_grad()
def _score_logits(
    hidden_cpu: torch.Tensor,
    target_cpu: torch.Tensor,
    fit_counts: torch.Tensor,
    logit_function,
) -> dict[str, object]:
    total_loss = 0.0
    total_count = 0
    by_bin_sum = torch.zeros(4, dtype=torch.float64)
    by_bin_count = torch.zeros(4, dtype=torch.long)
    for start in range(0, len(hidden_cpu), 512):
        hidden = hidden_cpu[start:start + 512].float().to(DEV)
        target = target_cpu[start:start + 512].to(DEV)
        logits = 30.0 * torch.tanh(logit_function(hidden) / 30.0)
        loss = F.cross_entropy(logits.float(), target, reduction="none").double().cpu()
        bins = _frequency_codes(target_cpu[start:start + len(loss)], fit_counts)
        total_loss += float(loss.sum())
        total_count += len(loss)
        for code in range(4):
            mask = bins == code
            by_bin_sum[code] += loss[mask].sum()
            by_bin_count[code] += int(mask.sum())
    labels = ("unseen", "count_1_2", "count_3_9", "count_ge_10")
    return {
        "ce": total_loss / total_count,
        "n_positions": total_count,
        "ce_by_frequency": {
            labels[code]: {
                "ce": float(by_bin_sum[code] / max(int(by_bin_count[code]), 1)),
                "n": int(by_bin_count[code]),
            }
            for code in range(4)
        },
    }


def _damage_by_frequency(candidate: dict[str, object], native: dict[str, object]) -> dict[str, object]:
    result = {}
    for label, value in candidate["ce_by_frequency"].items():
        result[label] = {
            "damage": value["ce"] - native["ce_by_frequency"][label]["ce"],
            "n": value["n"],
        }
    return result


@torch.no_grad()
def _right_eigensystem(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gram = matrix.T @ matrix
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    return values[order].clamp_min(0), vectors[:, order]


def _relative_frobenius_residual(values: torch.Tensor, rank: int) -> float:
    if rank == 0:
        return 1.0
    return float(values[rank:].sum() / values.sum().clamp_min(1e-20))


def _rare_stable(score: dict[str, object]) -> bool:
    frequency = score["damage_by_frequency"]
    rare_values = []
    rare_counts = []
    for label in ("unseen", "count_1_2"):
        rare_values.append(frequency[label]["damage"])
        rare_counts.append(frequency[label]["n"])
    rare = sum(value * count for value, count in zip(rare_values, rare_counts)) / max(sum(rare_counts), 1)
    common = frequency["count_ge_10"]["damage"]
    return rare <= 2.0 * max(common, 0.0) + 0.02


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n96_skip1200.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n96_skip80.pt").exists()
        print("JOINT VOCAB SHARED CODE SCREEN | dry run: rows, prices, baselines, and bars valid")
        return
    started = time.time()
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/qk_mdl")
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    embedding = model.transformer.wte.weight.detach().float().contiguous()
    output = model.lm_head.weight.detach().float().contiguous()
    assert embedding.shape == output.shape == (W, D)
    assert cfg["n_embd"] == D

    native_price = 2 * W * D
    price_ceiling = math.floor(0.75 * native_price)
    max_shared_rank_25 = math.floor((price_ceiling - W * D - D * D) / (W + D))
    price = {
        "native_vocab_scalars": native_price,
        "price_ceiling_25pct_saving": price_ceiling,
        "max_shared_residual_rank_at_ceiling": max_shared_rank_25,
        "shared_formula": "50304*1152 + 1152^2 + s*(50304+1152)",
        "independent_formula": "50304*1152 + r*(50304+1152)",
    }

    gram_e = embedding.T @ embedding
    ridge = RIDGE_REL * float(torch.trace(gram_e)) / D
    cross = embedding.T @ output
    mapping = torch.linalg.solve(gram_e + ridge * torch.eye(D, device=DEV), cross)
    residual = output - embedding @ mapping
    residual_values, residual_vectors = _right_eigensystem(residual)
    output_values, output_vectors = _right_eigensystem(output)
    residual_total = float(residual.square().sum())
    output_total = float(output.square().sum())
    base_r2 = 1.0 - residual_total / output_total

    fineweb = _load_rows(ROOT / ".rowcache/fineweb_n96_skip1200.pt")
    wikitext, wikitext_fingerprint = _wikitext_rows()
    fit_rows = _load_rows(ROOT / ".rowcache/fineweb_n96_skip80.pt", 96)
    fit_counts = torch.bincount(fit_rows[:, 1:].reshape(-1), minlength=V_REAL)
    fine_hidden, fine_target = _capture_hidden(model, fineweb)
    wiki_hidden, wiki_target = _capture_hidden(model, wikitext)

    native_fn = lambda hidden: hidden @ output.T
    native_fine = _score_logits(fine_hidden, fine_target, fit_counts, native_fn)
    native_wiki = _score_logits(wiki_hidden, wiki_target, fit_counts, native_fn)
    arms = {}
    for shared_rank in RANKS:
        shared_price = W * D + D * D + shared_rank * (W + D)
        independent_rank = min(D, math.floor((shared_price - W * D) / (W + D)))
        residual_basis = residual_vectors[:, :shared_rank]
        residual_code = residual @ residual_basis
        independent_basis = output_vectors[:, :independent_rank]
        independent_code = output @ independent_basis

        def shared_fn(hidden: torch.Tensor) -> torch.Tensor:
            logits = (hidden @ mapping.T) @ embedding.T
            if shared_rank:
                logits = logits + (hidden @ residual_basis) @ residual_code.T
            return logits

        def independent_fn(hidden: torch.Tensor) -> torch.Tensor:
            return (hidden @ independent_basis) @ independent_code.T

        shared_fine = _score_logits(fine_hidden, fine_target, fit_counts, shared_fn)
        shared_wiki = _score_logits(wiki_hidden, wiki_target, fit_counts, shared_fn)
        independent_fine = _score_logits(fine_hidden, fine_target, fit_counts, independent_fn)
        independent_wiki = _score_logits(wiki_hidden, wiki_target, fit_counts, independent_fn)
        for candidate, native in ((shared_fine, native_fine), (shared_wiki, native_wiki),
                                  (independent_fine, native_fine), (independent_wiki, native_wiki)):
            candidate["damage"] = candidate["ce"] - native["ce"]
            candidate["damage_by_frequency"] = _damage_by_frequency(candidate, native)
        arms[str(shared_rank)] = {
            "shared_residual_rank": shared_rank,
            "independent_matched_rank": independent_rank,
            "shared_price_scalars": shared_price,
            "independent_price_scalars": W * D + independent_rank * (W + D),
            "shared_storage_fraction_native_vocab": shared_price / native_price,
            "shared_output_weight_relative_mse": float(
                residual_values[shared_rank:].sum() / output_total
            ),
            "independent_output_weight_relative_mse": _relative_frobenius_residual(
                output_values, independent_rank
            ),
            "shared_fineweb": shared_fine,
            "shared_wikitext": shared_wiki,
            "independent_fineweb": independent_fine,
            "independent_wikitext": independent_wiki,
        }
        print(
            f"s={shared_rank} r_ind={independent_rank} price={shared_price/native_price:.3f} | "
            f"FW shared/ind {shared_fine['damage']:+.4f}/{independent_fine['damage']:+.4f} | "
            f"WT shared/ind {shared_wiki['damage']:+.4f}/{independent_wiki['damage']:+.4f}",
            flush=True,
        )

    eligible = [arm for arm in arms.values() if arm["shared_price_scalars"] <= price_ceiling]
    pred_a = any(
        arm["shared_fineweb"]["damage"] <= 0.05
        and arm["shared_wikitext"]["damage"] <= 0.08
        for arm in eligible
    )
    pred_b = any(
        arm["independent_fineweb"]["damage"] >= 0
        and arm["independent_wikitext"]["damage"] >= 0
        and arm["shared_fineweb"]["damage"] <= 0.8 * arm["independent_fineweb"]["damage"]
        and arm["shared_wikitext"]["damage"] <= 0.8 * arm["independent_wikitext"]["damage"]
        for arm in eligible
    )
    best = min(eligible, key=lambda arm: arm["shared_fineweb"]["damage"])
    pred_c = bool(
        _rare_stable(best["shared_fineweb"])
        and _rare_stable(best["shared_wikitext"])
    )
    null = bool(
        all(arm["shared_fineweb"]["damage"] >= 0.20 for arm in eligible)
        or not any(
            arm["shared_fineweb"]["damage"] < arm["independent_fineweb"]["damage"]
            and arm["shared_wikitext"]["damage"] < arm["independent_wikitext"]["damage"]
            for arm in eligible
        )
    )
    result = {
        "status": "joint_vocab_shared_code_screen_complete",
        "rung": 300,
        "claim_level": "two_corpus_predictive_screen_only",
        "price": price,
        "fit": {
            "ridge_relative": RIDGE_REL,
            "ridge_absolute": ridge,
            "full_map_output_weight_r2": base_r2,
            "fit_uses_text_or_labels": False,
        },
        "populations": {
            "fineweb_rows": len(fineweb),
            "wikitext_rows": len(wikitext),
            "positions_each": int(fine_target.numel()),
            "wikitext_fingerprint": wikitext_fingerprint,
            "frequency_fit_positions": int(fit_rows[:, 1:].numel()),
        },
        "native": {"fineweb": native_fine, "wikitext": native_wiki},
        "arms": arms,
        "best_eligible_shared_rank_by_fineweb": best["shared_residual_rank"],
        'pred_a_shared_code_predictive': bool(pred_a),
        'pred_b_shared_beats_price_matched_independent': bool(pred_b),
        'pred_c_rare_token_stability': bool(pred_c),
        "null_predictive_or_price_matched": null,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "price": price,
        "full_map_output_weight_r2": base_r2,
        "native_ce": {"fineweb": native_fine["ce"], "wikitext": native_wiki["ce"]},
        "best_rank": best["shared_residual_rank"],
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("JOINT VOCAB SHARED CODE SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
