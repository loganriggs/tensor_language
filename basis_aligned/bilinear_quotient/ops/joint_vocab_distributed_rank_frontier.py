"""RUNG 305 -- DISTRIBUTED-RANK FRONTIER FOR THE SHARED VOCABULARY CODE.

Rung 304 showed that 1,129 exact rare rows repair only 2--4% of aggregate
damage: the frequency-weighted shared-code tail is distributed.  This final
vocabulary exploit tests the corresponding distributed remedy.

For weights w_t in {sqrt(count_t+1), count_t+1}, fit the shared family

    U_hat = E M_w + P_s V_s^T,  s in {512,640,768},

on the same 480 frozen FineWeb fit rows.  At each price compare against an
identically weighted independent output SVD at rank r=s+25, the greatest rank
not exceeding the shared price.  Rank-768 still saves 14.758% of native
vocabulary storage; rank-640 saves 20.441%.

Prospective evaluation is 32 FineWeb skip11000 rows and 32 WikiText-2 test rows
after token skip20000.  These populations did not select the metric, ranks, or
bars.  No sparse exception rows are retained.

Frozen predictions
------------------
pred_a_distributed_rank_is_predictive:
    Some shared arm saving >=14.5% of vocabulary has FineWeb damage <=.10 and
    WikiText damage <=.12.
pred_b_shared_beats_matched_independent:
    The same arm has >=30% lower damage than its nonnegative, slightly cheaper
    matched independent arm on BOTH corpora.
pred_c_tail_is_predictive:
    The same arm has unseen-target damage <=.50 on BOTH corpora and count>=10
    damage <=.03 on BOTH.

Null: every eligible shared arm has FineWeb damage >=.18, or no eligible shared
arm beats its matched independent control on both corpora.  A pass is only a
frontier screen; census, certificates, standalone bill, composition, OOD, and
signed intervention gates remain mandatory.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "joint_vocab_distributed_rank_frontier_results.json"
DEV = "cuda"
D = 1152
W = 50304
V_REAL = 50257
RANKS = (512, 640, 768)
RANK_OFFSET = D * D // (W + D)
FIT_ROWS = 480
EVAL_ROWS = 32
WIKI_SKIP = 20000
RIDGE_REL = 1e-6


def _fit_full(embedding: torch.Tensor, output: torch.Tensor,
              weight: torch.Tensor) -> dict[str, torch.Tensor | float]:
    sqrt_weight = weight.sqrt()[:, None]
    weighted_embedding = embedding * sqrt_weight
    weighted_output = output * sqrt_weight
    gram = weighted_embedding.T @ weighted_embedding
    ridge = RIDGE_REL * float(torch.trace(gram)) / D
    mapping = torch.linalg.solve(
        gram + ridge * torch.eye(D, device=DEV), weighted_embedding.T @ weighted_output)
    residual = output - embedding @ mapping
    residual_gram = (residual * sqrt_weight).T @ (residual * sqrt_weight)
    residual_values, residual_vectors = torch.linalg.eigh(residual_gram)
    residual_vectors = residual_vectors[:, torch.argsort(residual_values, descending=True)]
    output_gram = weighted_output.T @ weighted_output
    output_values, output_vectors = torch.linalg.eigh(output_gram)
    output_vectors = output_vectors[:, torch.argsort(output_values, descending=True)]
    return {"mapping": mapping, "residual": residual,
            "residual_vectors": residual_vectors, "output_vectors": output_vectors,
            "ridge": ridge}


def _eligible(price: int, native: int) -> bool:
    return 1.0 - price / native >= 0.145


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n480_skip80.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert RANK_OFFSET == 25
        native = 2 * W * D
        for rank in RANKS:
            shared = W * D + D * D + rank * (W + D)
            independent = W * D + (rank + RANK_OFFSET) * (W + D)
            assert independent <= shared < independent + (W + D)
        assert _eligible(W * D + D * D + max(RANKS) * (W + D), native)
        print("JOINT VOCAB DISTRIBUTED RANK FRONTIER | dry run: fits, matched prices, populations, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import joint_vocab_shared_code_screen as parent
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    embedding = model.transformer.wte.weight.detach().float().contiguous()
    output = model.lm_head.weight.detach().float().contiguous()
    assert embedding.shape == output.shape == (W, D) and cfg["n_embd"] == D
    fit = parent._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    counts = torch.bincount(fit[:, 1:].reshape(-1), minlength=W).float()
    weights = {
        "sqrt_count_plus_1": (counts.to(DEV) + 1.0).sqrt(),
        "count_plus_1": counts.to(DEV) + 1.0,
    }
    fineweb = parent._load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", EVAL_ROWS)
    wikitext, fingerprint = parent._wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)
    fine_hidden, fine_target = parent._capture_hidden(model, fineweb)
    wiki_hidden, wiki_target = parent._capture_hidden(model, wikitext)
    counts_real = counts[:V_REAL].long().cpu()
    native_fn = lambda hidden: hidden @ output.T
    native_fine = parent._score_logits(fine_hidden, fine_target, counts_real, native_fn)
    native_wiki = parent._score_logits(wiki_hidden, wiki_target, counts_real, native_fn)
    native_price = 2 * W * D
    arms: dict[str, dict[str, object]] = {}

    for weight_name, weight in weights.items():
        weight = weight / weight.mean()
        fitted = _fit_full(embedding, output, weight)
        for rank in RANKS:
            independent_rank = rank + RANK_OFFSET
            residual_basis = fitted["residual_vectors"][:, :rank]
            residual_code = fitted["residual"] @ residual_basis
            independent_basis = fitted["output_vectors"][:, :independent_rank]
            independent_code = output @ independent_basis

            def shared_fn(hidden: torch.Tensor) -> torch.Tensor:
                return ((hidden @ fitted["mapping"].T) @ embedding.T
                        + (hidden @ residual_basis) @ residual_code.T)

            def independent_fn(hidden: torch.Tensor) -> torch.Tensor:
                return (hidden @ independent_basis) @ independent_code.T

            shared_price = W * D + D * D + rank * (W + D)
            independent_price = W * D + independent_rank * (W + D)
            row: dict[str, object] = {
                "weighting": weight_name,
                "shared_rank": rank,
                "independent_rank": independent_rank,
                "shared_scalars": shared_price,
                "independent_scalars": independent_price,
                "saving_fraction_native_vocab": 1.0 - shared_price / native_price,
                "ridge": fitted["ridge"],
            }
            for corpus, hidden, target, native in (
                ("fineweb", fine_hidden, fine_target, native_fine),
                ("wikitext", wiki_hidden, wiki_target, native_wiki),
            ):
                shared = parent._score_logits(hidden, target, counts_real, shared_fn)
                independent = parent._score_logits(hidden, target, counts_real, independent_fn)
                for score in (shared, independent):
                    score["damage"] = score["ce"] - native["ce"]
                    score["damage_by_frequency"] = parent._damage_by_frequency(score, native)
                row[f"shared_{corpus}"] = shared
                row[f"independent_{corpus}"] = independent
            key = f"{weight_name}_s{rank}"
            arms[key] = row
            print(f"{key} save={row['saving_fraction_native_vocab']:.1%}: shared/ind FW "
                  f"{row['shared_fineweb']['damage']:+.4f}/{row['independent_fineweb']['damage']:+.4f}, WT "
                  f"{row['shared_wikitext']['damage']:+.4f}/{row['independent_wikitext']['damage']:+.4f}; "
                  f"unseen {row['shared_fineweb']['damage_by_frequency']['unseen']['damage']:+.3f}/"
                  f"{row['shared_wikitext']['damage_by_frequency']['unseen']['damage']:+.3f}", flush=True)

    eligible = [row for row in arms.values() if _eligible(row["shared_scalars"], native_price)]

    def predictive(row: dict[str, object]) -> bool:
        return row["shared_fineweb"]["damage"] <= 0.10 and row["shared_wikitext"]["damage"] <= 0.12

    def beats_control(row: dict[str, object]) -> bool:
        return all(
            row[f"independent_{corpus}"]["damage"] >= 0
            and row[f"shared_{corpus}"]["damage"] <= 0.70 * row[f"independent_{corpus}"]["damage"]
            for corpus in ("fineweb", "wikitext")
        )

    def tail_safe(row: dict[str, object]) -> bool:
        return all(
            row[f"shared_{corpus}"]["damage_by_frequency"]["unseen"]["damage"] <= 0.50
            and row[f"shared_{corpus}"]["damage_by_frequency"]["count_ge_10"]["damage"] <= 0.03
            for corpus in ("fineweb", "wikitext")
        )

    qualifying = [row for row in eligible if predictive(row)]
    pred_a = bool(qualifying)
    pred_b = any(beats_control(row) for row in qualifying)
    pred_c = any(beats_control(row) and tail_safe(row) for row in qualifying)
    null = bool(
        all(row["shared_fineweb"]["damage"] >= 0.18 for row in eligible)
        or not any(
            all(row[f"shared_{corpus}"]["damage"] < row[f"independent_{corpus}"]["damage"]
                for corpus in ("fineweb", "wikitext"))
            for row in eligible
        )
    )
    result = {
        "status": "joint_vocab_distributed_rank_frontier_complete",
        "rung": 305,
        "claim_level": "fresh_two_corpus_distributed_rank_frontier_screen_only",
        "price": {"native_vocab_scalars": native_price, "shared_formula":
                  "50304*1152 + 1152^2 + s*(50304+1152)",
                  "matched_independent_rank_offset": RANK_OFFSET},
        "fit": {"rows": FIT_ROWS, "positions": int(fit[:, 1:].numel()),
                "evaluation_labels_used": False},
        "evaluation": {"fineweb_skip": 11000, "fineweb_rows": EVAL_ROWS,
                       "wikitext_skip": WIKI_SKIP, "wikitext_rows": EVAL_ROWS,
                       "wikitext_fingerprint": fingerprint},
        "native": {"fineweb": native_fine, "wikitext": native_wiki},
        "arms": arms,
        'pred_a_distributed_rank_is_predictive': bool(pred_a),
        'pred_b_shared_beats_matched_independent': bool(pred_b),
        'pred_c_tail_is_predictive': bool(pred_c),
        "null_no_useful_distributed_frontier": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("JOINT VOCAB DISTRIBUTED RANK FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
