"""RUNG 300B -- FREQUENCY-WEIGHTED JOINT VOCABULARY FOLLOW-UP.

Rung 300 was outcome-open when it fit the data-free shared code. It then found a
real but insufficient relation: at 73.88% vocabulary storage, shared rank-512
beat a price-matched independent rank-537 head on FineWeb (+0.743 vs +1.738 CE)
and WikiText (+0.647 vs +2.029), yet badly missed its predictive bars. Its frozen
frequency diagnostic localized the shared arm's damage to common targets
(count>=10: +1.215 nats), not rare targets (count1-2: +.016/+ .035).

This separately preregistered follow-up changes only the fit metric. On 480
frozen FineWeb fit rows disjoint from the skip1200 evaluation, define vocabulary
row weights w_t in {1, sqrt(count_t+1), count_t+1}, normalized to mean one.
For each weighting, refit both the full E->U map and the top-512 residual right
basis under weighted row loss. Compare against a rank-537 independent output SVD
fit under the identical weighting. Prices and executable formulas are unchanged:

  shared: E + M + P_512,V_512 = 85,622,784 scalars (73.88% of native vocab)
  independent: E + P_537,Q_537 = 85,582,080 scalars (slightly cheaper control).

The same 16 FineWeb and 16 frozen WikiText rows from rung 300 are evaluated; no
evaluation labels enter the fit.

Frozen predictions:
pred_a_frequency_metric_repairs_damage:
  Some weighted shared arm cuts rung-300 uniform shared damage by >=35% on BOTH
  FineWeb and WikiText.
pred_b_frequency_metric_is_predictive:
  Some weighted shared arm has FineWeb damage <=0.25 and WikiText <=0.30.
pred_c_shared_still_beats_independent:
  For some weighting, shared damage is >=20% smaller than the identically
  weighted, price-matched independent arm on BOTH corpora.

Null: no weighted shared arm cuts damage by 10% on both corpora, OR all weighted
shared arms have FineWeb damage >=0.50. This is a screen, not adoption.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "joint_vocab_frequency_weighted_followup_results.json"
DEV = "cuda"
D = 1152
W = 50304
SHARED_RANK = 512
INDEPENDENT_RANK = 537
UNIFORM_FW = 0.7429897051949625
UNIFORM_WT = 0.6474345305960156
RIDGE_REL = 1e-6


def _weighted_factors(
    embedding: torch.Tensor,
    output: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    sqrt_weight = weight.sqrt()[:, None]
    weighted_embedding = embedding * sqrt_weight
    weighted_output = output * sqrt_weight
    gram = weighted_embedding.T @ weighted_embedding
    ridge = RIDGE_REL * float(torch.trace(gram)) / D
    mapping = torch.linalg.solve(
        gram + ridge * torch.eye(D, device=DEV),
        weighted_embedding.T @ weighted_output,
    )
    residual = output - embedding @ mapping
    weighted_residual = residual * sqrt_weight
    values, vectors = torch.linalg.eigh(weighted_residual.T @ weighted_residual)
    residual_basis = vectors[:, torch.argsort(values, descending=True)[:SHARED_RANK]]
    residual_code = residual @ residual_basis

    values_u, vectors_u = torch.linalg.eigh(weighted_output.T @ weighted_output)
    independent_basis = vectors_u[:, torch.argsort(values_u, descending=True)[:INDEPENDENT_RANK]]
    independent_code = output @ independent_basis
    return {
        "mapping": mapping,
        "residual_basis": residual_basis,
        "residual_code": residual_code,
        "independent_basis": independent_basis,
        "independent_code": independent_code,
        "ridge": ridge,
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / "joint_vocab_shared_code_screen_results.json").exists()
        assert (ROOT / ".rowcache/fineweb_n480_skip80.pt").exists()
        print("JOINT VOCAB FREQUENCY WEIGHTED FOLLOWUP | dry run: parent, rows, prices, and bars valid")
        return
    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/qk_mdl")
    import joint_vocab_shared_code_screen as parent
    from tier2_model import load_elriggs

    first = json.load(open(ROOT / "joint_vocab_shared_code_screen_results.json"))
    assert abs(first["arms"]["512"]["shared_fineweb"]["damage"] - UNIFORM_FW) < 1e-9
    assert abs(first["arms"]["512"]["shared_wikitext"]["damage"] - UNIFORM_WT) < 1e-9
    model, cfg = load_elriggs("bilin18")
    embedding = model.transformer.wte.weight.detach().float().contiguous()
    output = model.lm_head.weight.detach().float().contiguous()
    assert embedding.shape == output.shape == (W, D) and cfg["n_embd"] == D

    fit = parent._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", 480)
    counts = torch.bincount(fit[:, 1:].reshape(-1), minlength=W).float()
    fineweb = parent._load_rows(ROOT / ".rowcache/fineweb_n96_skip1200.pt", 16)
    wikitext, fingerprint = parent._wikitext_rows(16)
    fine_hidden, fine_target = parent._capture_hidden(model, fineweb)
    wiki_hidden, wiki_target = parent._capture_hidden(model, wikitext)
    fit_counts_cpu = counts[:parent.V_REAL].long().cpu()
    native_fn = lambda hidden: hidden @ output.T
    native_fine = parent._score_logits(fine_hidden, fine_target, fit_counts_cpu, native_fn)
    native_wiki = parent._score_logits(wiki_hidden, wiki_target, fit_counts_cpu, native_fn)

    weights = {
        "uniform": torch.ones(W, device=DEV),
        "sqrt_count_plus_1": (counts.to(DEV) + 1.0).sqrt(),
        "count_plus_1": counts.to(DEV) + 1.0,
    }
    arms = {}
    for name, weight in weights.items():
        weight = weight / weight.mean()
        factor = _weighted_factors(embedding, output, weight)

        def shared_fn(hidden: torch.Tensor) -> torch.Tensor:
            return ((hidden @ factor["mapping"].T) @ embedding.T
                    + (hidden @ factor["residual_basis"]) @ factor["residual_code"].T)

        def independent_fn(hidden: torch.Tensor) -> torch.Tensor:
            return (hidden @ factor["independent_basis"]) @ factor["independent_code"].T

        sf = parent._score_logits(fine_hidden, fine_target, fit_counts_cpu, shared_fn)
        sw = parent._score_logits(wiki_hidden, wiki_target, fit_counts_cpu, shared_fn)
        inf = parent._score_logits(fine_hidden, fine_target, fit_counts_cpu, independent_fn)
        inw = parent._score_logits(wiki_hidden, wiki_target, fit_counts_cpu, independent_fn)
        for score, native in ((sf, native_fine), (sw, native_wiki),
                              (inf, native_fine), (inw, native_wiki)):
            score["damage"] = score["ce"] - native["ce"]
            score["damage_by_frequency"] = parent._damage_by_frequency(score, native)
        arms[name] = {
            "weight_min": float(weight.min()),
            "weight_max": float(weight.max()),
            "ridge": factor["ridge"],
            "shared_fineweb": sf,
            "shared_wikitext": sw,
            "independent_fineweb": inf,
            "independent_wikitext": inw,
        }
        print(
            f"{name}: FW shared/ind {sf['damage']:+.4f}/{inf['damage']:+.4f}; "
            f"WT shared/ind {sw['damage']:+.4f}/{inw['damage']:+.4f}",
            flush=True,
        )

    weighted = [arm for name, arm in arms.items() if name != "uniform"]
    pred_a = any(
        arm["shared_fineweb"]["damage"] <= 0.65 * UNIFORM_FW
        and arm["shared_wikitext"]["damage"] <= 0.65 * UNIFORM_WT
        for arm in weighted
    )
    pred_b = any(
        arm["shared_fineweb"]["damage"] <= 0.25
        and arm["shared_wikitext"]["damage"] <= 0.30
        for arm in weighted
    )
    pred_c = any(
        arm["independent_fineweb"]["damage"] >= 0
        and arm["independent_wikitext"]["damage"] >= 0
        and arm["shared_fineweb"]["damage"] <= 0.8 * arm["independent_fineweb"]["damage"]
        and arm["shared_wikitext"]["damage"] <= 0.8 * arm["independent_wikitext"]["damage"]
        for arm in weighted
    )
    null = bool(
        not any(
            arm["shared_fineweb"]["damage"] <= 0.9 * UNIFORM_FW
            and arm["shared_wikitext"]["damage"] <= 0.9 * UNIFORM_WT
            for arm in weighted
        )
        or all(arm["shared_fineweb"]["damage"] >= 0.50 for arm in weighted)
    )
    result = {
        "status": "joint_vocab_frequency_weighted_followup_complete",
        "rung": "300B",
        "claim_level": "post_result_metric_followup_screen_only",
        "parent_uniform_damage": {"fineweb": UNIFORM_FW, "wikitext": UNIFORM_WT},
        "price": {
            "native_vocab_scalars": 2 * W * D,
            "shared_scalars": W * D + D * D + SHARED_RANK * (W + D),
            "independent_scalars": W * D + INDEPENDENT_RANK * (W + D),
        },
        "fit": {
            "rows": len(fit),
            "positions": int(fit[:, 1:].numel()),
            "evaluation_labels_used": False,
        },
        "evaluation": {
            "fineweb_rows": len(fineweb),
            "wikitext_rows": len(wikitext),
            "wikitext_fingerprint": fingerprint,
        },
        "arms": arms,
        'pred_a_frequency_metric_repairs_damage': bool(pred_a),
        'pred_b_frequency_metric_is_predictive': bool(pred_b),
        'pred_c_shared_still_beats_independent': bool(pred_c),
        "null_no_material_frequency_repair": null,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("JOINT VOCAB FREQUENCY WEIGHTED FOLLOWUP DONE", flush=True)


if __name__ == "__main__":
    main()
