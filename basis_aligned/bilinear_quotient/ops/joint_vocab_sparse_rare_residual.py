"""RUNG 304 -- FISHER-SELECTED EXACT RARE ROWS FOR THE SHARED VOCAB CODE.

Keep rung 300B's count-weighted executable vocabulary program

    U_base = E M + P_512 V_512^T,

then spend the remaining strict-25%-saving budget on K=1129 indexed exact
output-row corrections.  Each correction stores one 1152-vector and one token
index, so total vocabulary storage is

    85,622,784 + 1129*(1152+1) = 86,924,521 scalars
                                      = 74.9993% of native vocabulary.

All selected tokens have fit target count <=2.  The proposed selector uses 32
fit-only FineWeb contexts and ranks rows by a diagonal empirical-Fisher estimate

    score_t = sum_i p_i(t)(1-p_i(t)) (l_native(i,t)-l_base(i,t))^2.

This estimates the second-order CE contribution of the base program's error in
row t.  Equal-K/equal-price controls select by exact correction-row norm or by a
seeded random rare ordering.  Selection never observes evaluation labels.

Prospective evaluation uses 32 FineWeb skip7000 rows and 32 WikiText-2 test rows
after token skip10000, neither previously used to tune this hybrid.

Frozen predictions
------------------
pred_a_fisher_repairs_aggregate:
    Fisher rows reduce base shared-code CE damage by >=20% on BOTH corpora.
pred_b_fisher_repairs_unseen_without_common_regression:
    Fisher rows reduce unseen-target damage by >=40% on BOTH corpora while
    increasing count>=10 damage by <=.02 nats on each.
pred_c_fisher_beats_equal_price_controls:
    Fisher damage is >=10% smaller than BOTH norm-selected and random-selected
    damage on BOTH corpora (all control damages nonnegative).

Null: Fisher repairs aggregate damage by <5% on either corpus, or repairs unseen
damage by <10% on both.  This is a new-population exploit screen only; a pass
still needs census, 62 certificates, exact standalone composition billing, OOD,
and signed interventions before adoption.
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
OUT = ROOT / "joint_vocab_sparse_rare_residual_results.json"
DEV = "cuda"
D = 1152
W = 50304
V_REAL = 50257
RANK = 512
K = 1129
FIT_ROWS = 480
SELECT_ROWS = 32
EVAL_ROWS = 32
WIKI_SKIP = 10000
RANDOM_SEED = 30420260901


@torch.no_grad()
def _fisher_scores(hidden_cpu: torch.Tensor, output: torch.Tensor, shared_fn) -> torch.Tensor:
    total = torch.zeros(W, dtype=torch.float64)
    for start in range(0, len(hidden_cpu), 256):
        hidden = hidden_cpu[start:start + 256].float().to(DEV)
        native = 30.0 * torch.tanh((hidden @ output.T) / 30.0)
        shared = 30.0 * torch.tanh(shared_fn(hidden) / 30.0)
        probability = torch.softmax(native, dim=-1)
        contribution = probability * (1.0 - probability) * (native - shared).square()
        total += contribution.sum(0).double().cpu()
    return total


def _top_allowed(score: torch.Tensor, allowed: torch.Tensor, k: int) -> torch.Tensor:
    ids = torch.nonzero(allowed, as_tuple=False).flatten()
    assert len(ids) >= k
    order = torch.argsort(score[ids], descending=True, stable=True)
    return ids[order[:k]].long()


def _selection_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return len(set(left.tolist()) & set(right.tolist())) / len(left)


def _repair_fraction(base: float, candidate: float) -> float:
    return (base - candidate) / max(abs(base), 1e-12)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n480_skip80.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        base_price = W * D + D * D + RANK * (W + D)
        hybrid_price = base_price + K * (D + 1)
        assert hybrid_price <= math.floor(0.75 * (2 * W * D))
        assert hybrid_price + (D + 1) > math.floor(0.75 * (2 * W * D))
        print("JOINT VOCAB SPARSE RARE RESIDUAL | dry run: exact maximal K, populations, and bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import joint_vocab_shared_code_screen as parent
    from joint_vocab_frequency_weighted_followup import _weighted_factors
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    embedding = model.transformer.wte.weight.detach().float().contiguous()
    output = model.lm_head.weight.detach().float().contiguous()
    assert embedding.shape == output.shape == (W, D) and cfg["n_embd"] == D
    fit = parent._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    counts = torch.bincount(fit[:, 1:].reshape(-1), minlength=W).float()
    weight = (counts.to(DEV) + 1.0)
    weight = weight / weight.mean()
    factor = _weighted_factors(embedding, output, weight)

    def shared_fn(hidden: torch.Tensor) -> torch.Tensor:
        return ((hidden @ factor["mapping"].T) @ embedding.T
                + (hidden @ factor["residual_basis"]) @ factor["residual_code"].T)

    approximation = embedding @ factor["mapping"] + factor["residual_code"] @ factor["residual_basis"].T
    correction = output - approximation
    rare_allowed = torch.zeros(W, dtype=torch.bool)
    rare_allowed[:V_REAL] = counts[:V_REAL].cpu() <= 2
    assert int(rare_allowed.sum()) >= K

    select_hidden, _select_target = parent._capture_hidden(model, fit[:SELECT_ROWS])
    fisher_score = _fisher_scores(select_hidden, output, shared_fn)
    norm_score = correction.square().sum(1).detach().double().cpu()
    fisher_ids = _top_allowed(fisher_score, rare_allowed, K)
    norm_ids = _top_allowed(norm_score, rare_allowed, K)
    rare_ids = torch.nonzero(rare_allowed, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    random_ids = rare_ids[torch.randperm(len(rare_ids), generator=generator)[:K]]
    selections = {"fisher_rare": fisher_ids, "norm_rare": norm_ids, "random_rare": random_ids}

    fineweb = parent._load_rows(ROOT / ".rowcache/fineweb_n192_skip7000.pt", EVAL_ROWS)
    wikitext, fingerprint = parent._wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)
    fine_hidden, fine_target = parent._capture_hidden(model, fineweb)
    wiki_hidden, wiki_target = parent._capture_hidden(model, wikitext)
    counts_real = counts[:V_REAL].long().cpu()
    native_fn = lambda hidden: hidden @ output.T
    native_fine = parent._score_logits(fine_hidden, fine_target, counts_real, native_fn)
    native_wiki = parent._score_logits(wiki_hidden, wiki_target, counts_real, native_fn)
    base_fine = parent._score_logits(fine_hidden, fine_target, counts_real, shared_fn)
    base_wiki = parent._score_logits(wiki_hidden, wiki_target, counts_real, shared_fn)
    for score, native in ((base_fine, native_fine), (base_wiki, native_wiki)):
        score["damage"] = score["ce"] - native["ce"]
        score["damage_by_frequency"] = parent._damage_by_frequency(score, native)

    arms: dict[str, dict[str, object]] = {}
    for name, ids_cpu in selections.items():
        ids = ids_cpu.to(DEV)
        rows = correction[ids].contiguous()

        def hybrid_fn(hidden: torch.Tensor, ids=ids, rows=rows) -> torch.Tensor:
            logits = shared_fn(hidden)
            logits[:, ids] = logits[:, ids] + hidden @ rows.T
            return logits

        fine = parent._score_logits(fine_hidden, fine_target, counts_real, hybrid_fn)
        wiki = parent._score_logits(wiki_hidden, wiki_target, counts_real, hybrid_fn)
        for score, native, base in ((fine, native_fine, base_fine), (wiki, native_wiki, base_wiki)):
            score["damage"] = score["ce"] - native["ce"]
            score["damage_by_frequency"] = parent._damage_by_frequency(score, native)
            score["aggregate_repair_fraction"] = _repair_fraction(base["damage"], score["damage"])
            for frequency in score["damage_by_frequency"]:
                base_damage = base["damage_by_frequency"][frequency]["damage"]
                candidate_damage = score["damage_by_frequency"][frequency]["damage"]
                score["damage_by_frequency"][frequency]["repair_fraction"] = _repair_fraction(
                    base_damage, candidate_damage)
        arms[name] = {
            "selected_count": len(ids_cpu),
            "selected_fit_count_histogram": {
                str(count): int((counts[ids_cpu].long().cpu() == count).sum()) for count in (0, 1, 2)
            },
            "fineweb": fine,
            "wikitext": wiki,
        }
        print(f"{name}: FW {fine['damage']:+.4f} ({fine['aggregate_repair_fraction']:.1%} repair), "
              f"WT {wiki['damage']:+.4f} ({wiki['aggregate_repair_fraction']:.1%} repair); unseen "
              f"{fine['damage_by_frequency']['unseen']['damage']:+.3f}/"
              f"{wiki['damage_by_frequency']['unseen']['damage']:+.3f}", flush=True)

    fisher = arms["fisher_rare"]
    pred_a = all(fisher[corpus]["aggregate_repair_fraction"] >= 0.20
                 for corpus in ("fineweb", "wikitext"))
    pred_b = all(
        fisher[corpus]["damage_by_frequency"]["unseen"]["repair_fraction"] >= 0.40
        and fisher[corpus]["damage_by_frequency"]["count_ge_10"]["damage"]
        <= base["damage_by_frequency"]["count_ge_10"]["damage"] + 0.02
        for corpus, base in (("fineweb", base_fine), ("wikitext", base_wiki))
    )
    pred_c = all(
        arms[control][corpus]["damage"] >= 0
        and fisher[corpus]["damage"] <= 0.90 * arms[control][corpus]["damage"]
        for control in ("norm_rare", "random_rare") for corpus in ("fineweb", "wikitext")
    )
    null = bool(
        any(fisher[corpus]["aggregate_repair_fraction"] < 0.05 for corpus in ("fineweb", "wikitext"))
        or all(fisher[corpus]["damage_by_frequency"]["unseen"]["repair_fraction"] < 0.10
               for corpus in ("fineweb", "wikitext"))
    )
    native_price = 2 * W * D
    base_price = W * D + D * D + RANK * (W + D)
    hybrid_price = base_price + K * (D + 1)
    result = {
        "status": "joint_vocab_sparse_rare_residual_complete",
        "rung": 304,
        "claim_level": "fresh_two_corpus_sparse_residual_exploit_screen_only",
        "price": {
            "native_vocab_scalars": native_price,
            "base_shared_scalars": base_price,
            "indexed_row_scalars_each": D + 1,
            "selected_rows": K,
            "hybrid_scalars": hybrid_price,
            "hybrid_fraction_native_vocab": hybrid_price / native_price,
            "saving_fraction_native_vocab": 1.0 - hybrid_price / native_price,
        },
        "fit": {"count_rows": FIT_ROWS, "fisher_context_rows": SELECT_ROWS,
                "selection_labels_used": False, "rare_count_max": 2},
        "evaluation": {"fineweb_skip": 7000, "fineweb_rows": EVAL_ROWS,
                       "wikitext_skip": WIKI_SKIP, "wikitext_rows": EVAL_ROWS,
                       "wikitext_fingerprint": fingerprint},
        "native": {"fineweb": native_fine, "wikitext": native_wiki},
        "base_shared": {"fineweb": base_fine, "wikitext": base_wiki},
        "selection_overlap": {
            "fisher_norm": _selection_overlap(fisher_ids, norm_ids),
            "fisher_random": _selection_overlap(fisher_ids, random_ids),
            "norm_random": _selection_overlap(norm_ids, random_ids),
        },
        "arms": arms,
        'pred_a_fisher_repairs_aggregate': bool(pred_a),
        'pred_b_fisher_repairs_unseen_without_common_regression': bool(pred_b),
        'pred_c_fisher_beats_equal_price_controls': bool(pred_c),
        "null_no_rare_row_repair": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"base_damage": {"fineweb": base_fine["damage"], "wikitext": base_wiki["damage"]},
                      "predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("JOINT VOCAB SPARSE RARE RESIDUAL DONE", flush=True)


if __name__ == "__main__":
    main()
