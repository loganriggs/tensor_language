"""RUNG 371 -- BEHAVIOR-NAMED FOUR-STATE MLP0 ROUTER.

Use an exact token-morphology state fixed before fitting: word-start alpha,
continuation alpha, digit-containing, or punctuation/other.  Each state selects
one fixed context-RRR rank128 expert.  Compare equal-price shared and random
routers.  This is a finite executable MoE, not generic top-k.

Frozen predictions
------------------
pred_a_morphology_router_beats_global_mean:
    Mean <=80% of global p517 on both corpora and <=.030.
pred_b_morphology_router_beats_global_tails:
    p95/max <=90% of global on both and max <=.100.
pred_c_named_states_beat_random_and_are_stable:
    Mean <=85% random on both; each state >=2% vocabulary and >=300 fit
    tokens/half; independent-fit mean gap <=.010.

Null: fails to beat global on either corpus or no better than random on both.
No state, rank, or morphology-definition tuning follows.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import tiktoken
import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_morphology_state_finite_router_results.json"
DEV = "cuda"
D = 1152
VOCAB = 50304
REAL_V = 50257
STATES = 4
EXPERT_RANK = 128
GLOBAL_RANK = 517
FIT_A = (0, 48)
FIT_B = (48, 96)
FINEWEB_EVAL = (0, 40)
WIKI_SKIP = 275_504
EVAL_ROWS = 40
ROUTER_PRICE = 10_668_288
GLOBAL_PRICE = 10_669_824


def _morphology_route():
    encoder = tiktoken.get_encoding("gpt2")
    route = torch.full((VOCAB,), 3, dtype=torch.long)
    labels = []
    for token in range(REAL_V):
        text = encoder.decode_single_token_bytes(token).decode("utf-8", "ignore")
        has_alpha = any(character.isalpha() for character in text)
        has_digit = any(character.isdigit() for character in text)
        if has_digit:
            state = 2
        elif has_alpha and text.startswith(" "):
            state = 0
        elif has_alpha:
            state = 1
        else:
            state = 3
        labels.append(state)
    route[:REAL_V] = torch.tensor(labels)
    return route.to(DEV)


def _ratio(value, control):
    return max(0.0, value) / max(1e-6, max(0.0, control))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert ROUTER_PRICE < GLOBAL_PRICE and WIKI_SKIP + EVAL_ROWS * 257 == 285_784
        print("MLP0 MORPHOLOGY STATE FINITE ROUTER | dry run: states, prices, splits, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp0_finite_moe_subspace_router_screen as H
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_tail_robust_context_metric_screen import _score_rows, _summary
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    H.MIN_STATE_SAMPLES = 300
    route = _morphology_route()
    state_fraction = [float((route[:REAL_V] == state).float().mean())
                      for state in range(STATES)]
    generator = torch.Generator(device=DEV).manual_seed(37100)
    permutation = torch.randperm(REAL_V, generator=generator, device=DEV)
    random_route = torch.zeros(VOCAB, dtype=torch.long, device=DEV)
    random_route[permutation] = torch.arange(REAL_V, device=DEV) % STATES

    fit_cache = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cache = fit_cache["rows"] if isinstance(fit_cache, dict) else fit_cache
    eval_cache = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cache = eval_cache["rows"] if isinstance(eval_cache, dict) else eval_cache
    fit_a = fit_cache[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    fit_b = fit_cache[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    fineweb = eval_cache[FINEWEB_EVAL[0]:FINEWEB_EVAL[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)

    x_a, token_a = H._capture_inputs(model, fit_a, _manual_logits)
    x_b, token_b = H._capture_inputs(model, fit_b, _manual_logits)
    morphology_a, counts_a = H._state_programs(model, x_a, token_a, route, _rrr_program)
    morphology_b, counts_b = H._state_programs(model, x_b, token_b, route, _rrr_program)
    random_a, random_counts = H._state_programs(
        model, x_a, token_a, random_route, _rrr_program)
    global_a = H._global_program(model, x_a, _rrr_program)
    del x_a, x_b, token_a, token_b
    torch.cuda.empty_cache()

    native = {"fineweb": _score_rows(model, fineweb, _manual_logits),
              "wikitext": _score_rows(model, wikitext, _manual_logits)}
    summaries = {}
    for name, used_route, programs in (
        ("morphology_fit_a", route, morphology_a),
        ("morphology_fit_b", route, morphology_b),
        ("random_fit_a", random_route, random_a),
    ):
        summaries[name] = {}
        for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
            ce = H._score_router_rows(model, rows, used_route, programs, _manual_logits)
            summaries[name][corpus] = _summary(ce - native[corpus])
            print(f"{name} {corpus}: {summaries[name][corpus]}", flush=True)
    summaries["global_p517"] = {}
    for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
        ce = _score_rows(model, rows, _manual_logits, global_a)
        summaries["global_p517"][corpus] = _summary(ce - native[corpus])
        print(f"global_p517 {corpus}: {summaries['global_p517'][corpus]}", flush=True)

    morphology = summaries["morphology_fit_a"]
    random = summaries["random_fit_a"]
    global_control = summaries["global_p517"]
    pred_a = all(_ratio(morphology[c]["mean"], global_control[c]["mean"]) <= .80
                 and morphology[c]["mean"] <= .030 for c in ("fineweb", "wikitext"))
    pred_b = all(_ratio(morphology[c]["p95"], global_control[c]["p95"]) <= .90
                 and _ratio(morphology[c]["max"], global_control[c]["max"]) <= .90
                 and morphology[c]["max"] <= .100 for c in ("fineweb", "wikitext"))
    pred_c = (all(_ratio(morphology[c]["mean"], random[c]["mean"]) <= .85
                  for c in ("fineweb", "wikitext"))
              and min(state_fraction) >= .02 and min(counts_a + counts_b) >= 300
              and all(abs(summaries["morphology_fit_a"][c]["mean"]
                          - summaries["morphology_fit_b"][c]["mean"]) <= .010
                      for c in ("fineweb", "wikitext")))
    null = (any(morphology[c]["mean"] >= global_control[c]["mean"]
                for c in ("fineweb", "wikitext"))
            or all(morphology[c]["mean"] >= random[c]["mean"]
                   for c in ("fineweb", "wikitext")))
    result = {
        "status": "mlp0_morphology_state_finite_router_complete",
        "rung": 371,
        "claim_level": "behavior_named_finite_router_equal_price_two_corpus_screen_only",
        "program": {"states": STATES, "expert_rank": EXPERT_RANK,
                    "route": ["word_start_alpha", "continuation_alpha",
                              "digit_containing", "punctuation_or_other"],
                    "router_table_entries": VOCAB,
                    "literal_mlp0_scalars": ROUTER_PRICE,
                    "matched_global_rank": GLOBAL_RANK,
                    "matched_global_scalars": GLOBAL_PRICE},
        "state_fractions": state_fraction,
        "fit": {"cache": "fineweb_n192_skip11000.pt", "fit_a": list(FIT_A),
                "fit_b": list(FIT_B), "morphology_counts_a": counts_a,
                "morphology_counts_b": counts_b, "random_counts_a": random_counts},
        "evaluation": {"fineweb_cache": "fineweb_n192_skip7000.pt",
                       "fineweb_rows_half_open": list(FINEWEB_EVAL),
                       "wikitext103_train_span_half_open": [WIKI_SKIP,
                                                             WIKI_SKIP + EVAL_ROWS * 257],
                       "dataset_fingerprint": fingerprint,
                       "source_token_count": token_count},
        "row_damage_summaries": summaries,
        'pred_a_morphology_router_beats_global_mean': bool(pred_a),
        'pred_b_morphology_router_beats_global_tails': bool(pred_b),
        'pred_c_named_states_beat_random_and_are_stable': bool(pred_c),
        "null_behavior_named_router_has_no_equal_price_advantage": bool(null),
        "stop_rule": "no_state_rank_or_morphology_definition_tuning",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"state_fraction": state_fraction, "counts_a": counts_a,
                      "counts_b": counts_b, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP0 MORPHOLOGY STATE FINITE ROUTER DONE", flush=True)


if __name__ == "__main__":
    main()
