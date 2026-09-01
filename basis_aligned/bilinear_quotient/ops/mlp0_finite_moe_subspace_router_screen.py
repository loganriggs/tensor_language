"""RUNG 354 -- LITERALLY PRICED FOUR-STATE MLP0 SUBSPACE ROUTER.

Cluster all exact folded token inputs into four fixed states and fit one
context-RRR rank128 Left/Right input subspace per state.  The executable stores
the 50,304-entry token-state table and chooses exactly one expert.  Compare at
equal price with global context-RRR p517 and a balanced random four-state
router.  This is finite MoE structure, not combinatorial per-token top-k.

Frozen predictions
------------------
pred_a_cluster_router_beats_global_mean_at_equal_price:
    Clustered mean damage <=80% of global p517 on BOTH corpora and <=.030.
pred_b_cluster_router_beats_global_tails:
    Clustered p95/max <=85% of global on both and maxima <=.080.
pred_c_states_are_real_balanced_and_split_stable:
    Clustered mean beats random by >=15% on both; each state has >=10% of the
    vocabulary and >=500 fit tokens/half; fit-A/B mean differs <=.010/corpus.

Null: clustered fails to beat global on either corpus, OR is no better than
random on both.  No state-count, rank, or cluster tuning follows this screen.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_finite_moe_subspace_router_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
VOCAB = 50304
REAL_V = 50257
STATES = 4
EXPERT_RANK = 128
GLOBAL_RANK = 517
PCA_RANK = 32
FIT_A = (0, 48)
FIT_B = (48, 96)
FINEWEB_EVAL = (40, 80)
WIKI_SKIP = 440 * 257
EVAL_ROWS = 40
ROUTER_PRICE = STATES * EXPERT_RANK * (D + 2 * H) + H * D + D + VOCAB
GLOBAL_PRICE = GLOBAL_RANK * (D + 2 * H) + H * D + D


@torch.no_grad()
def _capture_inputs(model, rows, manual_logits):
    pieces = []
    token_pieces = []

    def hook(_module, args, _output):
        pieces.append(args[0].detach().reshape(-1, D).float())

    handle = model.transformer.h[0].mlp.register_forward_hook(hook)
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            manual_logits(model, batch[:, :-1].to(DEV))
            token_pieces.append(batch[:, :-1].reshape(-1).to(DEV))
    finally:
        handle.remove()
    x = torch.cat(pieces)
    token = torch.cat(token_pieces)
    assert x.shape == (len(rows) * 256, D) and token.shape == (len(rows) * 256,)
    return x, token


@torch.no_grad()
def _state_programs(model, x, token, route, rrr_program):
    programs = {}
    counts = []
    labels = route[token]
    for state in range(STATES):
        member = x[labels == state]
        counts.append(int(len(member)))
        if len(member) < max(500, EXPERT_RANK):
            raise RuntimeError(f"state {state} has only {len(member)} contextual fit tokens")
        covariance = member.T @ member / len(member)
        covariance = .5 * (covariance + covariance.T)
        program, basis, diagnostic = rrr_program(
            model.transformer.h[0].mlp, covariance, rank=EXPERT_RANK)
        programs[state] = program
        programs[state]["_basis"] = basis
        programs[state]["_diagnostic"] = diagnostic
    return programs, counts


@torch.no_grad()
def _global_program(model, x, rrr_program):
    covariance = x.T @ x / len(x)
    covariance = .5 * (covariance + covariance.T)
    return rrr_program(model.transformer.h[0].mlp, covariance, rank=GLOBAL_RANK)[0]


@torch.no_grad()
def _score_router_rows(model, rows, route, programs, manual_logits):
    values = []
    current_token = None
    down = programs[0]["down"]
    bias = programs[0]["bias"]

    def hook(_module, args, output):
        x = args[0].float().reshape(-1, D)
        labels = route[current_token.reshape(-1)]
        result = torch.empty((len(x), D), device=x.device, dtype=torch.float32)
        for state in range(STATES):
            selected = labels == state
            if not bool(selected.any()):
                continue
            program = programs[state]
            z = x[selected] @ program["encoder"].T
            hidden = (z @ program["left"].T) * (z @ program["right"].T)
            result[selected] = hidden @ down.T + bias
        return result.reshape_as(output).to(output.dtype)

    handle = model.transformer.h[0].mlp.register_forward_hook(hook)
    try:
        for row in rows:
            current_token = row[:-1].unsqueeze(0).to(DEV)
            target = row[1:].to(DEV)
            logits = manual_logits(model, current_token)[0].float()
            values.append(float(torch.nn.functional.cross_entropy(logits, target)))
    finally:
        handle.remove()
    return torch.tensor(values, dtype=torch.float64)


def _ratio(value: float, control: float) -> float:
    return max(0.0, value) / max(1e-6, max(0.0, control))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] <= 192
        assert ROUTER_PRICE == 10_668_288 and GLOBAL_PRICE == 10_669_824
        assert ROUTER_PRICE < GLOBAL_PRICE and WIKI_SKIP == 113_080
        assert WIKI_SKIP + EVAL_ROWS * 257 == 123_360
        print("FINITE MLP0 MOE SUBSPACE ROUTER | dry run: states, prices, splits, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_embedding_fold_structure_screen import _kmeans
    from mlp0_exact_token_shared_input_encoder import _folded_inputs
    from mlp0_tail_robust_context_metric_screen import _score_rows, _summary
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    torch.manual_seed(35400)
    model, cfg = load_elriggs("bilin18")
    folded = _folded_inputs(model, cfg).to(DEV)
    centered = folded - folded.mean(0, keepdim=True)
    _u, _s, vectors = torch.pca_lowrank(centered, q=PCA_RANK, center=False, niter=4)
    features = torch.nn.functional.normalize(centered @ vectors, dim=1)
    cluster_label, _same, centers = _kmeans(features, features, STATES, 35401)
    assert torch.equal(cluster_label, _same)
    cluster_route = torch.zeros(VOCAB, dtype=torch.long, device=DEV)
    cluster_route[:REAL_V] = cluster_label
    generator = torch.Generator(device=DEV).manual_seed(35402)
    permutation = torch.randperm(REAL_V, generator=generator, device=DEV)
    random_route = torch.zeros(VOCAB, dtype=torch.long, device=DEV)
    random_route[permutation] = torch.arange(REAL_V, device=DEV) % STATES
    state_fraction = [float((cluster_label == state).float().mean()) for state in range(STATES)]
    del folded, centered, features, _u, _s
    torch.cuda.empty_cache()

    fit_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    eval_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cached = eval_cached["rows"] if isinstance(eval_cached, dict) else eval_cached
    fit_rows = {
        "a": fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous(),
        "b": fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous(),
    }
    fineweb = eval_cached[FINEWEB_EVAL[0]:FINEWEB_EVAL[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)

    x_a, token_a = _capture_inputs(model, fit_rows["a"], _manual_logits)
    x_b, token_b = _capture_inputs(model, fit_rows["b"], _manual_logits)
    cluster_a, cluster_count_a = _state_programs(
        model, x_a, token_a, cluster_route, _rrr_program)
    cluster_b, cluster_count_b = _state_programs(
        model, x_b, token_b, cluster_route, _rrr_program)
    random_a, random_count_a = _state_programs(
        model, x_a, token_a, random_route, _rrr_program)
    global_a = _global_program(model, x_a, _rrr_program)
    del x_a, x_b, token_a, token_b
    torch.cuda.empty_cache()

    native = {
        "fineweb": _score_rows(model, fineweb, _manual_logits),
        "wikitext": _score_rows(model, wikitext, _manual_logits),
    }
    summaries = {}
    for name, route, programs in (
        ("cluster_fit_a", cluster_route, cluster_a),
        ("cluster_fit_b", cluster_route, cluster_b),
        ("random_fit_a", random_route, random_a),
    ):
        summaries[name] = {}
        for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
            ce = _score_router_rows(model, rows, route, programs, _manual_logits)
            summaries[name][corpus] = _summary(ce - native[corpus])
            print(f"{name} {corpus}: {summaries[name][corpus]}", flush=True)
    summaries["global_p517"] = {}
    for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
        ce = _score_rows(model, rows, _manual_logits, global_a)
        summaries["global_p517"][corpus] = _summary(ce - native[corpus])
        print(f"global_p517 {corpus}: {summaries['global_p517'][corpus]}", flush=True)

    cluster = summaries["cluster_fit_a"]
    global_control = summaries["global_p517"]
    random_control = summaries["random_fit_a"]
    pred_a = all(
        _ratio(cluster[corpus]["mean"], global_control[corpus]["mean"]) <= .80
        and cluster[corpus]["mean"] <= .030
        for corpus in ("fineweb", "wikitext")
    )
    pred_b = all(
        _ratio(cluster[corpus]["p95"], global_control[corpus]["p95"]) <= .85
        and _ratio(cluster[corpus]["max"], global_control[corpus]["max"]) <= .85
        and cluster[corpus]["max"] <= .080
        for corpus in ("fineweb", "wikitext")
    )
    pred_c = (
        all(_ratio(cluster[corpus]["mean"], random_control[corpus]["mean"]) <= .85
            for corpus in ("fineweb", "wikitext"))
        and min(state_fraction) >= .10
        and min(cluster_count_a + cluster_count_b) >= 500
        and all(abs(summaries["cluster_fit_a"][corpus]["mean"]
                    - summaries["cluster_fit_b"][corpus]["mean"]) <= .010
                for corpus in ("fineweb", "wikitext"))
    )
    fails_global_either = any(
        cluster[corpus]["mean"] >= global_control[corpus]["mean"]
        for corpus in ("fineweb", "wikitext")
    )
    no_better_random_both = all(
        cluster[corpus]["mean"] >= random_control[corpus]["mean"]
        for corpus in ("fineweb", "wikitext")
    )
    null = fails_global_either or no_better_random_both
    result = {
        "status": "mlp0_finite_moe_subspace_router_screen_complete",
        "rung": 354,
        "claim_level": "exact_token_router_context_fit_two_corpus_equal_price_screen_only",
        "program": {
            "states": STATES, "expert_rank": EXPERT_RANK,
            "router": "stored_exact_token_to_state_table",
            "router_table_entries": VOCAB,
            "literal_mlp0_scalars": ROUTER_PRICE,
            "matched_global_rank": GLOBAL_RANK,
            "matched_global_scalars": GLOBAL_PRICE,
        },
        "clustering": {
            "population": "all_50257_exact_folded_token_inputs",
            "feature": "centered_embedding_fold_pca32_then_unit_normalize",
            "state_fractions": state_fraction,
            "center_norms": [float(value) for value in centers.norm(dim=1)],
        },
        "fit": {
            "cache": "fineweb_n192_skip11000.pt",
            "fit_a": list(FIT_A), "fit_b": list(FIT_B),
            "cluster_counts_a": cluster_count_a,
            "cluster_counts_b": cluster_count_b,
            "random_counts_a": random_count_a,
        },
        "evaluation": {
            "fineweb_cache": "fineweb_n192_skip7000.pt",
            "fineweb_rows_half_open": list(FINEWEB_EVAL),
            "wikitext103_train_token_span_half_open": [WIKI_SKIP,
                                                         WIKI_SKIP + EVAL_ROWS * 257],
            "dataset_fingerprint": fingerprint,
            "source_token_count": token_count,
        },
        "row_damage_summaries": summaries,
        'pred_a_cluster_router_beats_global_mean_at_equal_price': bool(pred_a),
        'pred_b_cluster_router_beats_global_tails': bool(pred_b),
        'pred_c_states_are_real_balanced_and_split_stable': bool(pred_c),
        "null_finite_router_has_no_equal_price_advantage": bool(null),
        "stop_rule": "no_state_count_rank_or_cluster_tuning_after_result",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "state_fraction": state_fraction, "runtime_s": result["runtime_s"]},
                     indent=2), flush=True)
    print("FINITE MLP0 MOE SUBSPACE ROUTER SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
