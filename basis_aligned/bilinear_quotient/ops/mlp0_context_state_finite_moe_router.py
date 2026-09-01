"""RUNG 359 -- CONTEXT-STATE FINITE-MOE FALSIFIER AT MLP0.

The failed rung354 router used token identity.  Here the state is instead a
literal runtime function of the live contextual MLP0 input: center, project to
PCA32, normalize, and select one of four fixed nearest centroids.  Each state
owns a fixed context-RRR rank128 Left/Right expert; Down and bias are shared.

Price: 4*128*(1152+2*4608) + 4608*1152 + 1152
       + mean1152 + projection1152*32 + centers4*32 = 10,656,128 scalars.
This lies between global context-RRR p515 (10,649,088) and p516 (10,659,456).

Frozen predictions
------------------
pred_a_context_router_beats_cheaper_global_mean:
    Kmeans-router mean <=80% of global p515 on both corpora and <=.020.
pred_b_context_router_beats_cheaper_global_tails:
    Kmeans-router p95/max <=90% of global p515 on both and max <=.080.
pred_c_states_are_live_balanced_stable_and_nonrandom:
    Kmeans mean <=85% random-centroid on both; all state fractions >=.10 and
    counts >=800/fit half; fit-A/B means differ <=.010/corpus; live routing
    reproduces fit labels; literal prices assert.

Null: kmeans mean fails to beat global p515 on either corpus, OR is no better
than random-centroid on both.  No state/rank/PCA/cluster tuning follows.
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
OUT = ROOT / "mlp0_context_state_finite_moe_router_results.json"
DEV = "cuda"
D = 1152
H = 4608
STATES = 4
EXPERT_RANK = 128
PCA_RANK = 32
GLOBAL_RANKS = (515, 516)
FIT_A = (0, 48)
FIT_B = (48, 96)
FINEWEB_EVAL = (80, 120)
WIKI_SKIP = 480 * 257
EVAL_ROWS = 40
ROUTER_PRICE = (STATES * EXPERT_RANK * (D + 2 * H) + H * D + D
                + D + D * PCA_RANK + STATES * PCA_RANK)
GLOBAL_PRICES = {rank: rank * (D + 2 * H) + H * D + D for rank in GLOBAL_RANKS}


def _nearest(features: torch.Tensor, centers: torch.Tensor):
    distance = (features.square().sum(1, keepdim=True)
                + centers.square().sum(1)[None]
                - 2.0 * features @ centers.T)
    return distance.argmin(1)


def _random_feasible_centers(features: torch.Tensor, seed: int):
    """First seeded random-row center set whose four fit states have >=800 rows."""
    for offset in range(100):
        generator = torch.Generator(device=features.device).manual_seed(seed + offset)
        ids = torch.randperm(len(features), generator=generator, device=features.device)[:STATES]
        centers = features[ids].clone()
        labels = _nearest(features, centers)
        counts = torch.bincount(labels, minlength=STATES)
        if int(counts.min()) >= 800:
            return centers, labels, seed + offset
    raise RuntimeError("no feasible random-centroid router in frozen 100-seed feasibility search")


@torch.no_grad()
def _fit_router(model, x: torch.Tensor, rrr_program, kmeans, seed: int):
    mean = x.mean(0)
    centered = x - mean
    _u, _s, projection = torch.pca_lowrank(centered, q=PCA_RANK, center=False, niter=4)
    features = torch.nn.functional.normalize(centered @ projection, dim=1)
    labels, labels_again, centers = kmeans(features, features, STATES, seed)
    assert torch.equal(labels, labels_again)
    assert torch.equal(labels, _nearest(features, centers))
    random_centers, random_labels, random_seed = _random_feasible_centers(features, seed + 1000)

    def programs_for(state_labels):
        programs, counts = {}, []
        for state in range(STATES):
            member = x[state_labels == state]
            counts.append(int(len(member)))
            if len(member) < max(800, EXPERT_RANK):
                raise RuntimeError(f"state {state} has only {len(member)} fit inputs")
            covariance = member.T @ member / len(member)
            covariance = .5 * (covariance + covariance.T)
            program, basis, diagnostic = rrr_program(
                model.transformer.h[0].mlp, covariance, rank=EXPERT_RANK)
            program["_basis"] = basis
            program["_diagnostic"] = diagnostic
            programs[state] = program
        return programs, counts

    kmeans_programs, kmeans_counts = programs_for(labels)
    random_programs, random_counts = programs_for(random_labels)
    router = {"mean": mean, "projection": projection, "centers": centers}
    random_router = {"mean": mean, "projection": projection, "centers": random_centers}
    # Explicit live-route tripwires before deleting fit features.
    assert torch.equal(_nearest(torch.nn.functional.normalize((x - mean) @ projection, dim=1), centers), labels)
    assert torch.equal(_nearest(torch.nn.functional.normalize((x - mean) @ projection, dim=1), random_centers),
                       random_labels)
    return router, kmeans_programs, kmeans_counts, random_router, random_programs, random_counts, random_seed


@torch.no_grad()
def _score_context_router(model, rows, router, programs, manual_logits):
    values = []
    down = programs[0]["down"]
    bias = programs[0]["bias"]

    def hook(_module, args, output):
        x = args[0].float().reshape(-1, D)
        features = torch.nn.functional.normalize((x - router["mean"]) @ router["projection"], dim=1)
        labels = _nearest(features, router["centers"])
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
            input_ids = row[:-1].unsqueeze(0).to(DEV)
            target = row[1:].to(DEV)
            logits = manual_logits(model, input_ids)[0].float()
            values.append(float(torch.nn.functional.cross_entropy(logits, target)))
    finally:
        handle.remove()
    return torch.tensor(values, dtype=torch.float64)


def _ratio(value: float, control: float):
    return max(0.0, value) / max(1e-6, max(0.0, control))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FIT_A[1] == FIT_B[0] and FINEWEB_EVAL == (80, 120)
        assert ROUTER_PRICE == 10_656_128
        assert GLOBAL_PRICES == {515: 10_649_088, 516: 10_659_456}
        assert GLOBAL_PRICES[515] < ROUTER_PRICE < GLOBAL_PRICES[516]
        assert WIKI_SKIP == 123_360 and WIKI_SKIP + EVAL_ROWS * 257 == 133_640
        print("CONTEXT-STATE FINITE MOE | dry run: router, prices, populations, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_embedding_fold_structure_screen import _kmeans
    from mlp0_finite_moe_subspace_router_screen import _capture_inputs
    from mlp0_tail_robust_context_metric_screen import _score_rows, _summary
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    torch.manual_seed(35900)
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    fit_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    eval_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cached = eval_cached["rows"] if isinstance(eval_cached, dict) else eval_cached
    fit_a_rows = fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    fit_b_rows = fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    fineweb = eval_cached[FINEWEB_EVAL[0]:FINEWEB_EVAL[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)

    x_a, _token_a = _capture_inputs(model, fit_a_rows, _manual_logits)
    x_b, _token_b = _capture_inputs(model, fit_b_rows, _manual_logits)
    (router_a, programs_a, counts_a, random_router_a, random_programs_a,
     random_counts_a, random_seed_a) = _fit_router(model, x_a, _rrr_program, _kmeans, 35901)
    router_b, programs_b, counts_b, _rr_b, _rp_b, _rc_b, _rs_b = _fit_router(
        model, x_b, _rrr_program, _kmeans, 35902)

    def global_program(x, rank):
        covariance = x.T @ x / len(x)
        covariance = .5 * (covariance + covariance.T)
        return _rrr_program(model.transformer.h[0].mlp, covariance, rank=rank)[0]

    globals_a = {rank: global_program(x_a, rank) for rank in GLOBAL_RANKS}
    fractions_a = [count / len(x_a) for count in counts_a]
    fractions_b = [count / len(x_b) for count in counts_b]
    del x_a, x_b, _token_a, _token_b
    torch.cuda.empty_cache()

    native = {
        "fineweb": _score_rows(model, fineweb, _manual_logits),
        "wikitext": _score_rows(model, wikitext, _manual_logits),
    }
    summaries = {}
    router_arms = (
        ("kmeans_fit_a", router_a, programs_a),
        ("kmeans_fit_b", router_b, programs_b),
        ("random_centroid_fit_a", random_router_a, random_programs_a),
    )
    for name, router, programs in router_arms:
        summaries[name] = {}
        for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
            ce = _score_context_router(model, rows, router, programs, _manual_logits)
            summaries[name][corpus] = _summary(ce - native[corpus])
            print(f"{name} {corpus}: {summaries[name][corpus]}", flush=True)
    for rank, program in globals_a.items():
        name = f"global_p{rank}"
        summaries[name] = {}
        for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
            ce = _score_rows(model, rows, _manual_logits, program)
            summaries[name][corpus] = _summary(ce - native[corpus])
            print(f"{name} {corpus}: {summaries[name][corpus]}", flush=True)

    cluster = summaries["kmeans_fit_a"]
    cluster_b = summaries["kmeans_fit_b"]
    random_control = summaries["random_centroid_fit_a"]
    global_control = summaries["global_p515"]
    pred_a = all(_ratio(cluster[corpus]["mean"], global_control[corpus]["mean"]) <= .80
                 and cluster[corpus]["mean"] <= .020
                 for corpus in ("fineweb", "wikitext"))
    pred_b = all(_ratio(cluster[corpus]["p95"], global_control[corpus]["p95"]) <= .90
                 and _ratio(cluster[corpus]["max"], global_control[corpus]["max"]) <= .90
                 and cluster[corpus]["max"] <= .080
                 for corpus in ("fineweb", "wikitext"))
    pred_c = (
        all(_ratio(cluster[corpus]["mean"], random_control[corpus]["mean"]) <= .85
            for corpus in ("fineweb", "wikitext"))
        and min(fractions_a + fractions_b) >= .10
        and min(counts_a + counts_b) >= 800
        and all(abs(cluster[corpus]["mean"] - cluster_b[corpus]["mean"]) <= .010
                for corpus in ("fineweb", "wikitext"))
        and ROUTER_PRICE == 10_656_128
        and GLOBAL_PRICES[515] < ROUTER_PRICE < GLOBAL_PRICES[516]
    )
    fails_global_either = any(cluster[corpus]["mean"] >= global_control[corpus]["mean"]
                              for corpus in ("fineweb", "wikitext"))
    no_better_random_both = all(cluster[corpus]["mean"] >= random_control[corpus]["mean"]
                                for corpus in ("fineweb", "wikitext"))
    null = fails_global_either or no_better_random_both
    result = {
        "status": "mlp0_context_state_finite_moe_router_complete",
        "rung": 359,
        "claim_level": "live_context_state_two_fit_two_corpus_equal_price_screen_only",
        "program": {
            "states": STATES,
            "expert_rank": EXPERT_RANK,
            "router": "center_then_pca32_normalize_then_nearest_of_four_centroids",
            "router_mean_scalars": D,
            "router_projection_scalars": D * PCA_RANK,
            "router_centroid_scalars": STATES * PCA_RANK,
            "literal_mlp0_scalars": ROUTER_PRICE,
            "global_control_prices": GLOBAL_PRICES,
        },
        "fit": {
            "cache": "fineweb_n192_skip11000.pt",
            "fit_a": list(FIT_A), "fit_b": list(FIT_B),
            "kmeans_counts_a": counts_a, "kmeans_counts_b": counts_b,
            "kmeans_fractions_a": fractions_a, "kmeans_fractions_b": fractions_b,
            "random_centroid_counts_a": random_counts_a,
            "random_centroid_feasible_seed_a": random_seed_a,
            "live_route_reproduction": True,
        },
        "evaluation": {
            "fineweb_cache": "fineweb_n192_skip7000.pt",
            "fineweb_rows_half_open": list(FINEWEB_EVAL),
            "wikitext103_train_token_span_half_open": [WIKI_SKIP, WIKI_SKIP + EVAL_ROWS * 257],
            "dataset_fingerprint": fingerprint,
            "source_token_count": token_count,
        },
        "row_damage_summaries": summaries,
        'pred_a_context_router_beats_cheaper_global_mean': bool(pred_a),
        'pred_b_context_router_beats_cheaper_global_tails': bool(pred_b),
        'pred_c_states_are_live_balanced_stable_and_nonrandom': bool(pred_c),
        "null_context_router_has_no_equal_price_advantage": bool(null),
        "stop_rule": "no_state_rank_pca_or_cluster_tuning_after_result",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "counts_a": counts_a, "counts_b": counts_b,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP0 CONTEXT-STATE FINITE-MOE ROUTER DONE", flush=True)


if __name__ == "__main__":
    main()
