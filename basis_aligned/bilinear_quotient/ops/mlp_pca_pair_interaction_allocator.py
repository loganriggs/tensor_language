"""RUNG 307 -- PAIR-INTERACTION ALLOCATION FOR FOUR PCA-COMPRESSED MLPS.

Rung 306 established broad single-layer rank-256 PCA compressibility but found
1.6--1.8x super-additive four-layer damage and calibration-rank instability.
Replace scalar layer ranking by an explicit quadratic interaction model.

Using the unchanged rank-256 PCA bases, score every one of the 153 layer pairs
on two disjoint eight-row halves of FineWeb skip7000.  For half h define

    s_ij^h = damage({i,j}) - damage({i}) - damage({j}).

For each of 3,060 quartets, predict damage by mean individual damage plus all
six mean pair excesses, and add |s_ij^A-s_ij^B| for every pair as a risk penalty
against non-replicating attractive interactions.  Freeze the minimum-risk
quartet before scoring FineWeb skip11000 or WikiText test skip30000.

The literal price is unchanged: four rank-256 factorized Down maps save
15,335,424 scalars.  Validation compares the selected quartet with rung306's
naive scalar-selected {6,7,9,12} and fixed-spaced {0,5,11,17} quartets.

Frozen predictions
------------------
pred_a_pair_interactions_replicate:
    Pair-damage Spearman between calibration halves >=.50 and pair-excess
    Spearman >=.10.
pred_b_interaction_selected_quartet_is_predictive:
    Selected quartet damage <=.08 FineWeb and <=.10 WikiText.
pred_c_allocator_beats_controls_and_composes:
    Selected damage is <=1.5 times summed nonnegative selected single-layer
    validation damage and >=15% smaller than BOTH control quartets on BOTH
    validation corpora (control damages nonnegative).

Null: selected damage >=.15 on either validation corpus, or calibration-half
pair-damage Spearman <=.10.  A pass remains a composition screen only.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp_pca_pair_interaction_allocator_results.json"
LAYERS = 18
RANK = 256
FIT_ROWS = 16
CAL_ROWS = 16
EVAL_ROWS = 8
NAIVE_CONTROL = (6, 7, 9, 12)
FIXED_CONTROL = (0, 5, 11, 17)
WIKI_SKIP = 30000


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for name in ("fineweb_n480_skip80.pt", "fineweb_n192_skip7000.pt",
                     "fineweb_n192_skip11000.pt"):
            assert (ROOT / ".rowcache" / name).exists()
        assert len(list(itertools.combinations(range(LAYERS), 2))) == 153
        assert len(list(itertools.combinations(range(LAYERS), 4))) == 3060
        print("MLP PCA PAIR INTERACTION ALLOCATOR | dry run: populations, pairs, quartets, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_activation_pca_four_layer_composition as base
    from mlp0_signed_response_rank_screen import _manual_logits, _wikitext_rows
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == base.D and len(model.transformer.h) == LAYERS
    fit = base._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    calibration = base._load_rows(ROOT / ".rowcache/fineweb_n192_skip7000.pt", CAL_ROWS)
    halves = {"a": calibration[:8], "b": calibration[8:16]}
    validation = base._load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", EVAL_ROWS)
    wikitext, fingerprint = _wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)
    pca = base._fit_pca(base._capture_outputs(model, fit, _manual_logits))
    native_half = {name: base._score(model, rows, {}, _manual_logits) for name, rows in halves.items()}
    individual: dict[str, dict[str, float]] = {}
    for layer in range(LAYERS):
        projector = {layer: pca[layer]}
        individual[str(layer)] = {
            name: base._score(model, rows, projector, _manual_logits) - native_half[name]
            for name, rows in halves.items()
        }

    pair_rows: dict[str, dict[str, object]] = {}
    pairs = list(itertools.combinations(range(LAYERS), 2))
    for index, (left, right) in enumerate(pairs):
        projector = {left: pca[left], right: pca[right]}
        damage = {
            name: base._score(model, rows, projector, _manual_logits) - native_half[name]
            for name, rows in halves.items()
        }
        excess = {
            name: damage[name] - individual[str(left)][name] - individual[str(right)][name]
            for name in halves
        }
        pair_rows[f"{left}_{right}"] = {
            "layers": [left, right], "damage": damage, "excess_over_additive": excess,
            "mean_excess": 0.5 * (excess["a"] + excess["b"]),
            "replication_penalty": abs(excess["a"] - excess["b"]),
        }
        if (index + 1) % 25 == 0 or index + 1 == len(pairs):
            print(f"scored pairs {index + 1}/{len(pairs)}", flush=True)

    pair_damage_rho = base._spearman(
        [pair_rows[f"{i}_{j}"]["damage"]["a"] for i, j in pairs],
        [pair_rows[f"{i}_{j}"]["damage"]["b"] for i, j in pairs])
    pair_excess_rho = base._spearman(
        [pair_rows[f"{i}_{j}"]["excess_over_additive"]["a"] for i, j in pairs],
        [pair_rows[f"{i}_{j}"]["excess_over_additive"]["b"] for i, j in pairs])

    def quartet_score(quartet: tuple[int, ...]) -> tuple[float, float, float]:
        individual_mean = sum(0.5 * (individual[str(layer)]["a"] + individual[str(layer)]["b"])
                              for layer in quartet)
        interaction_mean = 0.0
        risk_penalty = 0.0
        for left, right in itertools.combinations(quartet, 2):
            row = pair_rows[f"{left}_{right}"]
            interaction_mean += row["mean_excess"]
            risk_penalty += row["replication_penalty"]
        return individual_mean + interaction_mean + risk_penalty, interaction_mean, risk_penalty

    quartets = list(itertools.combinations(range(LAYERS), 4))
    scored_quartets = [(quartet_score(quartet), quartet) for quartet in quartets]
    scored_quartets.sort(key=lambda item: item[0][0])
    selected_score, selected = scored_quartets[0]
    native_validation = base._score(model, validation, {}, _manual_logits)
    native_wiki = base._score(model, wikitext, {}, _manual_logits)
    validation_sets = {"interaction_selected": selected,
                       "naive_scalar_control": NAIVE_CONTROL,
                       "fixed_spaced_control": FIXED_CONTROL}
    compositions = {}
    for name, layers in validation_sets.items():
        projector = {layer: pca[layer] for layer in layers}
        compositions[name] = {
            "layers": list(layers),
            "fineweb_damage": base._score(model, validation, projector, _manual_logits) - native_validation,
            "wikitext_damage": base._score(model, wikitext, projector, _manual_logits) - native_wiki,
        }
        print(f"{name} {layers}: FW/WT {compositions[name]['fineweb_damage']:+.4f}/"
              f"{compositions[name]['wikitext_damage']:+.4f}", flush=True)

    selected_individual_validation = {}
    for layer in selected:
        projector = {layer: pca[layer]}
        selected_individual_validation[str(layer)] = {
            "fineweb_damage": base._score(model, validation, projector, _manual_logits) - native_validation,
            "wikitext_damage": base._score(model, wikitext, projector, _manual_logits) - native_wiki,
        }
    additive = {
        corpus: sum(max(selected_individual_validation[str(layer)][f"{corpus}_damage"], 0.0)
                    for layer in selected)
        for corpus in ("fineweb", "wikitext")
    }
    chosen = compositions["interaction_selected"]
    ratios = {corpus: chosen[f"{corpus}_damage"] / max(additive[corpus], 1e-12)
              for corpus in ("fineweb", "wikitext")}
    pred_a = pair_damage_rho >= 0.50 and pair_excess_rho >= 0.10
    pred_b = chosen["fineweb_damage"] <= 0.08 and chosen["wikitext_damage"] <= 0.10
    pred_c = bool(
        ratios["fineweb"] <= 1.5 and ratios["wikitext"] <= 1.5
        and all(
            compositions[control][f"{corpus}_damage"] >= 0
            and chosen[f"{corpus}_damage"] <= 0.85 * compositions[control][f"{corpus}_damage"]
            for control in ("naive_scalar_control", "fixed_spaced_control")
            for corpus in ("fineweb", "wikitext")
        )
    )
    null = bool(chosen["fineweb_damage"] >= 0.15 or chosen["wikitext_damage"] >= 0.15
                or pair_damage_rho <= 0.10)
    native_price = 3 * base.H * base.D + base.D
    factor_price = 2 * base.H * base.D + RANK * (base.H + base.D) + base.D
    result = {
        "status": "mlp_pca_pair_interaction_allocator_complete",
        "rung": 307,
        "claim_level": "two_half_calibrated_pairwise_allocator_two_corpus_screen_only",
        "price": {"native_mlp_scalars_each": native_price,
                  "factorized_mlp_scalars_each": factor_price,
                  "selected_layers": 4,
                  "total_saving_scalars": 4 * (native_price - factor_price)},
        "fit": {"pca_rows": FIT_ROWS, "calibration_rows": CAL_ROWS,
                "calibration_halves": [8, 8], "pair_count": len(pairs),
                "quartet_count": len(quartets), "validation_used_for_selection": False},
        "evaluation": {"fineweb_skip": 11000, "fineweb_rows": EVAL_ROWS,
                       "wikitext_skip": WIKI_SKIP, "wikitext_rows": EVAL_ROWS,
                       "wikitext_fingerprint": fingerprint},
        "calibration": {"individual": individual, "pairs": pair_rows,
                        "pair_damage_half_spearman": pair_damage_rho,
                        "pair_excess_half_spearman": pair_excess_rho},
        "selection": {"selected_layers": list(selected),
                      "risk_adjusted_score": selected_score[0],
                      "mean_interaction_component": selected_score[1],
                      "replication_penalty": selected_score[2],
                      "top5": [{"layers": list(quartet), "score": score[0]}
                               for score, quartet in scored_quartets[:5]]},
        "native": {"fineweb_ce": native_validation, "wikitext_ce": native_wiki},
        "compositions": compositions,
        "selected_individual_validation": selected_individual_validation,
        "selected_additive_nonnegative_prediction": additive,
        "selected_composition_ratio": ratios,
        'pred_a_pair_interactions_replicate': bool(pred_a),
        'pred_b_interaction_selected_quartet_is_predictive': bool(pred_b),
        'pred_c_allocator_beats_controls_and_composes': bool(pred_c),
        "null_pair_allocator_failure": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected": selected, "pair_rhos": [pair_damage_rho, pair_excess_rho],
                      "ratios": ratios, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP PCA PAIR INTERACTION ALLOCATOR DONE", flush=True)


if __name__ == "__main__":
    main()
