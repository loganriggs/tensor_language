"""RUNG 308 -- FINAL THREE-LAYER PCA COMPOSITION GATE.

Rung 307's pair excess interaction replicated across calibration halves at
Spearman .917 and selected a substantially better quartet, but four layers
narrowly missed the FineWeb bar.  Reduce capacity without refitting selection.

Read rung 307's frozen individual and pair measurements.  For all 816 triples,
reuse exactly its risk-adjusted quadratic score: mean individual damage plus
mean pair excess plus |excess_A-excess_B|.  Freeze the minimum before model
execution.  Recompute the unchanged rank-256 PCA bases only to install that
program, then evaluate on untouched FineWeb skip11000 cache rows 8--15 and
WikiText-2 test after skip40000.

Three factorized Down replacements save 11,501,568 scalars. Equal-price controls
are scalar-calibration layers {6,9,12} and fixed-spaced layers {0,8,17}.

Frozen predictions
------------------
pred_a_selected_triple_is_predictive:
    Damage <=.06 FineWeb and <=.07 WikiText.
pred_b_triple_composition_is_controlled:
    Damage <=1.4 times summed nonnegative selected single-layer damage on both.
pred_c_triple_beats_equal_price_controls:
    Damage is >=15% smaller than BOTH nonnegative controls on BOTH corpora.

Null: damage >=.12 on either corpus, or the selected triple fails to beat either
whole control across both corpora.  This is the final MLP exploit screen; a pass
still needs census, certificates, exact composition billing, OOD, and signed
interventions before adoption.
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
PARENT = ROOT / "mlp_pca_pair_interaction_allocator_results.json"
OUT = ROOT / "mlp_pca_interaction_selected_triple_results.json"
LAYERS = 18
RANK = 256
FIT_ROWS = 16
EVAL_ROWS = 8
SCALAR_CONTROL = (6, 9, 12)
FIXED_CONTROL = (0, 8, 17)
WIKI_SKIP = 40000


def _triple_score(parent: dict[str, object], triple: tuple[int, ...]) -> tuple[float, float, float]:
    individual = parent["calibration"]["individual"]
    pairs = parent["calibration"]["pairs"]
    individual_mean = sum(0.5 * (individual[str(layer)]["a"] + individual[str(layer)]["b"])
                          for layer in triple)
    interaction_mean = 0.0
    penalty = 0.0
    for left, right in itertools.combinations(triple, 2):
        row = pairs[f"{left}_{right}"]
        interaction_mean += row["mean_excess"]
        penalty += row["replication_penalty"]
    return individual_mean + interaction_mean + penalty, interaction_mean, penalty


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists()
        assert len(list(itertools.combinations(range(LAYERS), 3))) == 816
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        print("MLP PCA INTERACTION-SELECTED TRIPLE | dry run: frozen parent, population, controls, bars valid")
        return

    started = time.time()
    parent = json.loads(PARENT.read_text())
    assert parent["rung"] == 307 and parent["fit"]["validation_used_for_selection"] is False
    triples = list(itertools.combinations(range(LAYERS), 3))
    scored = [(_triple_score(parent, triple), triple) for triple in triples]
    scored.sort(key=lambda item: item[0][0])
    selected_score, selected = scored[0]

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_activation_pca_four_layer_composition as base
    from mlp0_signed_response_rank_screen import _manual_logits, _wikitext_rows
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == base.D and len(model.transformer.h) == LAYERS
    fit = base._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    all_validation = base._load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", 16)
    validation = all_validation[8:16]
    wikitext, fingerprint = _wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)
    pca = base._fit_pca(base._capture_outputs(model, fit, _manual_logits))
    native_fine = base._score(model, validation, {}, _manual_logits)
    native_wiki = base._score(model, wikitext, {}, _manual_logits)
    candidates = {"interaction_selected": selected,
                  "scalar_control": SCALAR_CONTROL,
                  "fixed_spaced_control": FIXED_CONTROL}
    compositions = {}
    for name, layers in candidates.items():
        projector = {layer: pca[layer] for layer in layers}
        compositions[name] = {
            "layers": list(layers),
            "fineweb_damage": base._score(model, validation, projector, _manual_logits) - native_fine,
            "wikitext_damage": base._score(model, wikitext, projector, _manual_logits) - native_wiki,
        }
        print(f"{name} {layers}: FW/WT {compositions[name]['fineweb_damage']:+.4f}/"
              f"{compositions[name]['wikitext_damage']:+.4f}", flush=True)

    selected_individual = {}
    for layer in selected:
        projector = {layer: pca[layer]}
        selected_individual[str(layer)] = {
            "fineweb_damage": base._score(model, validation, projector, _manual_logits) - native_fine,
            "wikitext_damage": base._score(model, wikitext, projector, _manual_logits) - native_wiki,
        }
    additive = {
        corpus: sum(max(selected_individual[str(layer)][f"{corpus}_damage"], 0.0)
                    for layer in selected)
        for corpus in ("fineweb", "wikitext")
    }
    chosen = compositions["interaction_selected"]
    ratio = {corpus: chosen[f"{corpus}_damage"] / max(additive[corpus], 1e-12)
             for corpus in ("fineweb", "wikitext")}
    pred_a = chosen["fineweb_damage"] <= 0.06 and chosen["wikitext_damage"] <= 0.07
    pred_b = ratio["fineweb"] <= 1.4 and ratio["wikitext"] <= 1.4
    pred_c = all(
        compositions[control][f"{corpus}_damage"] >= 0
        and chosen[f"{corpus}_damage"] <= 0.85 * compositions[control][f"{corpus}_damage"]
        for control in ("scalar_control", "fixed_spaced_control")
        for corpus in ("fineweb", "wikitext")
    )
    beats_a_control_across_both = any(
        all(chosen[f"{corpus}_damage"] < compositions[control][f"{corpus}_damage"]
            for corpus in ("fineweb", "wikitext"))
        for control in ("scalar_control", "fixed_spaced_control")
    )
    null = bool(chosen["fineweb_damage"] >= 0.12 or chosen["wikitext_damage"] >= 0.12
                or not beats_a_control_across_both)
    native_price = 3 * base.H * base.D + base.D
    factor_price = 2 * base.H * base.D + RANK * (base.H + base.D) + base.D
    result = {
        "status": "mlp_pca_interaction_selected_triple_complete",
        "rung": 308,
        "claim_level": "frozen_pair_model_untouched_two_corpus_triple_screen_only",
        "price": {"native_mlp_scalars_each": native_price,
                  "factorized_mlp_scalars_each": factor_price,
                  "selected_layers": 3,
                  "total_saving_scalars": 3 * (native_price - factor_price)},
        "selection": {"parent_rung": 307, "candidate_count": len(triples),
                      "selected_layers": list(selected), "risk_adjusted_score": selected_score[0],
                      "mean_interaction_component": selected_score[1],
                      "replication_penalty": selected_score[2],
                      "top5": [{"layers": list(triple), "score": score[0]}
                               for score, triple in scored[:5]],
                      "new_population_used": False},
        "evaluation": {"fineweb_cache_skip": 11000, "fineweb_cache_rows": [8, 15],
                       "fineweb_rows": EVAL_ROWS, "wikitext_skip": WIKI_SKIP,
                       "wikitext_rows": EVAL_ROWS, "wikitext_fingerprint": fingerprint},
        "native": {"fineweb_ce": native_fine, "wikitext_ce": native_wiki},
        "compositions": compositions,
        "selected_individual": selected_individual,
        "selected_additive_nonnegative_prediction": additive,
        "selected_composition_ratio": ratio,
        'pred_a_selected_triple_is_predictive': bool(pred_a),
        'pred_b_triple_composition_is_controlled': bool(pred_b),
        'pred_c_triple_beats_equal_price_controls': bool(pred_c),
        "null_triple_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected": selected, "ratio": ratio,
                      "predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP PCA INTERACTION-SELECTED TRIPLE DONE", flush=True)


if __name__ == "__main__":
    main()
