"""RUNG 298 NEGATIVE CONTROL — false-positive rate of the support-poset recovery.

Generate a dense random bilinear teacher whose Left and Right factors read every
one of eight discrete state coordinates.  It has no planted block, hierarchy,
DAG, or fixed-state expert supports.  Apply the exact support-profile recovery
and planted-family scorer from rung 298 anyway.

Frozen predictions:
pred_a_no_false_family: mean best Jaccard to the planted family <= 0.40.
pred_b_no_false_dag: recovered inclusion-reachability F1 <= 0.20.
pred_c_shuffle_inert: shuffling state labels changes mean best Jaccard by <= 0.05.

Null: Jaccard >= 0.85 and reachability F1 >= 0.85 would be a structural false
positive and would invalidate rung 298's recovery instrument.

CPU-only; no model, corpus, or GPU access.
"""

import json
import os

import torch

from mlp0_embedding_fold_structure_screen import (
    TOY_D,
    TOY_H,
    TOY_STATES,
    _family_score,
    _infer_masks,
)


OUT = "/workspace/tensor_language/basis_aligned/bilinear_quotient/mlp0_embedding_fold_negative_control_results.json"
SAMPLES = 16384


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print("MLP0 EMBEDDING FOLD NEGATIVE CONTROL | dry run valid")
        return
    generator = torch.Generator().manual_seed(29890)
    state = torch.randint(0, TOY_STATES, (SAMPLES,), generator=generator)
    onehot = torch.nn.functional.one_hot(state, TOY_STATES).float()
    continuous = torch.randn(SAMPLES, TOY_D - TOY_STATES, generator=generator)
    inputs = torch.cat((onehot, continuous), 1)
    left = torch.randn(TOY_H, TOY_D, generator=generator) / TOY_D ** 0.5
    right = torch.randn(TOY_H, TOY_D, generator=generator) / TOY_D ** 0.5
    hidden = (inputs @ left.T) * (inputs @ right.T)
    score = _family_score(_infer_masks(hidden, state))
    shuffled = state[torch.randperm(SAMPLES, generator=generator)]
    shuffled_score = _family_score(_infer_masks(hidden, shuffled))
    pred_a = score["mean_best_jaccard"] <= 0.40
    pred_b = score["reachability_f1"] <= 0.20
    pred_c = abs(score["mean_best_jaccard"] - shuffled_score["mean_best_jaccard"]) <= 0.05
    result = {
        "status": "mlp0_embedding_fold_negative_control_complete",
        "rung": 298,
        "object": "dense random bilinear teacher with no discrete supports",
        "score": score,
        "shuffled_state_score": shuffled_score,
        'pred_a_no_false_family': bool(pred_a),
        'pred_b_no_false_dag': bool(pred_b),
        'pred_c_shuffle_inert': bool(pred_c),
        "null_structural_false_positive": bool(
            score["mean_best_jaccard"] >= 0.85 and score["reachability_f1"] >= 0.85
        ),
    }
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
