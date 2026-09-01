"""RUNG 298B — DOES A STRUCTURAL PRIOR MAKE THE PLANTED MLP0 OBJECT IDENTIFIABLE?

Rung 298 found the decisive non-identifiability: a dense student reproduced the
planted support-poset teacher at held-out R2=0.99994 while its recovered support
family was exactly the no-structure baseline (Jaccard 0.281, reachability F1 0).

This follow-up trains three hard structural classes.  The router side of every
bilinear atom is restricted to q discrete state coordinates, while the other side
reads only continuous coordinates.  The learner is NOT told which states form a
node or which unit matches a teacher atom.

  spectrum: each support cardinality q in the planted family is supplied with the
            correct multiplicity, but all actual supports are learned.
  pair_only: every unit has q=2 (wrong hierarchy; can still synthesize larger sets
             by superposition, so output fit alone cannot validate structure).
  singleton: every unit has q=1 (flat finite-state MoE null).

Frozen predictions
------------------
pred_a_spectrum_recovers_structure:
    spectrum held-out R2 >= .98, family Jaccard >= .85, reachability F1 >= .85.
pred_b_prior_beats_dense_identifiability:
    spectrum family Jaccard exceeds rung-298 dense 0.28125 by >= .40 while losing
    at most .02 held-out R2 relative to 0.99994.
pred_c_wrong_priors_discriminate:
    each wrong prior either has R2 < .95 or family Jaccard < .65.

Null: spectrum reaches high output R2 but Jaccard < .65 or reachability F1 < .65.
That would mean even the correct support-size spectrum does not select the planted
poset; a stronger explicit block/DAG MDL prior is required.

This is a planted identifiability screen only.  It prices every arm at the same
64*(16+16+24)=3,584 weight scalars; zeros are structural masks, not pruned storage
until encoded as sparse supports plus nonzero values.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from mlp0_embedding_fold_structure_screen import (
    DEV,
    TOY_D,
    TOY_H,
    TOY_OUT,
    TOY_STATES,
    TOY_STEPS,
    TOY_TEST,
    TOY_TRAIN,
    TRUE_MASKS,
    _bilinear,
    _family_score,
    _infer_masks,
    _make_teacher,
    _r2,
    _toy_inputs,
)


OUT = "/workspace/tensor_language/basis_aligned/bilinear_quotient/mlp0_embedding_fold_prior_sweep_results.json"
DENSE_JACCARD = 0.28125
DENSE_R2 = 0.9999397993087769


def _project(left: torch.Tensor, right: torch.Tensor, cardinality: torch.Tensor) -> None:
    with torch.no_grad():
        left[:, TOY_STATES:] = 0
        right[:, :TOY_STATES] = 0
        state_part = left[:, :TOY_STATES]
        mask = torch.zeros_like(state_part)
        for unit, width in enumerate(cardinality.tolist()):
            index = state_part[unit].abs().topk(width).indices
            mask[unit, index] = 1
        state_part.mul_(mask)


def _fit(
    name: str,
    cardinality: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    state_test: torch.Tensor,
) -> dict[str, object]:
    seed = 29900 + ("spectrum", "pair_only", "singleton").index(name)
    generator = torch.Generator(device=DEV).manual_seed(seed)
    left = (torch.randn(TOY_H, TOY_D, generator=generator, device=DEV) / math.sqrt(TOY_D)).requires_grad_()
    right = (torch.randn(TOY_H, TOY_D, generator=generator, device=DEV) / math.sqrt(TOY_D)).requires_grad_()
    down = (torch.randn(TOY_OUT, TOY_H, generator=generator, device=DEV) / math.sqrt(TOY_H)).requires_grad_()
    _project(left, right, cardinality)
    optimizer = torch.optim.Adam((left, right, down), lr=4e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TOY_STEPS)
    target_scale = y_train.square().mean().detach().clamp_min(1e-8)
    curve = []
    for step in range(TOY_STEPS):
        index = torch.randint(0, TOY_TRAIN, (1024,), generator=generator, device=DEV)
        prediction, _ = _bilinear(x_train[index], down, left, right)
        loss = F.mse_loss(prediction, y_train[index]) / target_scale
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((left, right, down), 10.0)
        optimizer.step()
        _project(left, right, cardinality)
        scheduler.step()
        if step % 250 == 0 or step == TOY_STEPS - 1:
            with torch.no_grad():
                prediction_test, _ = _bilinear(x_test, down, left, right)
                curve.append({"step": step, "heldout_r2": _r2(prediction_test, y_test)})
    with torch.no_grad():
        prediction, hidden = _bilinear(x_test, down, left, right)
        family = _family_score(_infer_masks(hidden, state_test))
    nonzero = int((left != 0).sum() + (right != 0).sum() + (down != 0).sum())
    return {
        "heldout_r2": _r2(prediction, y_test),
        "family": family,
        "structural_nonzero_scalars": nonzero,
        "dense_shape_scalars": TOY_H * (2 * TOY_D + TOY_OUT),
        "curve": curve,
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print("MLP0 EMBEDDING FOLD PRIOR SWEEP | dry run valid")
        return
    started = time.time()
    down_teacher, left_teacher, right_teacher = _make_teacher(29800)
    x_train, _ = _toy_inputs(TOY_TRAIN, 29801)
    x_test, state_test = _toy_inputs(TOY_TEST, 29802)
    with torch.no_grad():
        y_train, _ = _bilinear(x_train, down_teacher, left_teacher, right_teacher)
        y_test, _ = _bilinear(x_test, down_teacher, left_teacher, right_teacher)
    spectrum = []
    for mask in TRUE_MASKS:
        spectrum.extend([len(mask)] * 4)
    cards = {
        "spectrum": torch.tensor(spectrum, device=DEV),
        "pair_only": torch.full((TOY_H,), 2, device=DEV),
        "singleton": torch.ones(TOY_H, dtype=torch.long, device=DEV),
    }
    arms = {
        name: _fit(name, cardinality, x_train, y_train, x_test, y_test, state_test)
        for name, cardinality in cards.items()
    }
    selected = arms["spectrum"]
    pred_a = (
        selected["heldout_r2"] >= 0.98
        and selected["family"]["mean_best_jaccard"] >= 0.85
        and selected["family"]["reachability_f1"] >= 0.85
    )
    pred_b = (
        selected["family"]["mean_best_jaccard"] - DENSE_JACCARD >= 0.40
        and DENSE_R2 - selected["heldout_r2"] <= 0.02
    )
    pred_c = all(
        arm["heldout_r2"] < 0.95 or arm["family"]["mean_best_jaccard"] < 0.65
        for name, arm in arms.items() if name != "spectrum"
    )
    result = {
        "status": "mlp0_embedding_fold_prior_sweep_complete",
        "rung": "298B",
        "arms": arms,
        'pred_a_spectrum_recovers_structure': bool(pred_a),
        'pred_b_prior_beats_dense_identifiability': bool(pred_b),
        'pred_c_wrong_priors_discriminate': bool(pred_c),
        "null_spectrum_nonidentifying": bool(
            selected["heldout_r2"] >= 0.98
            and (
                selected["family"]["mean_best_jaccard"] < 0.65
                or selected["family"]["reachability_f1"] < 0.65
            )
        ),
        "runtime_s": time.time() - started,
    }
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
