from types import SimpleNamespace

import torch

import multi_construction_head_projector_fit as fit


def _context():
    logits = torch.zeros(8, 3)
    return {
        "index": torch.arange(8), "answer": torch.zeros(8, dtype=torch.long),
        "foil": torch.ones(8, dtype=torch.long), "base_margin": torch.zeros(8),
        "target_margin": torch.tensor([1., 1., 2., 2., 0., 0., 0., 0.]),
        "base_log_probs": torch.log_softmax(logits, -1),
        "panel_indices": {
            "A1": torch.tensor([0, 1]), "A2": torch.tensor([2, 3]),
            "P": torch.tensor([4, 5]), "C": torch.tensor([6, 7]),
        },
    }


def test_joint_loss_penalizes_a_single_failed_construction():
    backend = SimpleNamespace(torch=torch, F=torch.nn.functional)
    context = _context()
    good = torch.zeros(8, 3)
    good[:4, 0] = context["target_margin"][:4]
    failed = good.clone()
    failed[2:4, 0] = 0
    assert fit.joint_training_loss(backend, context, {"logits": failed}) > \
        fit.joint_training_loss(backend, context, {"logits": good})


def test_joint_loss_penalizes_either_control_panel():
    backend = SimpleNamespace(torch=torch, F=torch.nn.functional)
    context = _context()
    clean = torch.zeros(8, 3)
    clean[:4, 0] = context["target_margin"][:4]
    p_bad, c_bad = clean.clone(), clean.clone()
    p_bad[4:6, 2] = 4
    c_bad[6:8, 2] = 4
    clean_loss = fit.joint_training_loss(backend, context, {"logits": clean})
    assert fit.joint_training_loss(backend, context, {"logits": p_bad}) > clean_loss
    assert fit.joint_training_loss(backend, context, {"logits": c_bad}) > clean_loss


def test_fit_constants_match_preregistered_fixed_rank_schedule():
    assert fit.TARGET_PANELS == ("A1", "A2")
    assert fit.CONTROL_PANELS == ("P", "C")
    assert fit.CHECKPOINTS == (0, 4, 8)
    assert fit.STEPS == 8
