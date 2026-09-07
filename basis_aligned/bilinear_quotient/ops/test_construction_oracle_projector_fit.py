from types import SimpleNamespace

import pytest
import torch

import construction_oracle_projector_fit as fit


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


def test_a1_oracle_does_not_guide_on_a2():
    backend, context = SimpleNamespace(torch=torch, F=torch.nn.functional), _context()
    good = torch.zeros(8, 3)
    good[:2, 0] = 1.0
    changed_a2 = good.clone()
    changed_a2[2:4, 0] = torch.tensor([-20.0, 20.0])
    assert fit.training_loss(backend, context, {"logits": good}, "A1") == pytest.approx(
        fit.training_loss(backend, context, {"logits": changed_a2}, "A1"))


def test_each_oracle_penalizes_failure_on_its_own_construction():
    backend, context = SimpleNamespace(torch=torch, F=torch.nn.functional), _context()
    good = torch.zeros(8, 3)
    good[:2, 0], good[2:4, 0] = 1.0, 2.0
    bad_a1, bad_a2 = good.clone(), good.clone()
    bad_a1[:2, 0], bad_a2[2:4, 0] = 0.0, 0.0
    assert fit.training_loss(backend, context, {"logits": bad_a1}, "A1") > \
        fit.training_loss(backend, context, {"logits": good}, "A1")
    assert fit.training_loss(backend, context, {"logits": bad_a2}, "A2") > \
        fit.training_loss(backend, context, {"logits": good}, "A2")


def test_unknown_panel_fails_closed_and_schedule_is_frozen():
    backend, context = SimpleNamespace(torch=torch, F=torch.nn.functional), _context()
    with pytest.raises(fit.ConstructionOracleFitError):
        fit.training_loss(backend, context, {"logits": torch.zeros(8, 3)}, "P")
    assert fit.STEPS == 8 and fit.CHECKPOINTS == (0, 4, 8)
    assert fit.CONTROL_PANELS == ("P", "C")
