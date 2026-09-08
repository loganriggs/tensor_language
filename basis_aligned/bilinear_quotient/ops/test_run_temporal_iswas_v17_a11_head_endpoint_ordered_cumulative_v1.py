from types import SimpleNamespace

import numpy as np
import torch

import run_temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1 as run


def test_removed_head_replacement_is_exact_and_local():
    current = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    absent = -current
    changed = run.replace_removed_heads(current, absent, (1, 3), ((0, 2), (1,)), n_heads=4)
    for row, positions in enumerate(((0, 2), (1,))):
        for position in range(3):
            for head in range(4):
                sl = slice(2 * head, 2 * head + 2)
                expected = absent if position in positions and head in (1, 3) else current
                assert torch.equal(changed[row, position, sl], expected[row, position, sl])


def test_phase_masks_are_disjoint_and_exhaustive():
    rows = [{"group_number": index} for index in range(16)]
    fit, holdout = run.phase_mask(rows, "FIT"), run.phase_mask(rows, "HOLDOUT")
    assert fit.sum() == holdout.sum() == 8
    assert not np.any(fit & holdout)
    assert np.all(fit | holdout)


def test_endpoint_arm_and_price_inventory_is_exact():
    assert len(run.ENDPOINT_ARMS) == 20
    assert len({label for label, _removed in run.ENDPOINT_ARMS}) == 20
    assert run.PRICE["model_forwards"] == 3 + 20 + 8 == 31
    assert run.PRICE["sequence_evaluations"] == 31 * 16


def test_dry_authority_has_no_model_side_effects(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    monkeypatch.setattr(run.producer.Bilin18TorchBackend, "load",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model")))
    run.main()
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["authority_ok"] is True
    assert payload["model_loaded"] is False
    assert payload["price"]["model_forwards"] == 31
