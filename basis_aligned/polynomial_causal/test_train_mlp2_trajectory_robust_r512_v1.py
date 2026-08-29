import pytest
import torch

import train_mlp2_trajectory_robust_r512_v1 as assay


def test_relative_shift_known_answer_and_identity():
    native = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    assert assay.relative_shift(native, native) == 0.0
    shifted = native + torch.tensor([1.0, 0.0])
    assert assay.relative_shift(native, shifted) == pytest.approx(1.0)


def test_relative_shift_rejects_mismatch():
    with pytest.raises(ValueError):
        assay.relative_shift(torch.ones(2, 2), torch.ones(2, 3))


def test_frozen_price_and_training_split():
    assert assay.refit.RANK == 512
    assert assay.FIT_DOCUMENTS + assay.DEV_DOCUMENTS == 192
    assert assay.FIT_DOCUMENTS * assay.TOKENS_PER_DOCUMENT == 30_720
    assert assay.STEPS == 1200
    assert assay.BATCH_PER_BACKGROUND * 2 == 1024

