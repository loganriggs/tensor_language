from __future__ import annotations

import torch

from basis_aligned.bilinear_quotient.ops.target_token_classes import (
    CLASSES,
    target_token_classes,
)


def test_induction_uses_past_match_not_future_match() -> None:
    # At j=2, 5 -> 7 occurred at p=0, so it is an induction position.  At
    # j=0 the matching 5 -> 7 lies in the future and must not be visible.
    inputs = torch.tensor([[5, 7, 5, 7]])
    targets = torch.tensor([[7, 5, 7, 9]])
    classes = target_token_classes(inputs, targets)

    assert classes["induction"].tolist() == [[False, False, True, False]]
    assert classes["repeat"].tolist() == [[False, True, False, False]]
    assert classes["novel"].tolist() == [[True, False, False, True]]


def test_labels_do_not_depend_on_future_suffix() -> None:
    left = torch.tensor([[5, 7, 5, 7, 1, 2]])
    right = torch.tensor([[5, 7, 5, 7, 5, 7]])
    targets = torch.tensor([[7, 5, 7, 9, 3, 4]])

    left_classes = target_token_classes(left, targets)
    right_classes = target_token_classes(right, targets)
    for name in CLASSES:
        assert torch.equal(left_classes[name][:, :4], right_classes[name][:, :4])


def test_partition_is_disjoint_and_exhaustive() -> None:
    generator = torch.Generator().manual_seed(1727)
    inputs = torch.randint(0, 13, (4, 32), generator=generator)
    targets = torch.randint(0, 13, (4, 32), generator=generator)
    classes = target_token_classes(inputs, targets)
    membership_count = sum(classes[name].to(torch.int8) for name in CLASSES)

    assert bool((membership_count == 1).all())
    assert not bool((classes["induction"] & classes["novel"]).any())
