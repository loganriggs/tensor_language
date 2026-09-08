import torch

import run_temporal_iswas_v15_response_shape_router_v16_v1 as runner


def test_scope_and_price_are_exact():
    assert runner.PRICE["differentiable_transformer_forwards"] == 11
    assert runner.PRICE["fit_parameters"] == 70
    assert runner.PRICE["example_evaluations"] == 656


def test_all_authorities_are_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_shape_logo_holds_out_groups():
    rows = [{"row_id": f"r{group}_{panel}", "group_number": group, "transform_id": panel}
            for group in range(16) for panel in ("A1", "A2", "P", "C")]
    labels = runner.base.finite_router.labels_for_rows(torch, rows, device="cpu")
    prototypes = torch.eye(3, 14)
    features = torch.stack([prototypes[int(label)] for label in labels])
    report = runner.logo_report(torch, rows, features, labels)
    assert report["macro_accuracy"] == 1.0
    assert report["control_predicted_target_count"] == 0
