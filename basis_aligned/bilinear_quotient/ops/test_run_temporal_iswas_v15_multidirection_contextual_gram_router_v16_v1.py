import torch

import run_temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1 as runner


def test_scope_and_price_are_frozen():
    assert runner.PRICE == {"native_capture_forwards": 4, "differentiable_transformer_forwards": 11,
                            "transformer_backward_forwards": 0, "model_updates": 0,
                            "example_evaluations": 656, "fit_parameters": 30}
    assert runner.finite_router.CLASSES == ("A1", "A2", "off")


def test_every_authority_is_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_logo_holds_out_whole_groups_and_reports_controls():
    rows = [{"row_id": f"r{group}_{panel}", "group_number": group, "transform_id": panel}
            for group in range(16) for panel in ("A1", "A2", "P", "C")]
    labels = runner.finite_router.labels_for_rows(torch, rows, device="cpu")
    features = torch.stack([torch.tensor([1., 0, 0, 1, 0, 0]) if label == 0
                            else torch.tensor([0., 1, 0, 0, 1, 0]) if label == 1
                            else torch.tensor([0., 0, 1, 0, 0, 1]) for label in labels])
    report = runner.logo_report(torch, rows, features, labels)
    assert report["macro_accuracy"] == 1.0
    assert report["control_predicted_target_count"] == 0


def test_finite_rejects_nested_nan():
    assert runner.finite({"x": [1.0]})
    assert not runner.finite({"x": [float("nan")]})
