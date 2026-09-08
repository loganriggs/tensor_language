import torch

import entry12_finite_router_contract as router


def test_nearest_centroid_fit_predict_and_report():
    features = torch.tensor([[2., 0.], [3., 0.], [0., 2.], [0., 3.], [-2., -2.], [-3., -3.]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    fit = router.fit_centroids(torch, features, labels)
    predicted, scores = router.predict(torch, features, fit)
    report = router.classification_report(torch, labels, predicted)
    assert scores.shape == (6, 3)
    assert report["macro_accuracy"] == 1.0
    assert report["control_predicted_target_count"] == 0


def test_routed_absolute_selects_expert_or_off_by_row():
    off = torch.zeros(3, 2, 2)
    experts = {"A1": torch.ones_like(off), "A2": torch.full_like(off, 2.)}
    basis = torch.eye(2)
    output = router.routed_absolute(torch, off, experts, basis, torch.tensor([0, 1, 2]), (1, 1, 1))
    assert torch.equal(output[0], experts["A1"][0])
    assert torch.equal(output[1], experts["A2"][1])
    assert torch.equal(output[2], off[2])


def test_row_labels_merge_p_and_c_into_off():
    rows = [{"transform_id": name} for name in ("A1", "A2", "P", "C")]
    assert router.labels_for_rows(torch, rows, device="cpu").tolist() == [0, 1, 2, 2]
