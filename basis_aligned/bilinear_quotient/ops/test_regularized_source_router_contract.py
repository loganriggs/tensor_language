import torch

import regularized_source_router_contract as router


def toy():
    features=[]; labels=[]; groups=[]
    centers=(torch.tensor([2.,0.,0.]),torch.tensor([0.,2.,0.]),torch.tensor([0.,0.,2.]))
    for group in range(4):
        for label,center in enumerate(centers):
            features.append(center+torch.tensor([group*.01,-group*.01,0.]))
            labels.append(label); groups.append(group)
    return torch.stack(features),torch.tensor(labels),torch.tensor(groups)


def test_nested_ridge_selects_from_grid_and_generalizes():
    features,labels,groups=toy()
    fitted,report=router.nested_select(torch,features,labels,groups)
    prediction,scores=router.predict(torch,features,fitted)
    assert report["selected_penalty"] in router.PENALTIES
    assert len(report["grid"])==len(router.PENALTIES)
    assert scores.shape==(12,3)
    assert torch.equal(prediction,labels)


def test_selection_prioritizes_control_false_positives_then_accuracy():
    features,labels,groups=toy()
    _fit,report=router.nested_select(torch,features,labels,groups,penalties=(.001,10.0))
    selected=report["selected_nested_report"]
    assert selected["control_predicted_target_count"]==min(
        item["control_predicted_target_count"] for item in report["grid"])


def test_fit_has_literal_parameter_shape():
    features,labels,_groups=toy()
    fitted=router.fit(torch,features,labels,.1)
    assert fitted["weights"].shape==(features.shape[1]+1,3)
