"""Nested group-cross-fitted class-balanced dual-ridge source router."""

from __future__ import annotations

import entry12_finite_router_contract as base


PENALTIES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


class RegularizedRouterError(ValueError):
    pass


def _design(torch, features, mean):
    centered = features.float() - mean
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat((normalized, torch.ones(len(normalized), 1, device=features.device)), dim=1)


def fit(torch, features, labels, penalty):
    if features.ndim != 2 or labels.shape != (len(features),) or penalty not in PENALTIES:
        raise RegularizedRouterError("invalid ridge fit inputs")
    if set(labels.detach().cpu().tolist()) != {0, 1, 2}:
        raise RegularizedRouterError("all three classes are required")
    mean = features.float().mean(0)
    design = _design(torch, features, mean)
    counts = torch.stack([(labels == label).sum() for label in range(3)]).float()
    sample_weight = (len(labels) / (3.0 * counts[labels])).sqrt().to(design)
    weighted_design = design * sample_weight[:, None]
    targets = torch.nn.functional.one_hot(labels, num_classes=3).float() * sample_weight[:, None]
    gram = weighted_design @ weighted_design.T
    scale = float(gram.trace() / len(gram))
    dual = torch.linalg.solve(
        gram + float(penalty) * scale * torch.eye(len(gram), device=gram.device), targets)
    weights = weighted_design.T @ dual
    return {"mean": mean, "weights": weights, "penalty": float(penalty),
            "effective_penalty": float(penalty) * scale}


def predict(torch, features, fitted):
    scores = _design(torch, features, fitted["mean"]) @ fitted["weights"]
    return scores.argmax(-1), scores


def nested_select(torch, features, labels, groups, penalties=PENALTIES):
    if groups.shape != labels.shape or len(set(groups.detach().cpu().tolist())) < 3:
        raise RegularizedRouterError("nested selection requires at least three groups")
    grid = []
    for penalty in penalties:
        prediction = torch.empty_like(labels)
        for group in sorted(set(groups.detach().cpu().tolist())):
            held = groups == group
            fitted = fit(torch, features[~held], labels[~held], penalty)
            prediction[held] = predict(torch, features[held], fitted)[0]
        report = base.classification_report(torch, labels, prediction)
        grid.append({"penalty": float(penalty), **report,
                     "predictions": prediction.detach().cpu().tolist()})
    selected = min(grid, key=lambda item: (
        item["control_predicted_target_count"], -item["macro_accuracy"], -item["penalty"]))
    fitted = fit(torch, features, labels, selected["penalty"])
    return fitted, {"grid": grid, "selected_penalty": selected["penalty"],
                    "selected_nested_report": selected}
