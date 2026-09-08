"""Deterministic cross-fit nearest-centroid routing for entry12 state programs."""

from __future__ import annotations

import entry12_response_basis_contract as basis_contract


CLASSES = ("A1", "A2", "off")


class RouterError(ValueError):
    pass


def labels_for_rows(torch, rows, *, device):
    mapping = {"A1": 0, "A2": 1, "P": 2, "C": 2}
    try:
        return torch.tensor([mapping[row["transform_id"]] for row in rows], device=device)
    except KeyError as error:
        raise RouterError("unknown transform label") from error


def semantic_features(torch, states, semantic_positions):
    if states.ndim != 3 or len(semantic_positions) != states.shape[0]:
        raise RouterError("states must be [row,token,width] with one position per row")
    row = torch.arange(states.shape[0], device=states.device)
    position = torch.tensor(semantic_positions, device=states.device, dtype=torch.long)
    if bool(((position < 0) | (position >= states.shape[1])).any()):
        raise RouterError("semantic position out of range")
    return states[row, position].float()


def source_delta_features(torch, off, expert_states, semantic_positions):
    """Concatenate proposed A1/A2 write deltas without reading outcomes or labels."""
    if set(expert_states) != {"A1", "A2"}:
        raise RouterError("source features require exactly A1 and A2 expert states")
    base = semantic_features(torch, off, semantic_positions)
    pieces = [semantic_features(torch, expert_states[name], semantic_positions) - base
              for name in ("A1", "A2")]
    return torch.cat(pieces, dim=-1)


def fit_centroids(torch, features, labels):
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise RouterError("feature/label shape mismatch")
    mean = features.float().mean(0)
    centered = features.float() - mean
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    centroids = []
    for label in range(len(CLASSES)):
        selected = normalized[labels == label]
        if not len(selected):
            raise RouterError("every router class must be present")
        centroid = selected.mean(0)
        centroids.append(centroid / centroid.norm().clamp_min(1e-12))
    return {"mean": mean, "centroids": torch.stack(centroids)}


def predict(torch, features, fit):
    centered = features.float() - fit["mean"]
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = normalized @ fit["centroids"].T
    return scores.argmax(-1), scores


def routed_absolute(torch, off, expert_states, basis, predictions, semantic_positions):
    if set(expert_states) != {"A1", "A2"} or predictions.shape != (off.shape[0],):
        raise RouterError("expert states or predictions have wrong shape")
    selected = off.clone()
    for row, label in enumerate(predictions.detach().cpu().tolist()):
        if label not in (0, 1, 2):
            raise RouterError("router label out of range")
        if label < 2:
            selected[row] = expert_states[CLASSES[label]][row].to(selected)
    return basis_contract.projected_absolute(
        torch, off, selected, basis, semantic_positions)


def classification_report(torch, truth, prediction):
    confusion = [[int(((truth == actual) & (prediction == predicted)).sum())
                  for predicted in range(len(CLASSES))] for actual in range(len(CLASSES))]
    recalls = [confusion[index][index] / max(1, sum(confusion[index]))
               for index in range(len(CLASSES))]
    control_target_count = int(((truth == 2) & (prediction != 2)).sum())
    return {"classes": list(CLASSES), "confusion": confusion, "recalls": recalls,
            "macro_accuracy": sum(recalls) / len(recalls),
            "control_predicted_target_count": control_target_count}
