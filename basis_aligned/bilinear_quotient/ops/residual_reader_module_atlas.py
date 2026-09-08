"""Reusable complete-module capture and reciprocal patching for residual reader atlases."""
from __future__ import annotations


class ResidualReaderAtlasError(RuntimeError):
    pass


def module_targets(model, *, first_layer=12, last_layer=17):
    """Return ordered complete attention-write and MLP-write boundaries."""
    if not (0 <= first_layer <= last_layer < len(model.transformer.h)):
        raise ResidualReaderAtlasError("downstream layer interval is invalid")
    targets = {}
    for layer in range(first_layer, last_layer + 1):
        targets[f"A{layer}"] = model.transformer.h[layer].attn.c_proj
        targets[f"M{layer}"] = model.transformer.h[layer].mlp
    return targets


def replace_prefix(output, replacement, semantic_positions):
    """Replace every row through its inclusive semantic query position."""
    if output.shape != replacement.shape or output.shape[0] != len(semantic_positions):
        raise ResidualReaderAtlasError("module replacement shape or row count changed")
    changed = output.clone()
    for index, query in enumerate(semantic_positions):
        query = int(query)
        if query < 0 or query >= output.shape[1]:
            raise ResidualReaderAtlasError("semantic position is outside module output")
        changed[index, :query + 1] = replacement[index, :query + 1].to(changed)
    return changed


def capture_complete_modules(model, execute, *, first_layer=12, last_layer=17):
    """Execute once and capture every complete downstream attention/MLP write."""
    targets = module_targets(model, first_layer=first_layer, last_layer=last_layer)
    saved, calls, handles = {}, {name: 0 for name in targets}, []
    for name, target in targets.items():
        def capture(_module, _args, output, name=name):
            calls[name] += 1
            if not hasattr(output, "detach"):
                raise ResidualReaderAtlasError(f"{name} output is not a tensor")
            saved[name] = output.detach().clone()
        handles.append(target.register_forward_hook(capture))
    try:
        result = execute()
    finally:
        for handle in handles:
            handle.remove()
    if set(saved) != set(targets) or any(value != 1 for value in calls.values()):
        raise ResidualReaderAtlasError(f"complete-module capture coverage changed: {calls}")
    return result, saved, calls


def execute_with_module_patch(model, execute, *, label, replacement,
                              semantic_positions, first_layer=12, last_layer=17):
    """Execute once while replacing one complete module's inclusive-prefix output."""
    targets = module_targets(model, first_layer=first_layer, last_layer=last_layer)
    if label not in targets:
        raise ResidualReaderAtlasError(f"unknown downstream module: {label}")
    calls = {label: 0}
    def patch(_module, _args, output):
        calls[label] += 1
        return replace_prefix(output, replacement, semantic_positions)
    handle = targets[label].register_forward_hook(patch)
    try:
        result = execute()
    finally:
        handle.remove()
    if calls[label] != 1:
        raise ResidualReaderAtlasError(f"module patch count changed: {calls}")
    return result, calls


def select_reciprocal_candidates(reports, *, fit_projection, fit_cosine, maximum):
    """Freeze FIT candidates by the weaker of transfer sufficiency and reset necessity."""
    if maximum < 1:
        raise ResidualReaderAtlasError("maximum candidates must be positive")
    eligible = []
    for label, report in reports.items():
        transfer, reset = report["transfer"], report["reset"]
        score = min(float(transfer["signed_projection"]), float(reset["signed_projection"]))
        if (score >= fit_projection and float(transfer["cosine"]) >= fit_cosine
                and float(reset["cosine"]) >= fit_cosine):
            eligible.append((score, label))
    return [label for _score, label in sorted(eligible, key=lambda item: (-item[0], item[1]))[:maximum]]
