"""Install exact cached module-response deltas as a typed, jointly executed program."""

# BQGATE: LIBRARY
from __future__ import annotations


class CachedResponseProgramError(RuntimeError):
    pass


def intervene_cached_response_program(backend, batch, base_cache, changed_cache, components):
    """Add writer-minus-base responses at the semantic position for typed components.

    Attention components are ``("attn", layer, heads)`` and modify the head-concatenated
    input of c_proj. MLP components are ``("mlp", layer, ())`` and modify the MLP output.
    All hooks coexist during one real forward, so every reported combination is measured.
    """
    components = tuple((str(kind), int(layer), tuple(int(h) for h in heads))
                       for kind, layer, heads in components)
    if not components or len(components) != len(set(components)):
        raise CachedResponseProgramError("components must be nonempty and unique")
    if [(layer, 0 if kind == "attn" else 1) for kind, layer, _heads in components] != sorted(
            (layer, 0 if kind == "attn" else 1) for kind, layer, _heads in components):
        raise CachedResponseProgramError("components must be in causal layer/module order")
    head_count = int(backend.model.config.n_head)
    width = int(backend.model.config.n_embd) // head_count
    seen_modules = set()
    for kind, layer, heads in components:
        if kind not in {"attn", "mlp"} or not 0 <= layer < len(backend.model.transformer.h):
            raise CachedResponseProgramError("component kind or layer is invalid")
        if (kind, layer) in seen_modules:
            raise CachedResponseProgramError("at most one component may target each module")
        seen_modules.add((kind, layer))
        if kind == "attn" and (not heads or len(heads) != len(set(heads))
                               or any(not 0 <= head < head_count for head in heads)):
            raise CachedResponseProgramError("attention heads are invalid")
        if kind == "mlp" and heads:
            raise CachedResponseProgramError("MLP components do not take heads")

    def delta(row_id, site_id, value):
        base = base_cache.get((row_id, site_id))
        changed = changed_cache.get((row_id, site_id))
        if base is None or changed is None or tuple(base.shape) != tuple(changed.shape):
            raise CachedResponseProgramError(f"missing or mismatched cache for {row_id}/{site_id}")
        return (changed - base).to(device=value.device, dtype=value.dtype)

    handles = []
    for kind, layer, heads in components:
        block = backend.model.transformer.h[layer]
        if kind == "attn":
            def patch_attention(_module, arguments, layer=layer, heads=heads):
                value = arguments[0]
                modified = value.clone()
                for index, (row_id, position) in enumerate(
                        zip(batch.row_ids, batch.semantic_positions)):
                    for head in heads:
                        site_id = f"attn:{layer:02d}:head:{head:02d}"
                        start, stop = head * width, (head + 1) * width
                        modified[index, position, start:stop] += delta(row_id, site_id, value)
                return (modified,) + tuple(arguments[1:])
            handles.append(block.attn.c_proj.register_forward_pre_hook(patch_attention))
        else:
            def patch_mlp(_module, _arguments, output, layer=layer):
                modified = output.clone()
                site_id = f"mlp:{layer:02d}"
                for index, (row_id, position) in enumerate(
                        zip(batch.row_ids, batch.semantic_positions)):
                    modified[index, position] += delta(row_id, site_id, output)
                return modified
            handles.append(block.mlp.register_forward_hook(patch_mlp))
    try:
        return backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
