"""Install source-resolved attention-response deltas from a changed causal state."""

# BQGATE: LIBRARY
from __future__ import annotations

import attention_source_group_eval as groups


class AttentionResponseSourceError(RuntimeError):
    pass


def intervene_response_groups(
    backend,
    base_batch,
    semantic_donor_batch,
    base_capture,
    changed_capture,
    group_names,
    *,
    layer,
    selected_heads,
):
    """Add changed-minus-base P[k]V[k] response terms at the final query."""
    names = groups.validate_group_names(group_names)
    partitions = groups.batch_partitions(base_batch, semantic_donor_batch)
    head_count = int(backend.model.config.n_head)
    head_dim = int(backend.model.config.n_embd) // head_count
    heads = tuple(int(head) for head in selected_heads)
    if not heads or len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads):
        raise AttentionResponseSourceError("selected head set is invalid")
    groups._attention_shapes(backend, base_batch, base_capture)
    groups._attention_shapes(backend, base_batch, changed_capture)

    def patch(_module, arguments):
        flattened = arguments[0]
        output = flattened.clone().view(
            len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
        )
        for index, (query, partition) in enumerate(zip(base_batch.semantic_positions, partitions)):
            for position in (position for name in names for position in partition[name]):
                for head in heads:
                    base_term = (
                        base_capture["pattern"][index, head, query, position]
                        * base_capture["value"][index, position, head]
                    )
                    changed_term = (
                        changed_capture["pattern"][index, head, query, position]
                        * changed_capture["value"][index, position, head]
                    )
                    output[index, query, head] += (changed_term - base_term).to(
                        device=output.device, dtype=output.dtype
                    )
        return (output.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[int(layer)].attn.c_proj.register_forward_pre_hook(patch)
    try:
        return backend.native(base_batch, capture=False)
    finally:
        handle.remove()
