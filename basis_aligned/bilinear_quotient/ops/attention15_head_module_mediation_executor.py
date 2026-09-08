"""Thin execution adapter for the admitted attention-15 mediation atlas."""

from __future__ import annotations

import head_response_mediation_contract as mediation


LAYER = 15


def capture_live_response(parent, backend, context, counters, bases):
    """Run a native-attention parent arm and capture its layer-15 head responses."""
    model = backend.model
    return mediation.capture_preprojection(
        backend.torch, model.transformer.h[LAYER].attn.c_proj,
        lambda: parent.manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases, complete15=False),
        n_heads=int(model.config.n_head),
    )


def execute_absolute_response(parent, backend, context, counters, bases, absolute):
    """Run one upstream background while installing an absolute layer-15 response."""
    return mediation.execute_with_absolute_preprojection(
        backend.model.transformer.h[LAYER].attn.c_proj,
        lambda: parent.manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases, complete15=False),
        absolute,
    )


def execute_mediator_cells(parent, backend, context, counters, bases, captures, heads):
    """Reuse captured 00/11 outputs and execute only the rescue/reset hybrids."""
    tensors = mediation.build_absolute_set_cells(
        backend.torch, captures["00"], captures["11"], heads=heads,
        semantic_positions=context["base_batch"].semantic_positions)
    return {
        "00": captures["00_output"],
        "01": execute_absolute_response(parent, backend, context, counters, {}, tensors["01"]),
        "10": execute_absolute_response(parent, backend, context, counters, bases, tensors["10"]),
        "11": captures["11_output"],
    }
