"""Enumerate checkpoint writer/reader tensors in Bilin18 causal execution order."""

# BQGATE: LIBRARY
from __future__ import annotations


ATTENTION_READERS = ("c_q", "c_k", "c_q2", "c_k2", "c_v")


class CausalCheckpointTopologyError(ValueError):
    pass


def _geometry(model):
    try:
        width = int(model.config.n_embd)
        heads = int(model.config.n_head)
        layers = int(model.config.n_layer)
        blocks = tuple(model.transformer.h)
    except (AttributeError, TypeError, ValueError):
        raise CausalCheckpointTopologyError("model lacks Bilin18 checkpoint geometry")
    if width < 1 or heads < 1 or layers < 1 or width % heads or len(blocks) != layers:
        raise CausalCheckpointTopologyError("invalid Bilin18 width/head/layer geometry")
    return width, heads, width // heads, blocks


def _shape(matrix):
    return tuple(int(value) for value in matrix.shape)


def _require_shape(label, matrix, expected):
    if getattr(matrix, "ndim", None) != 2 or _shape(matrix) != tuple(expected):
        raise CausalCheckpointTopologyError(
            f"{label} has shape {_shape(matrix)}; expected {tuple(expected)}")


def reader_interfaces(model):
    """Return every Q/K/Q2/K2/V and bilinear-MLP reader with an exact stage."""
    residual, heads, head_width, blocks = _geometry(model)
    records = []
    for layer, block in enumerate(blocks):
        for attribute in ATTENTION_READERS:
            matrix = getattr(block.attn, attribute).weight
            _require_shape(f"L{layer}:{attribute}", matrix, (residual, residual))
            for head in range(heads):
                start = head * head_width
                records.append({
                    "label": f"L{layer:02d}H{head:02d}:{attribute[2:]}",
                    "kind": "attention",
                    "layer": layer,
                    "head": head,
                    "role": attribute[2:],
                    "stage": (layer, 0),
                    "weight": matrix[start:start + head_width],
                })
        left, right = block.mlp.Left.weight, block.mlp.Right.weight
        hidden = int(left.shape[0])
        _require_shape(f"MLP{layer}:left", left, (hidden, residual))
        _require_shape(f"MLP{layer}:right", right, (hidden, residual))
        for role, matrix in (("left", left), ("right", right)):
            records.append({
                "label": f"MLP{layer:02d}:{role}",
                "kind": "mlp",
                "layer": layer,
                "head": None,
                "role": role,
                "stage": (layer, 2),
                "weight": matrix,
            })
    if len({record["label"] for record in records}) != len(records):
        raise CausalCheckpointTopologyError("reader labels are not unique")
    return tuple(records)


def writer_interfaces(model, *, include_embedding=False):
    """Return every attention-head and MLP output weight with an exact stage."""
    residual, heads, head_width, blocks = _geometry(model)
    records = []
    if include_embedding:
        embedding = model.transformer.wte.weight
        if getattr(embedding, "ndim", None) != 2 or embedding.shape[1] != residual:
            raise CausalCheckpointTopologyError(
                "embedding weight must have shape [vocabulary, residual_width]")
        records.append({
            "label": "embedding", "kind": "embedding", "layer": -1,
            "head": None, "role": "write", "stage": (-1, 3),
            "weight": embedding.transpose(0, 1),
        })
    for layer, block in enumerate(blocks):
        output = block.attn.c_proj.weight
        _require_shape(f"A{layer}:c_proj", output, (residual, residual))
        for head in range(heads):
            start = head * head_width
            records.append({
                "label": f"L{layer:02d}H{head:02d}:attn_out",
                "kind": "attention",
                "layer": layer,
                "head": head,
                "role": "write",
                "stage": (layer, 1),
                "weight": output[:, start:start + head_width],
            })
        down = block.mlp.Down.weight
        hidden = int(block.mlp.Left.weight.shape[0])
        _require_shape(f"MLP{layer}:down", down, (residual, hidden))
        records.append({
            "label": f"MLP{layer:02d}:down",
            "kind": "mlp",
            "layer": layer,
            "head": None,
            "role": "write",
            "stage": (layer, 3),
            "weight": down,
        })
    if len({record["label"] for record in records}) != len(records):
        raise CausalCheckpointTopologyError("writer labels are not unique")
    return tuple(records)


def downstream_reader_interfaces(model, *, writer_layer, writer_kind):
    """Filter reader weights to sites that can causally follow the named module write."""
    _, _, _, blocks = _geometry(model)
    if not isinstance(writer_layer, int) or writer_layer < 0 or writer_layer >= len(blocks):
        raise CausalCheckpointTopologyError("writer layer is out of range")
    if writer_kind == "attention":
        writer_stage = (writer_layer, 1)
    elif writer_kind == "mlp":
        writer_stage = (writer_layer, 3)
    else:
        raise CausalCheckpointTopologyError("writer kind must be attention or mlp")
    return tuple(record for record in reader_interfaces(model)
                 if record["stage"] > writer_stage)


def upstream_writer_interfaces(model, *, reader_layer, reader_kind,
                               include_embedding=False):
    """Filter writer weights to sites that can causally precede the named module reader."""
    _, _, _, blocks = _geometry(model)
    if not isinstance(reader_layer, int) or reader_layer < 0 or reader_layer >= len(blocks):
        raise CausalCheckpointTopologyError("reader layer is out of range")
    if reader_kind == "attention":
        reader_stage = (reader_layer, 0)
    elif reader_kind == "mlp":
        reader_stage = (reader_layer, 2)
    else:
        raise CausalCheckpointTopologyError("reader kind must be attention or mlp")
    return tuple(record for record in writer_interfaces(
        model, include_embedding=include_embedding) if record["stage"] < reader_stage)
