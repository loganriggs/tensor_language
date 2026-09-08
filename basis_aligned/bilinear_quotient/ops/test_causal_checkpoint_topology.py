from types import SimpleNamespace

import pytest
import torch

import causal_checkpoint_topology as target


def linear(output, input_):
    return SimpleNamespace(weight=torch.zeros(output, input_))


def fake_model(*, layers=3, residual=6, heads=2, hidden=8, vocabulary=10):
    blocks = []
    for _layer in range(layers):
        attention = SimpleNamespace(
            c_proj=linear(residual, residual),
            c_q=linear(residual, residual), c_k=linear(residual, residual),
            c_q2=linear(residual, residual), c_k2=linear(residual, residual),
            c_v=linear(residual, residual),
        )
        mlp = SimpleNamespace(
            Left=linear(hidden, residual), Right=linear(hidden, residual),
            Down=linear(residual, hidden),
        )
        blocks.append(SimpleNamespace(attn=attention, mlp=mlp))
    return SimpleNamespace(
        config=SimpleNamespace(n_embd=residual, n_head=heads, n_layer=layers),
        transformer=SimpleNamespace(
            h=blocks, wte=SimpleNamespace(weight=torch.zeros(vocabulary, residual))
        ),
    )


def test_complete_interface_inventory_has_stable_counts_labels_and_shapes():
    model = fake_model()
    readers = target.reader_interfaces(model)
    writers = target.writer_interfaces(model)
    assert len(readers) == 3 * (5 * 2 + 2) == 36
    assert len(writers) == 3 * (2 + 1) == 9
    assert len(target.writer_interfaces(model, include_embedding=True)) == 10
    assert len({row["label"] for row in readers}) == len(readers)
    assert len({row["label"] for row in writers}) == len(writers)
    assert next(row for row in readers if row["label"] == "L01H01:v")["weight"].shape == (3, 6)
    assert next(row for row in writers if row["label"] == "L01H01:attn_out")["weight"].shape == (6, 3)
    assert next(row for row in writers if row["label"] == "MLP01:down")["weight"].shape == (6, 8)


def test_attention_write_can_reach_same_block_mlp_and_all_later_readers():
    labels = {row["label"] for row in target.downstream_reader_interfaces(
        fake_model(), writer_layer=1, writer_kind="attention"
    )}
    assert {"MLP01:left", "MLP01:right"} <= labels
    assert any(label.startswith("L02H") for label in labels)
    assert "L01H00:q" not in labels
    assert not any(label.startswith("L00") or label.startswith("MLP00") for label in labels)
    assert len(labels) == 14


def test_mlp_write_can_reach_only_later_blocks():
    labels = {row["label"] for row in target.downstream_reader_interfaces(
        fake_model(), writer_layer=1, writer_kind="mlp"
    )}
    assert len(labels) == 12
    assert all(label.startswith("L02") or label.startswith("MLP02") for label in labels)


def test_attention_and_mlp_readers_have_different_same_block_upstream_sets():
    model = fake_model()
    attention = {row["label"] for row in target.upstream_writer_interfaces(
        model, reader_layer=1, reader_kind="attention", include_embedding=True
    )}
    mlp = {row["label"] for row in target.upstream_writer_interfaces(
        model, reader_layer=1, reader_kind="mlp", include_embedding=True
    )}
    assert attention == {"embedding", "L00H00:attn_out", "L00H01:attn_out", "MLP00:down"}
    assert mlp == attention | {"L01H00:attn_out", "L01H01:attn_out"}


@pytest.mark.parametrize(
    "call",
    (
        lambda: target.downstream_reader_interfaces(
            fake_model(), writer_layer=3, writer_kind="attention"
        ),
        lambda: target.downstream_reader_interfaces(
            fake_model(), writer_layer=1, writer_kind="unknown"
        ),
        lambda: target.upstream_writer_interfaces(
            fake_model(), reader_layer=-1, reader_kind="attention"
        ),
        lambda: target.upstream_writer_interfaces(
            fake_model(), reader_layer=1, reader_kind="unknown"
        ),
        lambda: target.reader_interfaces(fake_model(residual=7, heads=2)),
    ),
)
def test_topology_fails_closed_on_invalid_sites_or_geometry(call):
    with pytest.raises(target.CausalCheckpointTopologyError):
        call()
