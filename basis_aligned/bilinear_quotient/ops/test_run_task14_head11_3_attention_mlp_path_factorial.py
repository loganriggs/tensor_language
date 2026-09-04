from __future__ import annotations

import pytest

import circuit_fast_screen_producer as producer
import run_task14_head11_3_attention_mlp_path_factorial as run
import run_task14_head11_3_downstream_module_reader_screen as parent_reader


class FakeBackend:
    def __init__(self, *, interaction=0.0, path_loss=0.0, control_loss=0.0, replay_shift=0.0):
        rows, self.native_pairs, self.head = run._load()
        self.family = {str(x["row_id"]): str(x["transform_id"]) for x in rows}
        self.interaction, self.path_loss = interaction, path_loss
        self.control_loss, self.replay_shift = control_loss, replay_shift
        self.scale = __import__("statistics").median(
            parent_reader._margin(self.native_pairs[(str(x["row_id"]), "donor")])
            + parent_reader._margin(self.native_pairs[(str(x["row_id"]), "base")])
            for x in rows if x["transform_id"] in {"A1", "A2"})

    def native(self, batch: producer.ModelBatch, *, capture: bool):
        pairs, cache = [], {}
        for rid in batch.row_ids:
            pair = self.native_pairs[(rid, batch.side)]; pairs.append((pair[0]+self.replay_shift, pair[1]))
            if capture:
                cache[(rid, parent_reader.HEAD_SITE)] = object()
                for site in run.MLP_SITES + run.ATTENTION_SITES: cache[(rid, site)] = object()
        return producer.BatchOutput(tuple(pairs), cache)

    def induce_and_restore(self, batch, *, restore_sites, donor_cache, recipient_cache):
        arm = "both" if len(restore_sites)==13 else "mlp" if restore_sites[0].startswith("mlp") else "attention"
        pairs = []
        for rid in batch.row_ids:
            family = self.family[rid]; base = self.native_pairs[(rid,"base")]; donor = self.native_pairs[(rid,"donor")]
            f0 = parent_reader._recovery(family, base, donor, self.head[rid], self.scale)
            loss = self.path_loss if family in {"A1","A2"} else self.control_loss
            f = f0-loss if arm != "both" else f0-2*loss+(self.interaction if family in {"A1","A2"} else 0.0)
            bm = parent_reader._margin(base)
            margin = bm-f*(parent_reader._margin(donor)+bm) if family in {"A1","A2"} else bm+f*self.scale
            pairs.append((margin, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def test_dryrun_price_and_gap():
    d = run.compile_dryrun()
    assert d["maximum_new_price"] == {"forward_calls":20,"example_evaluations":640,
        "backward_calls":0,"model_updates":0,"raw_numeric_evidence_bytes":3072}
    assert d["bars"]["additive_abs_max"] < d["bars"]["grouped_abs_min"]
    assert len(set(d["arms"]["both"])) == 13


def test_direct_residual_null():
    result = run.run_science(backend=FakeBackend(), clock=lambda:1.0)
    assert result["terminal"] == "additive_or_direct_null"
    assert result["additive_subtype"] == "direct_residual_read"
    assert result["predictions"]["pred_c_additive_or_direct_use"] is True
    assert set(result["control_mean_absolute_terms"]["P"]) == {
        "interaction", "mlp_path_loss", "attention_path_loss", "both_paths_loss"}


def test_nonlinear_cross_path_grouping():
    result = run.run_science(backend=FakeBackend(interaction=-0.2,path_loss=0.15), clock=lambda:2.0)
    assert result["terminal"] == "nonlinear_cross_path_screen"
    assert result["predictions"]["pred_b_nonlinear_cross_path_grouping"] is True
    assert all(x["interaction"] == pytest.approx(-0.2) for x in result["target_cells"].values())


def test_native_replay_failure_is_invalid():
    result = run.run_science(backend=FakeBackend(replay_shift=0.01), clock=lambda:3.0)
    assert result["terminal"] == "invalid"


def test_large_control_path_loss_blocks_additive_terminal():
    result = run.run_science(backend=FakeBackend(control_loss=0.2), clock=lambda:4.0)
    assert result["terminal"] == "inconclusive"
    assert result["predictions"]["pred_c_additive_or_direct_use"] is False


def test_output_hook_shapes_and_all_frozen_site_names():
    class Replacer:
        def __init__(self): self.sites = []
        def _replace(self, value, _batch, site, _cache):
            self.sites.append(site); return f"restored:{value}"
    backend = Replacer()
    for site in run.MLP_SITES:
        assert run._restore_output(backend, None, site, {}, "tensor") == "restored:tensor"
    for site in run.ATTENTION_SITES:
        assert run._restore_output(backend, None, site, {}, ("attention", "v1")) == (
            "restored:attention", "v1")
    assert backend.sites == list(run.MLP_SITES + run.ATTENTION_SITES)
    with pytest.raises(run.PathFactorialError, match="hook output"):
        run._restore_output(backend, None, "attn:12", {}, "not-a-tuple")
