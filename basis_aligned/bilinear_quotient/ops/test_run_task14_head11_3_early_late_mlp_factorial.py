from __future__ import annotations

import pytest

import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as reader
import run_task14_head11_3_early_late_mlp_factorial as run


class FakeBackend:
    def __init__(self, *, interaction=0.0, control_interaction=0.0, replay_shift=0.0):
        rows,self.native_pairs,_head,self.prior=run._load()
        self.family={str(x["row_id"]):str(x["transform_id"]) for x in rows}
        self.interaction,self.control_interaction,self.replay_shift=interaction,control_interaction,replay_shift
        self.scale=__import__("statistics").median(reader._margin(self.native_pairs[(str(x["row_id"]),"donor")])
            +reader._margin(self.native_pairs[(str(x["row_id"]),"base")])
            for x in rows if x["transform_id"] in {"A1","A2"})

    def native(self,batch:producer.ModelBatch,*,capture:bool):
        pairs=[]; cache={}
        for rid in batch.row_ids:
            pair=self.native_pairs[(rid,batch.side)]; pairs.append((pair[0]+self.replay_shift,pair[1]))
            if capture:
                cache[(rid,reader.HEAD_SITE)]=object()
                for site in run.LATE: cache[(rid,site)]=object()
        return producer.BatchOutput(tuple(pairs),cache)

    def induce_and_restore(self,batch,*,restore_sites,donor_cache,recipient_cache):
        assert tuple(restore_sites)==run.LATE
        pairs=[]
        for rid in batch.row_ids:
            f=self.prior[rid]; family=self.family[rid]
            interaction=self.interaction if family in {"A1","A2"} else self.control_interaction
            late=f["all"]-f["early"]+f["empty"]-interaction
            base=self.native_pairs[(rid,"base")]; donor=self.native_pairs[(rid,"donor")]; bm=reader._margin(base)
            margin=bm-late*(reader._margin(donor)+bm) if family in {"A1","A2"} else bm+late*self.scale
            pairs.append((margin,0.0))
        return producer.BatchOutput(tuple(pairs),{})


def test_dryrun_only_opens_late_corner():
    d=run.compile_dryrun()
    assert d["maximum_new_price"]=={"forward_calls":12,"example_evaluations":384,
        "backward_calls":0,"model_updates":0,"raw_numeric_evidence_bytes":1024}
    assert d["groups"]=={"early":["mlp:11","mlp:12"],
        "late":["mlp:13","mlp:14","mlp:15","mlp:16","mlp:17"]}


def test_exact_additive_split():
    result=run.run_science(backend=FakeBackend(),clock=lambda:1.0)
    assert result["terminal"]=="additive_split_screen"
    assert result["predictions"]["pred_c_additive_split"] is True
    assert max(abs(x["interaction"]) for x in result["target_cells"].values())<1e-6


def test_planted_nonlinear_interaction():
    result=run.run_science(backend=FakeBackend(interaction=-0.2),clock=lambda:2.0)
    assert result["terminal"]=="compensating_or_nonlinear_screen"
    assert all(x["interaction"]==pytest.approx(-0.2) for x in result["target_cells"].values())


def test_every_control_corner_is_gated():
    result=run.run_science(backend=FakeBackend(control_interaction=0.3),clock=lambda:3.0)
    assert result["terminal"]=="inconclusive"
    assert result["predictions"]["pred_c_additive_split"] is False
    assert set(result["control_mean_absolute_terms"]["P"])=={
        "empty","early","late","all","early_loss","late_loss","all_loss","interaction"}


def test_native_replay_failure_invalidates():
    result=run.run_science(backend=FakeBackend(replay_shift=0.01),clock=lambda:4.0)
    assert result["terminal"]=="invalid"
