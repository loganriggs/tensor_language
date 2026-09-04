from __future__ import annotations

import pytest

import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as reader
import run_task14_head11_3_late_mlp_halves_factorial as run


class FakeBackend:
    def __init__(self,*,nonlinear=False,control_interaction=0.0,replay_shift=0.0):
        rows,self.native_pairs,_head,self.prior=run._load()
        self.family={str(x["row_id"]):str(x["transform_id"]) for x in rows}
        self.nonlinear,self.control_interaction,self.replay_shift=nonlinear,control_interaction,replay_shift
        self.scale=__import__("statistics").median(reader._margin(self.native_pairs[(str(x["row_id"]),"donor")])
          +reader._margin(self.native_pairs[(str(x["row_id"]),"base")]) for x in rows
          if x["transform_id"] in {"A1","A2"})
    def native(self,batch:producer.ModelBatch,*,capture:bool):
        pairs=[]; cache={}
        for rid in batch.row_ids:
            pair=self.native_pairs[(rid,batch.side)]; pairs.append((pair[0]+self.replay_shift,pair[1]))
            if capture:
                cache[(rid,reader.HEAD_SITE)]=object()
                for site in run.FIRST+run.SECOND: cache[(rid,site)]=object()
        return producer.BatchOutput(tuple(pairs),cache)
    def induce_and_restore(self,batch,*,restore_sites,donor_cache,recipient_cache):
        arm="mlp13_14" if tuple(restore_sites)==run.FIRST else "mlp15_17"; pairs=[]
        for rid in batch.row_ids:
            p=self.prior[rid]; family=self.family[rid]
            interaction=(-0.2 if self.nonlinear and family in {"A1","A2"} else
                         self.control_interaction if family not in {"A1","A2"} else 0.0)
            if self.nonlinear or interaction:
                recovery=p["empty"] if arm=="mlp13_14" else p["all_late"]-interaction
            else:
                recovery=p["all_late"] if arm=="mlp13_14" else p["empty"]
            base=self.native_pairs[(rid,"base")]; donor=self.native_pairs[(rid,"donor")]; bm=reader._margin(base)
            margin=bm-recovery*(reader._margin(donor)+bm) if family in {"A1","A2"} else bm+recovery*self.scale
            pairs.append((margin,0.0))
        return producer.BatchOutput(tuple(pairs),{})


def test_dryrun_opens_only_two_halves():
    d=run.compile_dryrun()
    assert d["maximum_new_price"]=={"forward_calls":16,"example_evaluations":512,
      "backward_calls":0,"model_updates":0,"raw_numeric_evidence_bytes":2048}
    assert d["groups"]=={"mlp13_14":["mlp:13","mlp:14"],
                         "mlp15_17":["mlp:15","mlp:16","mlp:17"]}


def test_first_half_exactly_explains_all_late():
    result=run.run_science(backend=FakeBackend(),clock=lambda:1.0)
    assert result["terminal"]=="one_half_explains_screen"
    assert result["dominant_half"]==["mlp13_14"]
    assert result["half_diagnostics"]["mlp13_14"]["relative_l2_to_all_late"]==pytest.approx(0)


def test_planted_cross_half_interaction():
    result=run.run_science(backend=FakeBackend(nonlinear=True),clock=lambda:2.0)
    assert result["terminal"]=="cross_half_interaction_screen"
    assert result["predictions"]["pred_c_cross_half_interaction"] is True
    assert all(x["interaction"]==pytest.approx(-0.2) for x in result["target_cells"].values())


def test_control_failure_blocks_scientific_terminal():
    result=run.run_science(backend=FakeBackend(control_interaction=0.3),clock=lambda:3.0)
    assert result["terminal"]=="inconclusive"
    assert set(result["control_mean_absolute_terms"]["P"])=={
      "empty","all_late","mlp13_14","mlp15_17","all_late_loss",
      "mlp13_14_loss","mlp15_17_loss","interaction"}


def test_native_replay_failure_invalidates():
    assert run.run_science(backend=FakeBackend(replay_shift=0.01),clock=lambda:4.0)["terminal"]=="invalid"
