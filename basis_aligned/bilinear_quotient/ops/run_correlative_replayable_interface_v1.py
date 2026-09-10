#!/usr/bin/env python3
# BQGATE: five frozen interface checks; unchanged v505 recipe, new saved fit identity.
"""pred_a original5 hold; pred_b26units and4 metricdifferences<=.01;
pred_c saved tensors identical/q orthogonality<=1e-5; pred_d replay<=1e-5;
pred_e native/head/MLP bridge<=1e-4. No semantic promotion from these checks.
Caps200020bodyforwards6400320sequences; no native parameter updates.
"""
import json,os,sys,time
from pathlib import Path
import torch
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import run_unit_correlative_p_interchange_v505 as V
import circuit_unit_greedy as g
import circuit_fast_screen_producer as producer
import circuit_fast_screen_candidate_correlative_pair as canonical
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json

OUT=POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_RESULT.json'
RECIPE_OUT=POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_RECIPE_RESULT.json'
ARTIFACT=POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt'
ROWS=POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ROWS.json'
BINDING=POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_BINDING.json'
PRIOR=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/unit_correlative_p_interchange_v505_result.json'


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    authored=g.rows_of(canonical,'A1');expected=authored[2::4]+authored[3::4]
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'fit_rows':len(authored[0::4]+authored[1::4]),
                          'held_rows':len(expected),'body_forwards_max':200020,'sequences_max':6400320,
                          'new_fit_identity':True,'rank_per_block':1}));return
    assert all(not p.exists() for p in (OUT,RECIPE_OUT,ARTIFACT,ROWS))
    started=time.perf_counter();capture={};rowsets=[];counts=[0,0]
    original_load=producer.Bilin18TorchBackend.load
    original_prepare=g.prepare;original_fit=g.fit_block_subspace_constrained;original_greedy=g.greedy_heads
    def load(cls,device='cuda'):
        backend=original_load(device);capture['backend']=backend
        def count(_m,args):
            counts[0]+=1;counts[1]+=args[0].shape[0]
            if counts[0]>200020 or counts[1]>6400320:raise RuntimeError('Frozen forward cap exceeded')
        capture['counter']=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
        return backend
    def prepare(backend,rows,**kwargs):
        prep=original_prepare(backend,rows,**kwargs)
        rowsets.append({'call':len(rowsets),'kwargs':kwargs,'input_rows':rows,'kept_rows':prep.rows,'dropped':prep.dropped})
        if [r['row_id'] for r in rows]==[r['row_id'] for r in expected]:capture['held']=prep
        return prep
    def greedy(*args,**kwargs):
        result=original_greedy(*args,**kwargs);capture['greedy']=result[2];capture['greedy_kwargs']=kwargs
        return result
    def fit(backend,prep,units,**kwargs):
        q,history=original_fit(backend,prep,units,**kwargs)
        saved={'schema':'correlative.reconstructed_interface.v1','units':list(units),
               'q':{key:value.detach().cpu().clone() for key,value in q.items()},
               'mu':{key:torch.as_tensor(value).detach().cpu().clone() for key,value in kwargs['mu'].items()},
               'history':history,'fit_rows':prep.rows,
               'control_rows':[cp.rows for cp in kwargs['controls']],
               'fit_parameters':{key:value for key,value in kwargs.items() if key not in ('controls','mu')},
               'greedy':capture['greedy'],'greedy_kwargs':capture['greedy_kwargs'],
               'binding_sha256':digest(BINDING),'historical_identity_unavailable':True}
        with ARTIFACT.open('xb') as stream:torch.save(saved,stream)
        capture.update(q=q,saved=saved,units=list(units))
        return q,history
    producer.Bilin18TorchBackend.load=classmethod(load)
    g.prepare=prepare;g.greedy_heads=greedy;g.fit_block_subspace_constrained=fit
    V.OUT=RECIPE_OUT
    try:
        V.main()
    finally:
        producer.Bilin18TorchBackend.load=classmethod(lambda cls,device='cuda':original_load(device))
        g.prepare=original_prepare;g.greedy_heads=original_greedy;g.fit_block_subspace_constrained=original_fit
    atomic_create_json(ROWS,{'rowsets':rowsets,'source_binding_sha256':digest(BINDING)})
    recipe=json.loads(RECIPE_OUT.read_text());prior=json.loads(PRIOR.read_text())
    loaded=torch.load(ARTIFACT,map_location='cpu',weights_only=True)
    identical=all(torch.equal(loaded[name][key],val) for name in ('q','mu') for key,val in capture['saved'][name].items())
    orthogonal=max(float((q.T@q-torch.eye(q.shape[1])).abs().max()) for q in loaded['q'].values())
    ranks={str(key):q.shape[1] for key,q in loaded['q'].items()}
    backend=capture['backend'];units=capture['units'];held=capture['held']
    q_loaded={key:value.to(backend.device) for key,value in loaded['q'].items()}
    memory=g.patched_axis(backend,held,units,q=capture['q'])
    disk=g.patched_axis(backend,held,units,q=q_loaded)
    replay=max(abs(a-b) for a,b in zip(memory,disk))
    fit_rows=authored[0::4]+authored[1::4]
    bridge=g.verify_against_producer(backend,fit_rows[:4],layer=0,heads=(0,),mlp_layer=0,tolerance=1e-4)
    capture['counter'].remove()
    current=recipe['groups']['fit_canonical'];old=prior['groups']['fit_canonical']
    differences={'A1':abs(current['per_shape']['correlative_pair']['joint_extraction']-old['per_shape']['correlative_pair']['joint_extraction'])}
    for name in ('A2_extraction','P_same_answer_effect','C_same_answer_effect'):
        differences[name]=abs(current['hypotheses'][name]-old['hypotheses'][name])
    predictions={
        'pred_a_original_gates':all(recipe['predictions'].values()),
        'pred_b_repeat_metrics':len(units)==26 and all(v<=.01 for v in differences.values()),
        'pred_c_artifact_identity':identical and orthogonal<=1e-5 and all(r==1 for r in ranks.values()),
        'pred_d_disk_replay':replay<=1e-5,
        'pred_e_native_bridge':bridge['passed']}
    result={'schema':'correlative.replayable_interface.v1','predictions':predictions,
            'units':units,'blocks':ranks,'total_rank':sum(ranks.values()),'metric_differences':differences,
            'orthogonality_maxabs':orthogonal,'disk_replay_margin_maxabs':replay,'native_bridge':bridge,
            'new_recipe_result':str(RECIPE_OUT),'artifact':str(ARTIFACT),'artifact_sha256':digest(ARTIFACT),
            'rows_sha256':digest(ROWS),'binding_sha256':digest(BINDING),'runner_sha256':digest(RUNNER),
            'price':{'body_forwards':counts[0],'sequences':counts[1],
                     'q_scalars':sum(q.numel() for q in loaded['q'].values()),
                     'mean_scalars':sum(v.numel() for v in loaded['mu'].values()),
                     'native_parameters':sum(p.numel() for p in backend.model.parameters()),'native_weight_saving':0},
            'wall_seconds':time.perf_counter()-started,
            'scope':'New fitted conditional interface; metrics cannot establish historical tensor identity; no fresh OOD or standalone extraction'}
    atomic_create_json(OUT,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
