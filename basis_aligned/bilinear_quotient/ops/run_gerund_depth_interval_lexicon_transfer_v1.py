#!/usr/bin/env python3
# BQGATE: fixed R-lexicon depth bands,8forwards128seq; no fitting.
"""Held-out operator transfer with five frozen predicates and no band search."""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
from scalar_write_network_executor_v1 import ScalarWriteNetwork,bridge
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'GERUND_DEPTH_INTERVAL_LEXICON_TRANSFER_V1_RESULT.json';BIND=POLY/'GERUND_DEPTH_INTERVAL_LEXICON_TRANSFER_V1_BINDING.json'
SITES=[(l,k) for l in range(18) for k in range(2)];UPSTREAM=[(l,k) for l in range(17) for k in range(2)]+[(17,0)]
ARMS={'prefix7':[(l,k) for l in range(8) for k in range(2)],'prefix11':[(l,k) for l in range(12) for k in range(2)],'prefix13':[(l,k) for l in range(14) for k in range(2)],'pre_mlp17':UPSTREAM,'band7_pre17':[(l,k) for l in range(7,17) for k in range(2)]+[(17,0)],'band11_pre17':[(l,k) for l in range(11,17) for k in range(2)]+[(17,0)]}
def serial(x):return x.detach().cpu().tolist()
def metric(change,reference):return {'transfer':float((change*reference).sum()/reference.square().sum()),'relative_error':float((change-reference).norm()/reference.norm()),'change_per_row':serial(change),'reference_per_row':serial(reference)}
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    panel=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']['R'];assert len(panel)==16
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':8,'sequences':128,'module_hook_visits':288,'registered_predictions':5}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda();engine=ScalarWriteNetwork(backend,e,8,128);visits=0
    m=model.transformer.h[17].mlp;L,R,D=m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double();le,re=L@e,R@e
    def body(label,sites=(),cache=None,cycle=False):
        nonlocal visits
        ix=torch.arange(16,device='cuda');pos=torch.tensor([v['base_semantic_position'] for v in panel],device='cuda');capture={};hooks=[]
        def callback(l,k):
            def hook(_m,args,out):
                nonlocal visits
                visits+=1;y=out[0] if k==0 else out
                if (l,k) in sites:
                    target=cache[l,k].roll(-1,0) if cycle else cache[l,k];y=y.clone();y[ix,pos]=target.to(y)
                capture[l,k]=y[ix,pos].detach().clone()
                if (l,k)==(17,1):capture['u17']=args[0][ix,pos].detach().clone()
                return (y,out[1]) if k==0 else y
            return hook
        for l,b in enumerate(model.transformer.h):hooks.extend([b.attn.register_forward_hook(callback(l,0)),b.mlp.register_forward_hook(callback(l,1))])
        try:r=engine.body(panel,'base',label=label)
        finally:
            for h in hooks:h.remove()
        r['cache']={s:capture[s] for s in SITES};r['u17']=capture['u17'].double();return r
    try:
        ids=torch.tensor([[v['base_answer_id'],v['base_foil_id']] for v in panel],device='cuda');V=model.lm_head.weight[ids[:,0]].double()-model.lm_head.weight[ids[:,1]].double();c=V@D;k=(c*re)@L+(c*le)@R;a=(c*(le*re)).sum(-1);kp=k-2*a[:,None]*e
        base=body('R_base');cache=base['cache'];tau=(kp*base['u17']).sum(-1);moment=base['h'].double().square().mean(-1);gate_ref=(kp*base['u17'].roll(-1,0)).sum(-1)-tau;moment_ref=moment.roll(-1,0)-moment
        checks={'gate_reference_norm':float(gate_ref.norm()),'moment_reference_norm':float(moment_ref.norm()),'final_tokens_match':all(panel[i]['base_ids'][-1]==panel[(i+1)%16]['base_ids'][-1] for i in range(16))}
        noop=body('R_noop',UPSTREAM,cache);checks['noop']=bridge(noop['z'],base['z']);reports={};saved={}
        baseline_ce=F.cross_entropy(base['z'],ids[:,0],reduction='none')
        for label,sites in ARMS.items():
            live=body('R_'+label,sites,cache,True);change=(kp*live['u17']).sum(-1)-tau;live_moment=live['h'].double().square().mean(-1);ce=F.cross_entropy(live['z'],ids[:,0],reduction='none')-baseline_ce
            reports[label]={'gate':metric(change,gate_ref),'norm_moment':metric(live_moment-moment,moment_ref),'mean_ce':float(ce.mean()),'mean_absolute_ce':float(ce.abs().mean()),'ce_change_per_row':serial(ce),'site_count':len(sites)};saved[label]={'u17':live['u17'].cpu(),'h':live['h'].cpu()}
            if label=='pre_mlp17':checks['donor_u17']=bridge(live['u17'],base['u17'].roll(-1,0));checks['donor_h']=bridge(live['h'],base['h'].roll(-1,0))
        capability=int(((base['af'][:,0]-base['af'][:,1])>0).sum())
    finally:engine.close()
    valid=engine.valid() and visits==288 and capability==16 and checks['final_tokens_match'] and checks['gate_reference_norm']>1e-4 and checks['moment_reference_norm']>1e-4 and checks['noop']['max_abs']<=1e-3 and checks['noop']['relative_l2']<=1e-5 and checks['donor_u17']['max_abs']<=1e-3 and checks['donor_u17']['relative_l2']<=1e-5 and checks['donor_h']['max_abs']<=1e-3 and checks['donor_h']['relative_l2']<=1e-5 and abs(reports['pre_mlp17']['gate']['transfer']-1)<=1e-12 and reports['pre_mlp17']['gate']['relative_error']<=1e-12
    g7=reports['band7_pre17']['gate'];n7=reports['band7_pre17']['norm_moment'];g11=reports['band11_pre17']['gate'];n11=reports['band11_pre17']['norm_moment']
    predictions={'pred_a_instrument':bool(valid),'pred_b_heldout_band7_gate':bool(valid and g7['transfer']>=.75 and g7['relative_error']<=.5),'pred_c_heldout_onset':bool(valid and reports['prefix7']['gate']['transfer']<=.5 and reports['prefix11']['gate']['transfer']>=.5 and reports['prefix13']['gate']['transfer']>=.65),'pred_d_heldout_band7_norm':bool(valid and n7['transfer']>=.5 and n7['relative_error']<=.75),'pred_e_smaller_band11_interface':bool(valid and g11['transfer']>=.75 and g11['relative_error']<=.5 and n11['transfer']>=.5 and n11['relative_error']<=.75)}
    artifact=POLY/'GERUND_DEPTH_INTERVAL_LEXICON_TRANSFER_V1_STATES.pt';assert not artifact.exists();torch.save({'row_ids':[v['row_id'] for v in panel],'context_reader':kp.cpu(),'base_u17':base['u17'].cpu(),'base_h':base['h'].cpu(),'arms':saved},artifact)
    result={'schema':'gerund.depth_interval_lexicon_transfer.v1','predictions':predictions,'base_capability_count':capability,'reports':reports,'checks':checks,'engine_bridges':engine.bridges,'artifact_sha256':digest(artifact),'runner_sha256':digest(RUNNER),'binding_sha256':digest(BIND),'wall_seconds':time.perf_counter()-tic,'price':{'body_forwards':engine.counts[0],'sequences':engine.counts[1],'module_hook_visits':visits,'full_prefix_sites':35,'band7_sites':21,'band11_sites':13,'band11_conditional_port_reduction':1-13/35,'port_width':1152,'native_parameters':sum(p.numel() for p in model.parameters()),'native_weight_saving':0,'artifact_bytes':artifact.stat().st_size},'scope':'Held-out R-lexicon validation of fixed depth bands. Existing G collateral failure remains; native generators, weights, positions, MLP17 and suffix are external.'}
    atomic_create_json(OUT,result);print(json.dumps({'predictions':predictions,'metrics':{k:{'gate':reports[k]['gate'],'norm':reports[k]['norm_moment']} for k in ARMS},'wall_seconds':result['wall_seconds']}))
if __name__=='__main__':main()
