#!/usr/bin/env python3
# BQGATE: fixed contiguous gerund depth bands,36forwards576seq; no fitting.
"""A instrument; B band7_pre17 gate transfer; C cumulative onset;
D band7_pre17 norm moment; E G collateral. Fixed preregistration, no rescue.
"""
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
from bilinear_scalar_consumer_v1 import controls
OUT=POLY/'GERUND_DEPTH_INTERVAL_V1_RESULT.json';BIND=POLY/'GERUND_DEPTH_INTERVAL_V1_BINDING.json'
SITES=[(l,k) for l in range(18) for k in range(2)]
UPSTREAM=[(l,k) for l in range(17) for k in range(2)]+[(17,0)]
ARMS={
    'prefix7':[(l,k) for l in range(8) for k in range(2)],
    'prefix11':[(l,k) for l in range(12) for k in range(2)],
    'prefix13':[(l,k) for l in range(14) for k in range(2)],
    'prefix16':[(l,k) for l in range(17) for k in range(2)],
    'pre_mlp17':UPSTREAM,
    'band7_pre17':[(l,k) for l in range(7,17) for k in range(2)]+[(17,0)],
    'band8_pre17':[(l,k) for l in range(8,17) for k in range(2)]+[(17,0)],
    'band11_pre17':[(l,k) for l in range(11,17) for k in range(2)]+[(17,0)],
    'band14_pre17':[(l,k) for l in range(14,17) for k in range(2)]+[(17,0)],
    'attention17':[(17,0)],
}
def serial(x):return x.detach().cpu().tolist()
def metric(change,reference):
    return dict(transfer=float((change*reference).sum()/reference.square().sum()),relative_error=float((change-reference).norm()/reference.norm()),change_per_row=serial(change),reference_per_row=serial(reference))

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items());tiny=controls()
    panels=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=36,sequences=576,module_hook_visits=1296,arm_sizes={k:len(v) for k,v in ARMS.items()},controls=tiny)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    engine=ScalarWriteNetwork(backend,e,36,576);capture_visits=0;checks={};reports={};saved={}
    m=model.transformer.h[17].mlp;L,R,D=m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double();le,re=L@e,R@e
    def body(panel,label,sites=(),cache=None,cycle=False):
        nonlocal capture_visits
        ix=torch.arange(16,device='cuda');pos=torch.tensor([v['base_semantic_position'] for v in panel],device='cuda');capture={};hooks=[]
        def callback(l,k):
            def hook(_m,args,out):
                nonlocal capture_visits
                capture_visits+=1;y=out[0] if k==0 else out
                if (l,k) in sites:
                    target=cache[l,k].roll(-1,0) if cycle else cache[l,k]
                    y=y.clone();y[ix,pos]=target.to(y)
                capture[l,k]=y[ix,pos].detach().clone()
                if (l,k)==(17,1):capture['u17']=args[0][ix,pos].detach().clone()
                return (y,out[1]) if k==0 else y
            return hook
        for l,block in enumerate(model.transformer.h):hooks.extend([block.attn.register_forward_hook(callback(l,0)),block.mlp.register_forward_hook(callback(l,1))])
        try:r=engine.body(panel,'base',label=label)
        finally:
            for h in hooks:h.remove()
        r['cache']={s:capture[s] for s in SITES};r['u17']=capture['u17'].double();return r
    try:
        for name in ['A1','A2','G']:
            panel=panels[name];assert len(panel)==16
            ids=torch.tensor([[v['base_answer_id'],v['base_foil_id']] for v in panel],device='cuda')
            V=model.lm_head.weight[ids[:,0]].double()-model.lm_head.weight[ids[:,1]].double()
            c=V@D;k=(c*re)@L+(c*le)@R;a=(c*(le*re)).sum(-1);kp=k-2*a[:,None]*e
            base=body(panel,name+'_base');cache=base['cache'];tau=(kp*base['u17']).sum(-1);moment=base['h'].double().square().mean(-1)
            gate_ref=tau.roll(-1,0)-tau;moment_ref=moment.roll(-1,0)-moment
            checks[name+'_gate_reference_norm']=float(gate_ref.norm());checks[name+'_moment_reference_norm']=float(moment_ref.norm())
            checks[name+'_final_tokens_match']=all(panel[i]['base_ids'][-1]==panel[(i+1)%16]['base_ids'][-1] for i in range(16))
            noop=body(panel,name+'_noop',UPSTREAM,cache);checks[name+'_noop']=bridge(noop['z'],base['z'])
            arm_reports={};arm_states={}
            baseline_ce=F.cross_entropy(base['z'],ids[:,0],reduction='none')
            for label,sites in ARMS.items():
                live=body(panel,name+'_'+label,sites,cache,True);live_tau=(kp*live['u17']).sum(-1);live_moment=live['h'].double().square().mean(-1);ce=F.cross_entropy(live['z'],ids[:,0],reduction='none')-baseline_ce
                arm_reports[label]=dict(gate=metric(live_tau-tau,gate_ref),norm_moment=metric(live_moment-moment,moment_ref),mean_ce=float(ce.mean()),mean_absolute_ce=float(ce.abs().mean()),ce_change_per_row=serial(ce),site_count=len(sites))
                arm_states[label]=dict(u17=live['u17'].cpu(),h=live['h'].cpu())
                if label=='pre_mlp17':
                    checks[name+'_donor_u17']=bridge(live['u17'],base['u17'].roll(-1,0));checks[name+'_donor_h']=bridge(live['h'],base['h'].roll(-1,0))
            margin=base['af'][:,0]-base['af'][:,1]
            reports[name]=dict(base_capability_count=int((margin>0).sum()),base_margins=serial(margin),arms=arm_reports)
            saved[name]=dict(row_ids=[v['row_id'] for v in panel],context_reader=kp.cpu(),base_u17=base['u17'].cpu(),base_h=base['h'].cpu(),arms=arm_states)
    finally:engine.close()
    valid=engine.valid() and capture_visits==1296 and all(r['base_capability_count']==16 for r in reports.values())
    for name in reports:
        valid=valid and checks[name+'_final_tokens_match'] and checks[name+'_gate_reference_norm']>1e-4 and checks[name+'_moment_reference_norm']>1e-4
        valid=valid and checks[name+'_noop']['max_abs']<=1e-3 and checks[name+'_noop']['relative_l2']<=1e-5
        valid=valid and checks[name+'_donor_u17']['relative_l2']<=1e-5 and checks[name+'_donor_h']['relative_l2']<=1e-5
    def both(label,field,tmin=None,emax=None):
        vals=[reports[n]['arms'][label][field] for n in ['A1','A2']]
        return all((tmin is None or v['transfer']>=tmin) and (emax is None or v['relative_error']<=emax) for v in vals)
    onset=all(reports[n]['arms']['prefix7']['gate']['transfer']<=.5 and reports[n]['arms']['prefix11']['gate']['transfer']>=.5 and reports[n]['arms']['prefix13']['gate']['transfer']>=.65 for n in ['A1','A2'])
    artifact=POLY/'GERUND_DEPTH_INTERVAL_V1_STATES.pt';assert not artifact.exists();torch.save(saved,artifact)
    predictions=dict(pred_a_instrument=bool(valid),pred_b_distributed_target_band=bool(valid and both('band7_pre17','gate',.75,.5)),pred_c_cumulative_onset=bool(valid and onset),pred_d_norm_state_sufficiency=bool(valid and both('band7_pre17','norm_moment',.5,.75)),pred_e_collateral_control=bool(valid and reports['G']['arms']['band7_pre17']['mean_absolute_ce']<=.15))
    result=dict(schema='gerund.depth_interval.v1',predictions=predictions,reports=reports,checks=checks,engine_bridges=engine.bridges,controls=tiny,arm_sites={k:[list(s) for s in v] for k,v in ARMS.items()},artifact_sha256=digest(artifact),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],module_hook_visits=capture_visits,pre_mlp17_sites=len(UPSTREAM),band7_pre17_sites=len(ARMS['band7_pre17']),conditional_port_reduction=1-len(ARMS['band7_pre17'])/len(UPSTREAM),port_width=1152,native_parameters=sum(p.numel() for p in model.parameters()),artifact_bytes=artifact.stat().st_size,native_weight_saving=0),scope='Fixed contiguous-depth causal interface for the opened gerund token-context gate and final norm moment. Native weights, donor state generation, all token positions, MLP17 and suffix remain external; no semantic, natural-text OOD or static compression claim.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=predictions,target={n:{a:{'gate':reports[n]['arms'][a]['gate'],'norm':reports[n]['arms'][a]['norm_moment']} for a in ['prefix7','prefix11','prefix13','band7_pre17','band11_pre17','pre_mlp17']} for n in ['A1','A2']},g_control=reports['G']['arms']['band7_pre17']['mean_absolute_ce'],wall_seconds=result['wall_seconds'])))
if __name__=='__main__':main()
