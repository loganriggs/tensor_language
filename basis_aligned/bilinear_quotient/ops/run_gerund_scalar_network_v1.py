#!/usr/bin/env python3
# BQGATE: fixed gerund e through all36 output sites,48forwards768seq; no selection.
"""pred_a numeric/port identities; pred_b native capability; pred_c all edit
prediction errors<=.10; pred_d all-port cue recovery>=.80 with controls;
pred_e target zero-removal CE>=.10/C absCE<=.10. Native background retained.
Full fixed protocol GERUND_SCALAR_NETWORK_V1_PREREGISTRATION.md.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
import gerund_scalar_network_v1 as S
from induction_context_transport_v2 import digest
from circuit_endpoint_capability_v1 import summarize
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'GERUND_SCALAR_NETWORK_V1_RESULT.json';BIND=POLY/'GERUND_SCALAR_NETWORK_V1_BINDING.json'
GROUPS={'attention':(0,),'mlp':(1,),'all':(0,1)}
def serial(x):return x.detach().cpu().tolist()
def center(x):return x-x.mean(-1,keepdim=True)
def bridge(a,b):
    a=a.double();b=b.double();err=a-b
    return {'max_abs':float(err.abs().max()),'relative_l2':float(err.norm()/b.norm().clamp_min(1e-30))}

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items());controls=S.controls()
    rows=json.loads((POLY/'GERUND_SHARED_READER_V1_ROWS.json').read_text())['panels'];parent=json.loads((POLY/'GERUND_SHARED_READER_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'gpu_accessed':False,'model_loaded':False,'body_forwards':48,'sequences':768,'controls':controls}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;U=model.lm_head.weight.float()
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    lam=torch.stack([b.lambdas.detach() for b in model.transformer.h]);beta,gamma=S.coefficients(lam)
    counts=[0,0];bridges={};reports={};all_port_checks=[];finite=True;hook_visits=0
    def count(_m,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=48 and counts[1]<=768
    handle=model.transformer.h[0].attn.register_forward_pre_hook(count)
    def read(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(1152,)),U)/30)
    def body(panel,side,group=None,mode=None,donor=None,label=''):
        nonlocal finite,hook_visits
        batch=g.batch_of(panel,side);ix=torch.arange(16,device='cuda');pos=torch.tensor(batch.semantic_positions,device='cuda')
        cap={};raw={};hooks=[]
        def callback(l,k):
            def hook(_m,args,out):
                nonlocal hook_visits
                hook_visits+=1;y=out[0] if k==0 else out;value=y[ix,pos]
                if group is not None and k in GROUPS[group]:
                    target=donor[l,k] if mode=='donor' else torch.zeros(16,device='cuda',dtype=torch.float64)
                    updated=value.double()+(target-value.double()@e)[:,None]*e
                    changed=y.clone();changed[ix,pos]=updated.to(y);y=changed;value=y[ix,pos]
                cap[l,k]=value.detach().clone()
                return (y,out[1]) if k==0 else y
            return hook
        for l,block in enumerate(model.transformer.h):
            hooks.extend([block.attn.register_forward_hook(callback(l,0)),block.mlp.register_forward_hook(callback(l,1))])
        try:af,z=g.forward_units(backend,batch,return_logits=True,capture_resid=raw)
        finally:
            for h in hooks:h.remove()
        writes=torch.stack([torch.stack([cap[l,0],cap[l,1]]) for l in range(18)])
        h=torch.stack([raw[(rid,17)] for rid in batch.row_ids])+cap[17,1]
        last_ids=torch.tensor([t[p] for t,p in zip(batch.token_rows,batch.semantic_positions)],device='cuda')
        x0=F.rms_norm(model.transformer.wte(last_ids),(1152,))
        reconstructed=gamma*x0.double()+(beta[:,None,None]*writes.double().sum(1)).sum(0)
        bridges[label+'_state']=bridge(reconstructed,h);bridges[label+'_readout']=bridge(read(reconstructed),z)
        sc=writes.double()@e
        if group=='all':
            expected=gamma*(x0.double()@e)
            if mode=='donor':expected=expected+(beta[:,None]*donor.sum(1)).sum(0)
            err=(h.double()@e-expected).abs();tol=torch.maximum(1e-3+1e-5*expected.abs(),1e-6*h.double().norm(dim=1))
            all_port_checks.append({'arm':label,'max_abs':float(err.max()),'max_scaled':float((err/tol).max())})
        finite=finite and all(bool(torch.isfinite(v).all()) for v in [writes,h,z,af])
        return {'af':af,'z':z,'h':h,'x0':x0,'scalars':sc}
    def arm_metrics(z,baseline,panel,side,reference_margin,den,pred):
        answer=torch.tensor([r[side+'_answer_id'] for r in panel],device='cuda');foil=torch.tensor([r[side+'_foil_id'] for r in panel],device='cuda');ix=torch.arange(16,device='cuda')
        ce=F.cross_entropy(z,answer,reduction='none')-F.cross_entropy(baseline,answer,reduction='none')
        recovery=(reference_margin-(z[ix,answer]-z[ix,foil]))/den
        effect=center(z.double()-baseline.double());error=center(pred.double()-z.double());en=effect.square().sum(-1);er=error.square().sum(-1)
        return {'raw_recovery':float(recovery.mean()),'raw_recovery_per_row':serial(recovery),'ce_change_per_row':serial(ce),
                'mean_ce_change':float(ce.mean()),'mean_absolute_ce_change':float(ce.abs().mean()),
                'live_effect_norm':float(en.sum().sqrt()),'prediction_error_relative':float(er.sum().sqrt()/en.sum().sqrt().clamp_min(1e-30)),
                'prediction_error_squared_per_row':serial(er),'live_effect_squared_per_row':serial(en)}
    try:
        with torch.inference_mode():
            for name,panel in rows.items():
                b=body(panel,'base',label=name+'_native_base');d=body(panel,'donor',label=name+'_native_donor')
                noop=body(panel,'base','all','donor',b['scalars'],name+'_noop');bridges[name+'_noop']=bridge(noop['z'],b['z'])
                bm=b['af'][:,0]-b['af'][:,1];dm=d['af'][:,0]-d['af'][:,1];den=bm+dm
                parentz=read(b['h'].double()+((d['h'].double()-b['h'].double())@e)[:,None]*e)
                pm=arm_metrics(parentz,b['z'],panel,'base',bm,den,parentz)
                old=parent['reports'][name];bridges[name+'_parent_recovery']=bridge(torch.tensor(pm['raw_recovery_per_row']),torch.tensor(old['arms']['scalar_final']['raw_recovery_per_row']))
                for side,capture in [('base',b),('donor',d)]:
                    z=read(capture['h'].double()-(capture['h'].double()@e)[:,None]*e)
                    target=torch.tensor([r[side+'_answer_id'] for r in panel],device='cuda')
                    ce=F.cross_entropy(z,target,reduction='none')-F.cross_entropy(capture['z'],target,reduction='none')
                    bridges[name+'_parent_zero_'+side]=bridge(ce,torch.tensor(old['zero_removal'][side+'_ce_change_per_row'],device='cuda'))
                arms={}
                for group,kinds in GROUPS.items():
                    delta=(beta[:,None,None]*(d['scalars'][:,kinds]-b['scalars'][:,kinds])).sum((0,1))
                    pred=read(b['h'].double()+delta[:,None]*e)
                    live=body(panel,'base',group,'donor',d['scalars'],name+'_'+group+'_swap')
                    arms[group+'_swap']=arm_metrics(live['z'],b['z'],panel,'base',bm,den,pred)
                    arms[group+'_swap']['final_scalar_delta_per_row']=serial((live['h'].double()-b['h'].double())@e)
                    arms[group+'_swap']['direct_scalar_delta_per_row']=serial(delta)
                    for side,capture,margin in [('base',b,bm),('donor',d,dm)]:
                        delta=-(beta[:,None,None]*capture['scalars'][:,kinds]).sum((0,1));pred=read(capture['h'].double()+delta[:,None]*e)
                        live=body(panel,side,group,'zero',label=name+'_'+group+'_zero_'+side)
                        arms[group+'_zero_'+side]=arm_metrics(live['z'],capture['z'],panel,side,margin,den,pred)
                reports[name]={'capability':summarize(serial(bm),serial(dm)),'positive_denominators':bool((den>1e-6).all()),
                    'paired_x0_equal':bool(torch.equal(b['x0'],d['x0'])),'arms':arms,
                    'native_weighted_scalar_delta':serial(beta[:,None,None]*(d['scalars']-b['scalars'])),
                    'native_final_scalar_delta':serial((d['h'].double()-b['h'].double())@e)}
    finally:handle.remove()
    instrument=counts==[48,768] and hook_visits==48*36 and finite and abs(float(e.norm())-1)<=1e-5 and all(
        v['relative_l2']<=1e-5 if k.endswith('_state') else v['max_abs']<=1e-3 if '_parent_' in k else v['max_abs']<=1e-3 and v['relative_l2']<=1e-5 for k,v in bridges.items()) and all(v['max_scaled']<=1 for v in all_port_checks)
    capability=all(v['capability']['both_endpoints_correct']==16 and v['paired_x0_equal'] for v in reports.values()) and all(reports[k]['positive_denominators'] for k in ['A1','A2','C'])
    closure=instrument and capability and all(v['prediction_error_relative']<=.1 and v['live_effect_norm']>1e-4 for k in ['A1','A2'] for v in reports[k]['arms'].values())
    suff=instrument and capability and all(reports[k]['arms']['all_swap']['raw_recovery']>=.8 for k in ['A1','A2']) and all(reports[k]['arms']['all_swap']['mean_absolute_ce_change']<=.1 for k in ['P','C']) and abs(reports['C']['arms']['all_swap']['raw_recovery'])<=.1
    removal=instrument and capability and all(sum(reports[k]['arms']['all_zero_'+s]['mean_ce_change'] for s in ['base','donor'])/2>=.1 for k in ['A1','A2']) and sum(reports['C']['arms']['all_zero_'+s]['mean_absolute_ce_change'] for s in ['base','donor'])/2<=.1
    result={'schema':'gerund.scalar_network.v1','predictions':{'pred_a_instrument':instrument,'pred_b_capability':capability,'pred_c_edit_prediction_closure':closure,'pred_d_scalar_network_sufficiency':suff,'pred_e_selective_network_removal':removal},'reports':reports,'bridges':bridges,'all_port_scalar_checks':all_port_checks,'controls':controls,
            'lambdas':serial(lam),'propagation_coefficients':serial(beta),'embedding_coefficient':float(gamma),'price':{'body_forwards':counts[0],'sequences':counts[1],'hook_visits':hook_visits,'native_parameters':sum(v.numel() for v in model.parameters()),'native_weight_saving':0},
            'runner_sha256':digest(RUNNER),'binding_sha256':digest(BIND),'wall_seconds':time.perf_counter()-tic,
            'scope':'Global fixed-e output-port interventions with native complement/routing/norms live. Scalar accumulation is an identity, not independently extracted producer. Frozen-background endpoint prediction tested separately.'}
    atomic_create_json(OUT,result);print(json.dumps({'predictions':result['predictions'],'summary':{k:{'all_swap_recovery':v['arms']['all_swap']['raw_recovery'],'all_swap_prediction_error':v['arms']['all_swap']['prediction_error_relative'],'all_zero_base_prediction_error':v['arms']['all_zero_base']['prediction_error_relative']} for k,v in reports.items()},'wall_seconds':result['wall_seconds']}))

if __name__=='__main__':main()
