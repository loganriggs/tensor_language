#!/usr/bin/env python3
# BQGATE: fixed weight-derived -ing contrast, 12 forwards144seq; no activation fit.
"""pred_a instrument; pred_b native capability; pred_c scalar and cross-verb
recovery>=.80 with P/C preservation; pred_d MLP scalar recovery>=.80;
pred_e target zero-removal CE>=.10 and C meanabsCE<=.10. No rank/gain rescue.
See GERUND_SHARED_READER_V1_PREREGISTRATION.md for exact scope and bars.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
import gerund_shared_reader_v1 as S
from circuit_endpoint_capability_v1 import summarize
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'GERUND_SHARED_READER_V1_RESULT.json';BIND=POLY/'GERUND_SHARED_READER_V1_BINDING.json'
ART=POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt'

def serial(x):return x.detach().cpu().tolist()
def center(x):return x-x.mean(-1,keepdim=True)
def bridge(a,b):
    a=a.double();b=b.double();d=a-b
    return {'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/b.norm().clamp_min(1e-30)),
            'scaled_error':float((d.abs()/(1e-3+1e-5*b.abs())).max())}

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    rows=json.loads((POLY/'GERUND_SHARED_READER_V1_ROWS.json').read_text());controls=S.controls()
    assert all(len(v)==16 for v in rows['panels'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'gpu_accessed':False,'model_loaded':False,'body_forwards':12,'sequences':144,'controls':controls}));return
    assert not OUT.exists() and not ART.exists();signal.alarm(900);tic=time.perf_counter()
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;torch.set_num_threads(2)
    M=model.transformer.h[17].mlp;U=model.lm_head.weight.float();counts=[0,0]
    def counter(_m,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=12 and counts[1]<=144
    count_hook=model.transformer.h[0].attn.register_forward_pre_hook(counter)
    def body(panel,side,delta=None,full_mlp=None):
        batch=g.batch_of(panel,side);ix=torch.arange(len(panel),device='cuda');pos=torch.tensor(batch.semantic_positions,device='cuda')
        cap={};raw={}
        def mh(_m,args,out):
            cap['u']=args[0][ix,pos].detach().clone();cap['m']=out[ix,pos].detach().clone()
            if delta is not None:
                changed=out.clone();changed[ix,pos]+=delta.to(out);return changed
        hook=M.register_forward_hook(mh)
        try:
            donor_cache=None if full_mlp is None else {(rid,'mlp:17'):full_mlp[i] for i,rid in enumerate(batch.row_ids)}
            af,z=g.forward_units(backend,batch,units=() if full_mlp is None else ('mlp:17',),donor_cache=donor_cache,return_logits=True,capture_resid=raw)
            cap['r']=torch.stack([raw[(rid,17)] for rid in batch.row_ids]);cap['h']=cap['r']+cap['m']
            return af,z,cap
        finally:hook.remove()
    def read(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(1152,)),U)/30)
    bridges={};reports={};finite=True
    try:
        with torch.inference_mode():
            fi=torch.tensor(rows['fit_ids'],device='cuda');e=(U[fi[:,1]].double()-U[fi[:,0]].double()).mean(0);e=e/e.norm()
            c=e@M.Down.weight.double();q=M.Left.weight.double().T@(c[:,None]*M.Right.weight.double());q=(q+q.T)/2;bias=e@M.Down_bias.double()
            torch.save({'e':e.cpu(),'product_reader':c.cpu(),'Q':q.cpu(),'bias':bias.cpu(),'fit_pairs':rows['fit_pairs'],
                        'scope':'Weight-defined conditional scalar, native upstream and remainder retained.'},ART)
            for name,panel in rows['panels'].items():
                ba,bz,b=body(panel,'base');da,dz,d=body(panel,'donor')
                bm=(b['u'].double()@q*b['u'].double()).sum(1)+bias;dm=(d['u'].double()@q*d['u'].double()).sum(1)+bias
                bh=b['h'].double()@e;dh=d['h'].double()@e;br=b['r'].double()@e;dr=d['r'].double()@e
                delta_m=dm-bm;delta_h=dh-bh;delta_r=dr-br;perm=torch.arange(16,device='cuda').roll(-1)
                deltas={'scalar_mlp':delta_m[:,None]*e,'scalar_input':delta_r[:,None]*e,
                        'scalar_final':delta_h[:,None]*e,'cross_verb':delta_h[perm,None]*e}
                bridges[name+'_native_base']=bridge(read(b['h']),bz);bridges[name+'_native_donor']=bridge(read(d['h']),dz)
                bridges[name+'_fold_base']=bridge(bm,b['m'].double()@e);bridges[name+'_fold_donor']=bridge(dm,d['m'].double()@e)
                bridges[name+'_scalar_addition']=bridge(delta_m+delta_r,delta_h)
                zarms={'full_mlp':read(b['r']+d['m']),**{k:read(b['h'].double()+v) for k,v in deltas.items()}}
                zero_base=read(b['h'].double()-bh[:,None]*e);zero_donor=read(d['h'].double()-dh[:,None]*e)
                if name=='A1':
                    _,online,_=body(panel[:4],'base',full_mlp=d['m'][:4]);bridges['online_full_mlp']=bridge(online,zarms['full_mlp'][:4])
                    for key in ('scalar_mlp','scalar_final','cross_verb'):
                        _,online,_=body(panel[:4],'base',delta=deltas[key][:4]);bridges['online_'+key]=bridge(online,zarms[key][:4])
                native_margin=ba[:,0]-ba[:,1];donor_margin=da[:,0]-da[:,1];den=native_margin+donor_margin
                ix=torch.arange(16,device='cuda');answer=torch.tensor([r['base_answer_id'] for r in panel],device='cuda');foil=torch.tensor([r['base_foil_id'] for r in panel],device='cuda')
                donor_answer=torch.tensor([r['donor_answer_id'] for r in panel],device='cuda')
                bce=F.cross_entropy(bz,answer,reduction='none');dce=F.cross_entropy(dz,donor_answer,reduction='none')
                arms={}
                for key,z in zarms.items():
                    patched=z[ix,answer]-z[ix,foil];recovery=(native_margin-patched)/den
                    ce=F.cross_entropy(z,answer,reduction='none')-bce
                    arms[key]={'raw_recovery':float(recovery.mean()),'raw_recovery_per_row':serial(recovery),
                        'ce_change':float(ce.mean()),'mean_absolute_ce_change':float(ce.abs().mean()),'ce_change_per_row':serial(ce),
                        'centered_logit_effect_norm':float(center(z.double()-bz.double()).norm())}
                zbce=F.cross_entropy(zero_base,answer,reduction='none')-bce;zdce=F.cross_entropy(zero_donor,donor_answer,reduction='none')-dce
                reports[name]={'capability':summarize(serial(native_margin),serial(donor_margin)),'positive_denominators':bool((den>1e-6).all()),
                    'base_margins':serial(native_margin),'donor_margins':serial(donor_margin),'arms':arms,
                    'zero_removal':{'mean_ce_damage':float((zbce.mean()+zdce.mean())/2),'mean_absolute_ce_change':float((zbce.abs().mean()+zdce.abs().mean())/2),
                                    'base_ce_change_per_row':serial(zbce),'donor_ce_change_per_row':serial(zdce)},
                    'scalars':{'mlp_delta':serial(delta_m),'input_delta':serial(delta_r),'final_delta':serial(delta_h),
                               'base_final':serial(bh),'donor_final':serial(dh),'cross_verb_source_indices':serial(perm)}}
                finite=finite and all(bool(torch.isfinite(x).all()) for x in [ba,da,bz,dz,zero_base,zero_donor,*zarms.values()])
    finally:count_hook.remove()
    instrument=counts==[12,144] and finite and abs(float(e.norm())-1)<=1e-5 and all(
        v['scaled_error']<=1 if '_fold_' in k or '_scalar_addition' in k else v['max_abs']<=1e-3 and v['relative_l2']<=1e-5 for k,v in bridges.items())
    capable=all(v['capability']['both_endpoints_correct']==16 for v in reports.values()) and all(reports[k]['positive_denominators'] for k in ('A1','A2','C'))
    reusable=instrument and capable and all(reports[k]['arms'][a]['raw_recovery']>=.8 for k in ('A1','A2') for a in ('scalar_final','cross_verb')) and reports['P']['arms']['scalar_final']['mean_absolute_ce_change']<=.1 and abs(reports['C']['arms']['scalar_final']['raw_recovery'])<=.1 and reports['C']['arms']['scalar_final']['mean_absolute_ce_change']<=.1
    producer=instrument and capable and all(reports[k]['arms']['scalar_mlp']['raw_recovery']>=.8 for k in ('A1','A2'))
    removal=instrument and capable and all(reports[k]['zero_removal']['mean_ce_damage']>=.1 for k in ('A1','A2')) and reports['C']['zero_removal']['mean_absolute_ce_change']<=.1
    result={'schema':'gerund.shared_reader.v1','predictions':{'pred_a_instrument':instrument,'pred_b_native_capability':capable,'pred_c_reusable_state':reusable,'pred_d_mlp_producer':producer,'pred_e_selective_removal':removal},
            'reports':reports,'bridges':bridges,'controls':controls,'price':{'body_forwards':counts[0],'sequences':counts[1],'native_parameters':sum(p.numel() for p in model.parameters()),'additional_coefficients':e.numel()+c.numel()+q.numel()+1,'native_weight_saving':0},
            'runner_sha256':digest(RUNNER),'binding_sha256':digest(BIND),'component_sha256':digest(ART),'wall_seconds':time.perf_counter()-tic,
            'scope':'Eight weight-pair fit lexicon, sixteen distinct test verbs, new authored frames, reused C. Conditional final-state or MLP scalar; cross-verb donor cue deltas, not independently extracted producer.'}
    atomic_create_json(OUT,result);print(json.dumps({'predictions':result['predictions'],'reports':{k:{'capability':v['capability'],'recovery':{a:s['raw_recovery'] for a,s in v['arms'].items()},'zero_removal_mean_ce':v['zero_removal']['mean_ce_damage']} for k,v in reports.items()},'wall_seconds':result['wall_seconds']}))

if __name__=='__main__':main()
