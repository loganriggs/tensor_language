#!/usr/bin/env python3
# BQGATE: exact token/context quadratic split;18bodyforwards70seq,6arms.
"""pred_a instrument<=1e-3abs/1e-5rel; pred_b cross-term CE>=.005;
pred_c token, pred_d context, pred_e interaction sufficiency:scalar/error<=.10,
absCE<=.01 bothcorpora. Null: neither direct-token nor context-only producer.
All nativeparameters retained. Full protocol CALIBRATION_TOKEN_CONTEXT_V1_PREREGISTRATION.md.
"""
import os,sys,json,signal,time
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import calibration_two_readers_v1 as C
import calibration_scalar_path_v1 as P
import calibration_token_context_v1 as T
import bilin18_observed_model_facade as facade
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'CALIBRATION_TOKEN_CONTEXT_V1_RESULT.json';BINDING=POLY/'CALIBRATION_TOKEN_CONTEXT_V1_BINDING.json'


def bridge(a,b):
    a=a.double();b=b.double()
    return dict(max_abs=float((a-b).abs().max()),relative=float((a-b).norm()/b.norm().clamp_min(1e-30)))


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==v for p,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,body_forwards=18,sequences=70,controls=T.controls())));return
    assert not OUT.exists();signal.alarm(900)
    rows=json.loads((POLY/'CALIBRATION_TWO_READERS_V1_ROWS.json').read_text())['rows']
    model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32);torch.set_num_threads(2);tic=time.perf_counter();counts=[0,0]
    def count(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    saved=torch.load(POLY/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt',weights_only=True,map_location='cpu')
    w=saved['w'].cuda();Q=saved['Q'].cuda();beta=saved['beta'].cuda();top=saved['top20'].numpy();U=model.lm_head.weight
    lambdas=[block.lambdas.detach().double().cpu().tolist() for block in model.transformer.h];coefficient=T.embedding_coefficient(lambdas)
    bridges={};norm_exact=True;finite=True;reports={};per_row=[]
    def logits(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(1152,)),U.float())/30)
    def body(ids,remove_cross=False):
        cap={};mixed=None
        def ah(_m,_a,out):cap['a']=mixed+out[0]
        def mh(_m,args,out):
            cap.update(u=args[0],m=out)
            if remove_cross:
                tc=T.terms(cap['a'],cap['token'],Q)['interaction']
                return out-(tc[...,None]*w).to(out)
            return out
        h1=model.transformer.h[17].attn.register_forward_hook(ah);h2=model.transformer.h[17].mlp.register_forward_hook(mh)
        try:
            x0=F.rms_norm(model.transformer.wte(ids),(1152,));x=x0;v=None;cap['token']=coefficient*x0.double()
            for i,block in enumerate(model.transformer.h):
                if i==17:mixed=block.lambdas[0]*x+block.lambdas[1]*x0
                x,v=block(x,v,x0)
            return x,cap
        finally:h1.remove();h2.remove()
    try:
        with torch.inference_mode():
            for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
                selected=[r for r in rows if r['split']==cohort];storage={k:[] for k in ('h','a','u','m','token')}
                for start in range(0,len(selected),4):
                    ids=torch.tensor([r['tokens'][:-1] for r in selected[start:start+4]],device='cuda');h,cap=body(ids)
                    norm_exact &= bool(torch.equal(F.rms_norm(cap['a'],(1152,)),cap['u']))
                    storage['h'].append(h.cpu())
                    for k in cap:storage[k].append(cap[k].cpu())
                    if start==0:
                        q=C.scalar(cap['u'],Q,beta);cross=T.terms(cap['a'],cap['token'],Q)['interaction'];expected=P.path(h.double()-q[...,None]*w,q-cross,w,U)[0]
                        online,_=body(ids,True);bridges[cohort+'_online_cross_removal']=bridge(logits(online),expected)
                        if cohort=='FW_HOLDOUT':
                            reference=facade.forward_with_dispatch(model,ids,lambda e:e.block.attn(e.state,e.first_value),lambda e:e.block.mlp(e.state),require_production=False)
                            bridges['independent_facade']=bridge(logits(h),reference)
                data={k:torch.cat(v).reshape(-1,1152) for k,v in storage.items()};targets=np.array([r['tokens'][1:] for r in selected]).ravel();common=np.isin(targets,top)
                losses={k:[] for k in ('native','remove','no_interaction','token','context','interaction')}
                errors={k:0. for k in ('token','context','interaction')};scalar_errors={k:0. for k in errors};term_energy={k:0. for k in errors}
                effect_energy=0.;qenergy=0.;sum_error=0.;producer_error=0.;native_err=0.;native_norm=0.;native_max=0.
                for start in range(0,len(targets),256):
                    batch={k:v[start:start+256].cuda() for k,v in data.items()};q=C.scalar(batch['u'],Q,beta);parts=T.terms(batch['a'],batch['token'],Q)
                    direct=batch['m'].double()@w/(w@w);producer_error+=float((q-direct).square().sum());sum_error+=float((sum(parts.values())+beta-q).square().sum());qenergy+=float(q.square().sum())
                    h=batch['h'];g=h.double()-q[:,None]*w
                    scalars={'native':q,'remove':torch.zeros_like(q),'no_interaction':q-parts['interaction'],**{k:parts[k]+beta for k in parts}}
                    z={k:P.path(g,s,w,U)[0] for k,s in scalars.items()}
                    actual=logits(h).double();native_err+=float((z['native']-actual).square().sum());native_norm+=float(actual.square().sum());native_max=max(native_max,float((z['native']-actual).abs().max()))
                    for k in parts:scalar_errors[k]+=float((scalars[k]-q).square().sum());term_energy[k]+=float(parts[k].square().sum())
                    target=torch.tensor(targets[start:start+256],device='cuda')
                    for k,zs in z.items():finite &= bool(torch.isfinite(zs).all());losses[k].extend(F.cross_entropy(zs,target,reduction='none').cpu().tolist())
                    def ss(a):return float((a-a.mean(-1,keepdim=True)).square().sum())
                    effect_energy+=ss(z['remove']-z['native'])
                    for k in errors:errors[k]+=ss(z[k]-z['native'])
                bridges[cohort+'_native_formula']=dict(max_abs=native_max,relative=(native_err/native_norm)**.5)
                loss={k:np.array(v) for k,v in losses.items()};means={k:dict(all=float(v.mean()),frequent=float(v[common].mean()),rare=float(v[~common].mean())) for k,v in loss.items()};effects={k:{c:means[k][c]-means['native'][c] for c in means[k]} for k in means if k!='native'}
                rel={k:(v/effect_energy)**.5 if effect_energy>1e-20 else None for k,v in errors.items()};srel={k:(v/qenergy)**.5 for k,v in scalar_errors.items()}
                sufficient={k:bool(rel[k] is not None and rel[k]<=.10 and srel[k]<=.10 and abs(effects[k]['all'])<=.01) for k in errors}
                reports[cohort]=dict(rows=len(selected),mean_ce=means,ce_effects=effects,scalar_relative_errors=srel,effect_relative_errors=rel,sufficient=sufficient,source_sum_relative_error=(sum_error/qenergy)**.5,fold_relative_error=(producer_error/qenergy)**.5,term_rms_relative_to_q={k:(v/qenergy)**.5 for k,v in term_energy.items()},interaction_necessary=bool(effects['no_interaction']['all']>=.005))
                for i,r in enumerate(selected):
                    sl=slice(i*256,(i+1)*256);mask=common[sl]
                    per_row.append(dict(row_id=r['row_id'],split=cohort,frequent_count=int(mask.sum()),rare_count=int((~mask).sum()),ce_sums={k:dict(all=float(v[sl].sum()),frequent=float(v[sl][mask].sum()),rare=float(v[sl][~mask].sum())) for k,v in loss.items()}))
    finally:counter.remove()
    instrument=counts==[18,70] and finite and norm_exact and all(r['source_sum_relative_error']<=1e-5 and r['fold_relative_error']<=1e-5 for r in reports.values()) and all(v['max_abs']<=1e-3 and v['relative']<=1e-5 for v in bridges.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_interaction_necessity':all(r['interaction_necessary'] for r in reports.values()),'pred_c_token_sufficiency':all(r['sufficient']['token'] for r in reports.values()),'pred_d_context_sufficiency':all(r['sufficient']['context'] for r in reports.values()),'pred_e_interaction_sufficiency':all(r['sufficient']['interaction'] for r in reports.values())}
    out=dict(terminal='complete' if instrument else 'invalid',predictions=predictions,reports=reports,bridges=bridges,raw_input_native_normalization_exact=norm_exact,embedding_coefficient=coefficient,native_lambdas=lambdas,per_row=per_row,price=dict(body_forwards=counts[0],sequences=counts[1],native_parameters=sum(p.numel() for p in model.parameters()),weight_saving=0),runner_sha256=digest(RUNNER),binding_sha256=digest(BINDING),checkpoint_sha256=checkpoint.weights_sha256,wall_seconds=time.perf_counter()-tic,scope='Direct token-injection route and native complement, conditional normalization retained; compiled-term edits, not deletion of upstream native injections')
    atomic_create_json(OUT,out);print(json.dumps({k:out[k] for k in ('predictions','reports','bridges','embedding_coefficient','price','wall_seconds')},indent=2));assert instrument


if __name__=='__main__':main()
