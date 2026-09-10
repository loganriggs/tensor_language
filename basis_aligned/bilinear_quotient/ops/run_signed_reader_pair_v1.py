#!/usr/bin/env python3
# BQGATE: native vector-valued signed reader pair;22forwards86seq,8arms.
"""pred_a instrument<=1e-3abs/1e-5rel; pred_b rareCE>=.10/frequent<=-.02;
pred_c projection-effecterror<=.10; pred_d randomrare<=.10full andfreq<=.05.
Full fixed gates SIGNED_READER_PAIR_V1_PREREGISTRATION.md. No rank/gain/pair rescue.
3456-scalar conditional component, all nativeparams/remainder retained.
"""
import json,os,sys,signal,time,math
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import signed_reader_pair_v1 as S
import calibration_two_readers_v1 as C
import bilin18_observed_model_facade as facade
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'SIGNED_READER_PAIR_V1_RESULT.json';BINDING=POLY/'SIGNED_READER_PAIR_V1_BINDING.json'


def bridge(a,b):
    a=a.double();b=b.double()
    return dict(max_abs=float((a-b).abs().max()),relative=float((a-b).norm()/b.norm().clamp_min(1e-30)))


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==v for p,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,forwards=22,sequences=86,controls=S.controls())));return
    assert not OUT.exists();signal.alarm(900)
    model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32);torch.set_num_threads(2);tic=time.perf_counter();counts=[0,0]
    def count(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count);M=model.transformer.h[17].mlp;U=model.lm_head.weight
    saved=torch.load(POLY/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt',map_location='cpu',weights_only=True);w=saved['w'].cuda();Q=saved['Q'].cuda();top=saved['top20'].numpy()
    pair=torch.load(POLY/'CALIBRATION_READER_ENERGY_V1_WITNESS.pt',map_location='cpu',weights_only=True)['normalized_inputs'].cuda();e1=(pair[0]+pair[1])/(2*math.sqrt(288));e2=(pair[0]-pair[1])/(2*math.sqrt(288))
    basis=torch.stack([e1,e2],1);orthogonal_error=float((basis.T@basis-torch.eye(2,device='cuda')).abs().max())
    rows=json.loads((POLY/'CALIBRATION_TWO_READERS_V1_ROWS.json').read_text())['rows'];reports={};per_row=[];bridges={};finite=True
    def logits(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(1152,)),U.float())/30)
    with torch.inference_mode():
        v=S.writer(M,e1,e2);cal=w*(w@v)/(w@w);other=v-cal;writers={'full':v,'calibration':cal,'complement':other}
        g=torch.Generator().manual_seed(9111236);random_basis=[torch.linalg.qr(torch.randn(1152,2,generator=g,dtype=torch.float64)).Q.cuda() for _ in range(3)]
        random_writers=[S.writer(M,E[:,0],E[:,1]) for E in random_basis]
        lhs=float((w@v)/(w@w));rhs=float(2*e1@Q@e2);correspondence=abs(lhs-rhs)/max(abs(rhs),1e-30)
        artifact=POLY/'SIGNED_READER_PAIR_V1_COMPONENT.pt';assert not artifact.exists();torch.save(dict(e1=e1.cpu(),e2=e2.cpu(),writer=v.cpu(),calibration_writer=cal.cpu(),complement_writer=other.cpu(),random_readers=torch.stack(random_basis).cpu(),random_writers=torch.stack(random_writers).cpu()),artifact)
        def body(ids,mode=None):
            cap={}
            def hook(_m,args,out):
                cap.update(u=args[0],m=out)
                if mode:return out-(S.coefficient(args[0],e1,e2)[...,None]*writers[mode]).to(out)
                return out
            mh=M.register_forward_hook(hook)
            try:
                x0=F.rms_norm(model.transformer.wte(ids),(1152,));x=x0;v0=None
                for block in model.transformer.h:x,v0=block(x,v0,x0)
                return x,cap
            finally:mh.remove()
        try:
            for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
                selected=[r for r in rows if r['split']==cohort];storage={k:[] for k in ('h','u','m')}
                for start in range(0,len(selected),4):
                    ids=torch.tensor([r['tokens'][:-1] for r in selected[start:start+4]],device='cuda');h,cap=body(ids)
                    storage['h'].append(h.cpu())
                    for k,value in cap.items():storage[k].append(value.cpu())
                    if start==0:
                        ab=S.coefficient(cap['u'],e1,e2)
                        for mode,writer in writers.items():
                            expected=C.read(h,ab,writer,U,True,True);online,_=body(ids,mode);bridges[cohort+'_online_'+mode]=bridge(logits(online),expected)
                        if cohort=='FW_HOLDOUT':
                            reference=facade.forward_with_dispatch(model,ids,lambda e:e.block.attn(e.state,e.first_value),lambda e:e.block.mlp(e.state),require_production=False)
                            bridges['independent_facade']=bridge(logits(h),reference)
                data={k:torch.cat(value).reshape(-1,1152) for k,value in storage.items()};targets=np.array([r['tokens'][1:] for r in selected]).ravel();common=np.isin(targets,top)
                losses={k:[] for k in ('native','full','calibration','complement','random0','random1','random2','old_q')};sq={k:0. for k in ('full','projection_error','complement','interaction','old_q_error')};native_error=0.;native_norm=0.;native_max=0.;ab_values=[]
                for start in range(0,len(targets),256):
                    h=data['h'][start:start+256].cuda();u=data['u'][start:start+256].cuda();m=data['m'][start:start+256].cuda();ab=S.coefficient(u,e1,e2);ab_values.extend(ab.cpu().tolist())
                    z={'native':C.read(h,ab,v,U,False,False),**{mode:C.read(h,ab,writer,U,True,True) for mode,writer in writers.items()}}
                    for i,E in enumerate(random_basis):z['random'+str(i)]=C.read(h,S.coefficient(u,E[:,0],E[:,1]),random_writers[i],U,True,True)
                    z['old_q']=C.read(h,m.double()@w/(w@w),w,U,True,True);actual=logits(h).double()
                    native_error+=float((z['native']-actual).square().sum());native_norm+=float(actual.square().sum());native_max=max(native_max,float((z['native']-actual).abs().max()))
                    target=torch.tensor(targets[start:start+256],device='cuda')
                    for k,zs in z.items():finite &= bool(torch.isfinite(zs).all());losses[k].extend(F.cross_entropy(zs,target,reduction='none').cpu().tolist())
                    def ss(a):return float((a-a.mean(-1,keepdim=True)).square().sum())
                    sq['full']+=ss(z['full']-z['native']);sq['projection_error']+=ss(z['calibration']-z['full']);sq['complement']+=ss(z['complement']-z['native']);sq['interaction']+=ss(z['full']-z['calibration']-z['complement']+z['native']);sq['old_q_error']+=ss(z['full']-z['old_q'])
                bridges[cohort+'_native_formula']=dict(max_abs=native_max,relative=(native_error/native_norm)**.5)
                loss={k:np.array(value) for k,value in losses.items()};means={k:dict(all=float(value.mean()),frequent=float(value[common].mean()),rare=float(value[~common].mean())) for k,value in loss.items()};effects={k:{cl:means[k][cl]-means['native'][cl] for cl in means[k]} for k in means if k!='native'}
                rel={k:(value/sq['full'])**.5 if sq['full']>1e-20 else None for k,value in sq.items() if k!='full'};full=effects['full'];random_rare=float(np.mean([abs(effects['random'+str(i)]['rare']) for i in range(3)]));random_freq=float(np.mean([abs(effects['random'+str(i)]['frequent']) for i in range(3)]))
                necessity=full['rare']>=.10 and full['frequent']<=-.02;projection=rel['projection_error'] is not None and rel['projection_error']<=.10 and all(abs(effects['calibration'][cl]-full[cl])<=max(.002,.20*abs(full[cl])) for cl in ('frequent','rare'));specific=full['rare']>0 and random_rare<=.10*full['rare'] and random_freq<=.05
                reports[cohort]=dict(rows=len(selected),mean_ce=means,ce_effects=effects,relative_effects=rel,calibration_necessity=bool(necessity),calibration_projection_sufficient=bool(projection),random_specificity=bool(specific),random_rare_abs_mean=random_rare,random_frequent_abs_mean=random_freq,ab_mean=float(np.mean(ab_values)),ab_sd=float(np.std(ab_values)))
                for i,r in enumerate(selected):
                    sl=slice(i*256,(i+1)*256);mask=common[sl]
                    per_row.append(dict(row_id=r['row_id'],split=cohort,frequent_count=int(mask.sum()),rare_count=int((~mask).sum()),ce_sums={k:dict(all=float(value[sl].sum()),frequent=float(value[sl][mask].sum()),rare=float(value[sl][~mask].sum())) for k,value in loss.items()}))
        finally:counter.remove()
    instrument=counts==[22,86] and finite and orthogonal_error<=1e-10 and correspondence<=1e-10 and all(v['max_abs']<=1e-3 and v['relative']<=1e-5 for v in bridges.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_calibration_necessity':all(r['calibration_necessity'] for r in reports.values()),'pred_c_calibration_projection_sufficient':all(r['calibration_projection_sufficient'] for r in reports.values()),'pred_d_random_specificity':all(r['random_specificity'] for r in reports.values())}
    out=dict(terminal='complete' if instrument else 'invalid',predictions=predictions,reports=reports,bridges=bridges,orthogonality_max_abs=orthogonal_error,scalar_weight_correspondence_relative=correspondence,writer_projection_norm_fraction=float(cal.norm()/v.norm()),writer_norm=float(v.norm()),per_row=per_row,price=dict(body_forwards=counts[0],sequences=counts[1],native_parameters=sum(p.numel() for p in model.parameters()),weight_saving=0,component_scalars=3456,component_file_bytes=artifact.stat().st_size),runner_sha256=digest(RUNNER),binding_sha256=digest(BINDING),component_sha256=digest(artifact),checkpoint_sha256=checkpoint.weights_sha256,wall_seconds=time.perf_counter()-tic,scope='Weight-defined complete native bilinear cross term; conditional normalized input and full native remainder retained, no semantic identification or fullmodel savings')
    atomic_create_json(OUT,out);print(json.dumps({k:out[k] for k in ('predictions','reports','bridges','writer_projection_norm_fraction','price','wall_seconds')},indent=2));assert instrument


if __name__=='__main__':main()
