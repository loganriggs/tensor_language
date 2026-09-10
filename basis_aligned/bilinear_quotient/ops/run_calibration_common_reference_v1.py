#!/usr/bin/env python3
# BQGATE: controlled shared frequency reference; unchanged32forwards126seq and20%effect bar.
"""Shared-reference diagnostic; pred_d unchanged-arm replay <=1e-12 CE.
pred_a instrument <=1e-3abs/1e-5rel; pred_b stable effects <=.20;
pred_c mean and donor CE damage >=.005 both cohorts. Null: constant offset
or fitting-dependent projection. Price32bodyforwards126seq; nativeweights retained.
Arm formulas and full gates: CALIBRATION_COMMON_REFERENCE_V1_PREREGISTRATION.md.
"""
import json,os,signal,sys,time
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import calibration_two_readers_v1 as C
import calibration_scalar_path_v1 as P
import bilin18_observed_model_facade as facade
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'CALIBRATION_COMMON_REFERENCE_V1_RESULT.json'
BINDING=POLY/'CALIBRATION_COMMON_REFERENCE_V1_BINDING.json'
ROWS=POLY/'CALIBRATION_TWO_READERS_V1_ROWS.json'
PRODUCER=POLY/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt'


def bridge(a,b):
    a=a.double();b=b.double()
    return dict(max_abs=float((a-b).abs().max()),relative=float((a-b).norm()/b.norm().clamp_min(1e-30)))


def fit_axis(outputs,rows):
    targets=np.array([r['tokens'][1:] for r in rows]).ravel()
    reference_rows=[r for r in json.loads(ROWS.read_text())['rows'] if r['split']=='FIT']
    reference_targets=np.array([r['tokens'][1:] for r in reference_rows]).ravel()
    hist=np.bincount(reference_targets,minlength=50304)
    labels=torch.tensor(np.log1p(hist[targets]),dtype=torch.float64)
    x=outputs.reshape(-1,1152).double();a=(x-x.mean(0)).T@(labels-labels.mean())
    assert float(a.norm())>1e-8
    return a/a.norm()


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==sha for p,sha in binding.items())
    rows=json.loads(ROWS.read_text())['rows']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,forwards=32,sequences=126,controls=P.controls())));return
    assert not OUT.exists();signal.alarm(900)
    model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32)
    torch.set_num_threads(2);tic=time.perf_counter();counts=[0,0];bridges={}
    def count(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    handle=model.transformer.h[0].attn.register_forward_hook(count)
    U=model.lm_head.weight;M=model.transformer.h[17].mlp;D=1152
    saved=torch.load(PRODUCER,weights_only=True,map_location='cpu')
    w=saved['w'].cuda();Q=saved['Q'].cuda();beta=saved['beta'].cuda();top=saved['top20'].numpy()
    def logits(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(D,)),U.float())/30)
    def body(ids,donor=None,compiled=False):
        capture={}
        def hook(_m,a,out):
            capture.update(u=a[0].detach(),m=out.detach())
            if donor is None:return out
            q=C.scalar(a[0],Q,beta) if compiled else out.double()@w/(w@w)
            return out+((donor-q)[...,None]*w).to(out)
        mh=M.register_forward_hook(hook)
        try:
            x=F.rms_norm(model.transformer.wte(ids),(D,));x0=x;v=None
            for block in model.transformer.h:x,v=block(x,v,x0)
            return x,capture['u'],capture['m']
        finally:mh.remove()
    reports={};per_row=[];producer_errors={};finite=True
    try:
        with torch.inference_mode():
            fits=[r for r in rows if r['split']=='FIT'];fit_m=[]
            for start in range(0,48,4):
                ids=torch.tensor([r['tokens'][:-1] for r in fits[start:start+4]],device='cuda')
                h,u,m=body(ids);fit_m.append(m.cpu())
                if start==0:
                    reference=facade.forward_with_dispatch(model,ids,lambda e:e.block.attn(e.state,e.first_value),lambda e:e.block.mlp(e.state),require_production=False)
                    bridges['independent_native_facade']=bridge(logits(h),reference)
            fit_m=torch.cat(fit_m);replay=bridge(fit_axis(fit_m,fits),saved['w'])
            axes={k:v.cuda() for k,v in {'A':fit_axis(fit_m[:24],fits[:24]),'B':fit_axis(fit_m[24:],fits[24:])}.items()}
            split_cosine=float(axes['A']@axes['B'])
            fit_q=fit_m.double()@saved['w']/(saved['w']@saved['w']);mean_q=fit_q.mean().cuda()
            fit_stats=dict(mean=float(mean_q),sd=float(fit_q.std(unbiased=False)),direction_replay=replay)
            del fit_m,fit_q
            artifact=POLY/'CALIBRATION_COMMON_REFERENCE_V1_AXES.pt';assert not artifact.exists()
            torch.save(dict(A=axes['A'].cpu(),B=axes['B'].cpu(),fit_mean_q=mean_q.cpu()),artifact)
            for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
                selected=[r for r in rows if r['split']==cohort];hs=[];us=[];ms=[]
                for start in range(0,len(selected),4):
                    ids=torch.tensor([r['tokens'][:-1] for r in selected[start:start+4]],device='cuda')
                    h,u,m=body(ids);hs.append(h.cpu());us.append(u.cpu());ms.append(m.cpu())
                hs=torch.cat(hs);us=torch.cat(us);ms=torch.cat(ms)
                q_all=C.scalar(us.cuda(),Q,beta).cpu();native_q=ms.double()@saved['w']/(saved['w']@saved['w'])
                producer_errors[cohort]=bridge(q_all,native_q)['relative']
                donor_q=q_all.roll(-1,0);donor_native=native_q.roll(-1,0)
                ids=torch.tensor([r['tokens'][:-1] for r in selected[:4]],device='cuda')
                expected=P.path(hs[:4].cuda().double()-q_all[:4].cuda()[...,None]*w,donor_q[:4].cuda(),w,U)[0]
                for name,dq,compiled in [('native',donor_native,False),('compiled',donor_q,True)]:
                    online,_,_=body(ids,donor=dq[:4].cuda(),compiled=compiled)
                    bridges[cohort+'_online_'+name+'_donor']=bridge(logits(online),expected)
                del expected,online
                targets=np.array([r['tokens'][1:] for r in selected]).ravel();common=np.isin(targets,top)
                losses={k:[] for k in ('native','remove_ref','remove_A','remove_B','mean','donor')}
                effect2=0.;err2={'A':0.,'B':0.};native_err=0.;native_norm=0.;native_max=0.
                hflat=hs.reshape(-1,D);mflat=ms.reshape(-1,D);qflat=q_all.reshape(-1);dqflat=donor_q.reshape(-1)
                for start in range(0,len(hflat),256):
                    h=hflat[start:start+256].cuda();m=mflat[start:start+256].cuda();q=qflat[start:start+256].cuda()
                    b=h.double()-q[:,None]*w
                    z={'native':C.read(h,q,w,U,False,False),'remove_ref':C.read(h,q,w,U,True,True)}
                    for k,a in axes.items():z['remove_'+k]=C.read(h,m.double()@a/(a@a),a,U,True,True)
                    z['mean']=P.path(b,mean_q.expand(len(h)),w,U)[0]
                    z['donor']=P.path(b,dqflat[start:start+256].cuda(),w,U)[0]
                    actual=logits(h).double();native_err+=float((z['native']-actual).square().sum());native_norm+=float(actual.square().sum());native_max=max(native_max,float((z['native']-actual).abs().max()))
                    target=torch.tensor(targets[start:start+256],device='cuda')
                    for k,value in z.items():
                        finite &= bool(torch.isfinite(value).all());losses[k].extend(F.cross_entropy(value,target,reduction='none').cpu().tolist())
                    def ss(a):return float((a-a.mean(-1,keepdim=True)).square().sum())
                    effect2+=ss(z['remove_ref']-z['native'])
                    for k in axes:err2[k]+=ss(z['remove_'+k]-z['remove_ref'])
                    del z,actual
                bridges[cohort+'_native_formula']=dict(max_abs=native_max,relative=(native_err/native_norm)**.5)
                losses={k:np.array(v) for k,v in losses.items()}
                means={k:dict(all=float(v.mean()),frequent=float(v[common].mean()),rare=float(v[~common].mean())) for k,v in losses.items()}
                effects={k:{c:means[k][c]-means['native'][c] for c in means[k]} for k in means if k!='native'}
                errors={k:(err2[k]/effect2)**.5 if effect2>1e-20 else None for k in axes}
                stable=all(errors[k] is not None and errors[k]<=.20 and effects['remove_'+k]['rare']>=.10 and effects['remove_'+k]['frequent']<=-.02 for k in axes)
                useful=all(effects[k]['all']>=.005 for k in ('mean','donor'))
                reports[cohort]=dict(rows=len(selected),tokens=len(targets),mean_ce=means,ce_effects=effects,split_effect_relative_errors=errors,operational_stability=bool(stable),useful_pairing=bool(useful),q_mean=float(q_all.mean()),q_sd=float(q_all.std(unbiased=False)),q_centered_energy_fraction=float((q_all-q_all.mean()).square().sum()/q_all.square().sum()),donor_minus_mean_ce=effects['donor']['all']-effects['mean']['all'])
                for i,r in enumerate(selected):
                    sl=slice(i*256,(i+1)*256);mask=common[sl]
                    per_row.append(dict(row_id=r['row_id'],donor_row_id=selected[(i+1)%len(selected)]['row_id'],split=cohort,frequent_count=int(mask.sum()),rare_count=int((~mask).sum()),q_mean=float(q_all[i].mean()),q_sd=float(q_all[i].std(unbiased=False)),ce_sums={k:dict(all=float(v[sl].sum()),frequent=float(v[sl][mask].sum()),rare=float(v[sl][~mask].sum())) for k,v in losses.items()}))
    finally:handle.remove()
    instrument=counts==[32,126] and finite and replay['relative']<=1e-10 and all(v<=1e-5 for v in producer_errors.values()) and all(v['max_abs']<=1e-3 and v['relative']<=1e-5 for v in bridges.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_operational_stability':bool(abs(split_cosine)>=.90 and all(r['operational_stability'] for r in reports.values())),'pred_c_useful_context_pairing':bool(all(r['useful_pairing'] for r in reports.values()))}
    parent=json.loads((POLY/'CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json').read_text())
    replay_max=max(abs(reports[c]['mean_ce'][arm][cls]-parent['reports'][c]['mean_ce'][arm][cls]) for c in reports for arm in ('native','remove_ref','mean','donor') for cls in ('all','frequent','rare'))
    predictions['pred_d_unchanged_arm_replay']=bool(replay_max<=1e-12)
    instrument=instrument and predictions['pred_d_unchanged_arm_replay']
    out=dict(reference_histogram='frozen full48FIT rows, shared by both24-row fits',unchanged_arm_replay_max_ce_error=replay_max,parent_result_sha256=digest(POLY/'CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json'),terminal='complete' if instrument else 'invalid',predictions=predictions,reports=reports,per_row=per_row,split_axis_cosine=split_cosine,fit_stats=fit_stats,producer_relative_errors=producer_errors,bridges=bridges,price=dict(body_forwards=counts[0],sequences=counts[1],native_parameters=sum(p.numel() for p in model.parameters()),weight_saving=0,axes_file_bytes=artifact.stat().st_size),runner_sha256=digest(RUNNER),binding_sha256=digest(BINDING),axes_sha256=digest(artifact),checkpoint_sha256=checkpoint.weights_sha256,wall_seconds=time.perf_counter()-tic,scope='Conditional interchange execution, disjoint-row fit stability, reused evaluation text; not semantic donor-task transfer or full extraction')
    atomic_create_json(OUT,out);print(json.dumps({k:out[k] for k in ('predictions','reports','split_axis_cosine','fit_stats','bridges','price','wall_seconds')},indent=2));assert instrument


if __name__=='__main__':main()
