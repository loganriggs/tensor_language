#!/usr/bin/env python3
# BQGATE: exact folded calibration scalar and numerator/RMS uses;30bodyforwards118seq.
import json,os,signal,sys,time
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import calibration_two_readers_v1 as C
import bilin18_observed_model_facade as facade
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'CALIBRATION_TWO_READERS_V1_RESULT.json'
BINDING=POLY/'CALIBRATION_TWO_READERS_V1_BINDING.json'
ROWS=POLY/'CALIBRATION_TWO_READERS_V1_ROWS.json'


def bridge(a,b):
    a=a.double();b=b.double()
    return dict(max_abs=float((a-b).abs().max()),relative=float((a-b).norm()/b.norm().clamp_min(1e-30)))


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==v for p,v in binding.items())
    manifest=json.loads(ROWS.read_text());rows=manifest['rows']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,forwards=30,sequences=118,fit_rows=48,fw_rows=42,pile_rows=16,controls=C.controls())));return
    assert not OUT.exists();signal.alarm(900)
    model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32)
    torch.set_num_threads(2);tic=time.perf_counter();counts=[0,0]
    def count(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    handle=model.transformer.h[0].attn.register_forward_hook(count)
    U=model.lm_head.weight;M=model.transformer.h[17].mlp;D=1152
    def logits(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(D,)),U.float())/30)
    def body(ids,remove=None,Q=None,beta=None):
        capture={}
        def mlp_hook(_m,a,out):
            capture.update(u=a[0].detach(),m=out.detach())
            if remove is None:return out
            q=C.scalar(a[0],Q,beta) if Q is not None else out.double()@remove/(remove@remove)
            return out-(q[...,None]*remove).to(out)
        hook=M.register_forward_hook(mlp_hook)
        try:
            x=F.rms_norm(model.transformer.wte(ids),(D,));x0=x;v=None
            for block in model.transformer.h:x,v=block(x,v,x0)
            return x,capture['u'],capture['m']
        finally:hook.remove()
    fits=[r for r in rows if r['split']=='FIT'];hist=np.bincount(np.array([r['tokens'][1:] for r in fits]).ravel(),minlength=50304)
    top=np.argsort(-hist,kind='stable')[:20];fit_m=[];bridges={}
    try:
        with torch.inference_mode():
            for start in range(0,len(fits),4):
                ids=torch.tensor([r['tokens'][:-1] for r in fits[start:start+4]],device='cuda')
                h,u,m=body(ids);fit_m.append(m.reshape(-1,D).cpu())
                if start==0:
                    def attn(e):return e.block.attn(e.state,e.first_value)
                    def mlp(e):return e.block.mlp(e.state)
                    reference=facade.forward_with_dispatch(model,ids,attn,mlp,require_production=False)
                    bridges['independent_native_facade']=bridge(logits(h),reference)
            output=torch.cat(fit_m).double();del fit_m
            labels=torch.tensor(np.log1p(hist[np.array([r['tokens'][1:] for r in fits]).ravel()]),dtype=torch.float64)
            centered=labels-labels.mean();w=(output-output.mean(0)).T@centered
            assert float(w.norm())>1e-8;w=(w/w.norm()).cuda();del output
            Q,beta=C.fold(M,w)
            rng=torch.Generator(device='cpu').manual_seed(9111130)
            random=torch.randn(3,D,generator=rng,dtype=torch.float64);random=(random/random.norm(dim=1,keepdim=True)).cuda()
            producer_path=POLY/'CALIBRATION_TWO_READERS_V1_PRODUCER.pt'
            assert not producer_path.exists();torch.save(dict(w=w.cpu(),Q=Q.cpu(),beta=beta.cpu(),random=random.cpu(),top20=torch.as_tensor(top),fit_rows=48),producer_path)
            reports={};per_row=[];producer_errors={};first_eval=True;all_finite=True
            for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
                selected=[r for r in rows if r['split']==cohort];hs=[];us=[];ms=[]
                for start in range(0,len(selected),4):
                    part=selected[start:start+4];ids=torch.tensor([r['tokens'][:-1] for r in part],device='cuda')
                    h,u,m=body(ids);hs.append(h.cpu());us.append(u.cpu());ms.append(m.cpu())
                    if first_eval:
                        q=C.scalar(u,Q,beta);expected=C.read(h,q,w,U,True,True)
                        online,_,_=body(ids,remove=w);compiled,_,_=body(ids,remove=w,Q=Q,beta=beta)
                        bridges['online_native_removal']=bridge(logits(online),expected)
                        bridges['online_compiled_removal']=bridge(logits(compiled),expected)
                        first_eval=False
                hs=torch.cat(hs).reshape(-1,D);us=torch.cat(us).reshape(-1,D);ms=torch.cat(ms).reshape(-1,D)
                targets=np.array([r['tokens'][1:] for r in selected]).reshape(-1)
                frequent=np.isin(targets,top);losses={name:[] for name in ('native','numerator','denominator','joint','random0','random1','random2')}
                squared={name:0. for name in ('denominator_effect','numerator_error','denominator_error','interaction')}
                p_err=0.;p_norm=0.;native_max=0.;native_err=0.;native_norm=0.
                for start in range(0,len(hs),256):
                    end=min(start+256,len(hs));h=hs[start:end].cuda();u=us[start:end].cuda();m=ms[start:end].cuda()
                    q=C.scalar(u,Q,beta);direct=m.double()@w/(w@w)
                    p_err+=float((q-direct).square().sum());p_norm+=float(direct.square().sum())
                    z={name:C.read(h,q,w,U,n,d) for name,n,d in (('native',False,False),('numerator',True,False),('denominator',False,True),('joint',True,True))}
                    actual=logits(h).double();native_max=max(native_max,float((z['native']-actual).abs().max()))
                    native_err+=float((z['native']-actual).square().sum());native_norm+=float(actual.square().sum())
                    for i in range(3):
                        qr=m.double()@random[i]/(random[i]@random[i]);z['random'+str(i)]=C.read(h,qr,random[i],U,True,True)
                    target=torch.tensor(targets[start:end],device='cuda')
                    for name,a in z.items():
                        all_finite &= bool(torch.isfinite(a).all())
                        losses[name].extend(F.cross_entropy(a,target,reduction='none').cpu().tolist())
                    def ss(a):a=a-a.mean(-1,keepdim=True);return float(a.square().sum())
                    squared['denominator_effect']+=ss(z['joint']-z['native'])
                    squared['numerator_error']+=ss(z['numerator']-z['joint'])
                    squared['denominator_error']+=ss(z['denominator']-z['joint'])
                    squared['interaction']+=ss(z['joint']-z['numerator']-z['denominator']+z['native'])
                    del z,actual
                producer_errors[cohort]=(p_err/p_norm)**.5
                bridges[cohort+'_native_formula']=dict(max_abs=native_max,relative=(native_err/native_norm)**.5)
                loss={k:np.array(v) for k,v in losses.items()};means={k:{'frequent':float(v[frequent].mean()),'rare':float(v[~frequent].mean())} for k,v in loss.items()}
                effects={k:{c:means[k][c]-means['native'][c] for c in ('frequent','rare')} for k in means if k!='native'}
                joint=effects['joint'];random_rare=float(np.mean([abs(effects['random'+str(i)]['rare']) for i in range(3)]));random_freq=float(np.mean([abs(effects['random'+str(i)]['frequent']) for i in range(3)]))
                signature=joint['rare']>=.10 and joint['frequent']<=-.02 and random_rare<=.10*joint['rare'] and random_freq<=.05
                errors={k:(squared[k+'_error']/squared['denominator_effect'])**.5 if squared['denominator_effect']>1e-20 else None for k in ('numerator','denominator')}
                sufficient={k:bool(errors[k] is not None and errors[k]<=.10 and all(abs(effects[k][c]-joint[c])<=max(.002,.20*abs(joint[c])) for c in ('frequent','rare'))) for k in errors}
                reports[cohort]=dict(rows=len(selected),tokens=len(targets),frequent_tokens=int(frequent.sum()),mean_ce=means,ce_effects=effects,calibration_signature=bool(signature),random_rare_absolute_mean=random_rare,random_frequent_absolute_mean=random_freq,consumer_errors=errors,consumer_sufficient=sufficient,relative_joint_interaction=(squared['interaction']/squared['denominator_effect'])**.5)
                for i,r in enumerate(selected):
                    sl=slice(i*256,(i+1)*256);mask=frequent[sl]
                    per_row.append(dict(row_id=r['row_id'],split=cohort,frequent_count=int(mask.sum()),rare_count=int((~mask).sum()),ce_sums={k:{'frequent':float(v[sl][mask].sum()),'rare':float(v[sl][~mask].sum())} for k,v in loss.items()}))
    finally:handle.remove()
    instrument=counts==[30,118] and all_finite and all(v<=1e-5 for v in producer_errors.values()) and all(v['max_abs']<=1e-3 and v['relative']<=1e-5 for v in bridges.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_heldout_calibration':all(r['calibration_signature'] for r in reports.values()),'pred_c_numerator_sufficiency':all(r['consumer_sufficient']['numerator'] for r in reports.values()),'pred_d_denominator_sufficiency':all(r['consumer_sufficient']['denominator'] for r in reports.values())}
    out=dict(terminal='complete' if instrument else 'invalid',predictions=predictions,reports=reports,producer_relative_errors=producer_errors,bridges=bridges,per_row=per_row,fit_direction_uses_fit_target_frequencies=True,evaluation_producer_uses_no_target_labels=True,price=dict(body_forwards=counts[0],sequences=counts[1],native_parameters=sum(p.numel() for p in model.parameters()),weight_saving=0,folded_producer_scalars=Q.numel()+w.numel()+1,folded_producer_file_bytes=producer_path.stat().st_size()),producer_sha256=digest(producer_path),runner_sha256=digest(RUNNER),binding_sha256=digest(BINDING),checkpoint_sha256=checkpoint.weights_sha256,wall_seconds=time.perf_counter()-tic,scope='Conditional weight-folded producer and two readout uses; full native background retained; FineWeb row holdout and Pile corpus shift, no four-property adoption')
    atomic_create_json(OUT,out);print(json.dumps({k:out[k] for k in ('predictions','reports','producer_relative_errors','bridges','price','wall_seconds')},indent=2));assert instrument


if __name__=='__main__':main()
