#!/usr/bin/env python3
# BQGATE: weight-only reader-energy coefficients;18bodyforwards70seq,4arms.
"""pred_a instrument<=1e-3abs/1e-5rel; pred_b coefficientresidual<=1e-10;
pred_c scalar/effecterror<=.10 andabsCE<=.01bothcohorts; pred_d controlCEreplay<=1e-10.
Null: calibration is not this weight-defined vocabulary-spread computation.
No fitting axes/activations, no rank sweep. All nativeweights retained.
Full protocol CALIBRATION_READER_ENERGY_V1_PREREGISTRATION.md.
"""
import os,sys,json,time,signal
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import calibration_two_readers_v1 as C
import calibration_scalar_path_v1 as P
import reader_energy_certificate_v1 as E
import bilin18_observed_model_facade as facade
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'CALIBRATION_READER_ENERGY_V1_RESULT.json';BINDING=POLY/'CALIBRATION_READER_ENERGY_V1_BINDING.json'


def bridge(a,b):
    a=a.double();b=b.double()
    return dict(max_abs=float((a-b).abs().max()),relative=float((a-b).norm()/b.norm().clamp_min(1e-30)))


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==v for p,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,forwards=18,sequences=70,controls=E.controls())));return
    assert not OUT.exists();signal.alarm(900)
    model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32);torch.set_num_threads(2);tic=time.perf_counter();counts=[0,0]
    def count(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    U=model.lm_head.weight;M=model.transformer.h[17].mlp
    saved=torch.load(POLY/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt',weights_only=True,map_location='cpu');w=saved['w'].cuda();Q=saved['Q'].cuda();beta=saved['beta'].cuda();top=saved['top20'].numpy()
    rows=json.loads((POLY/'CALIBRATION_TWO_READERS_V1_ROWS.json').read_text())['rows'];parent=json.loads((POLY/'CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json').read_text());meanq=parent['fit_stats']['mean']
    reports={};bridges={};per_row=[];finite=True
    def logits(h):return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(1152,)),U.float())/30)
    with torch.inference_mode():
        K=E.reader_gram(U);cert=E.certificate(Q,K);assert cert['identified'],'K0 degenerate'
        alpha=cert['slope'];gamma=cert['isotropic'];foldQ,foldbeta=C.fold(M,w)
        Qreplay=bridge(foldQ,Q);beta_error=float((foldbeta-beta).abs())
        rng=torch.Generator().manual_seed(9111225);samples=torch.randn(16,1152,generator=rng,dtype=torch.float64).cuda();samples=samples/samples.norm(dim=1,keepdim=True)*1152**.5
        direct=(samples@U.double().T).var(-1,unbiased=False);gram=C.scalar(samples,K,torch.zeros_like(beta));gram_bridge=bridge(gram,direct)
        artifact=POLY/'CALIBRATION_READER_ENERGY_V1_GRAM.pt';assert not artifact.exists();torch.save(dict(K=K.cpu(),alpha=alpha,gamma=gamma,beta=beta.cpu()),artifact)
        def energy(u):return alpha*C.scalar(u,K,torch.zeros_like(beta))+gamma*u.double().square().sum(-1)+beta
        def body(ids,replace=False):
            cap={}
            def hook(_m,args,out):
                cap.update(u=args[0],m=out)
                if replace:
                    current=out.double()@w/(w@w);proposed=energy(args[0])
                    return out+((proposed-current)[...,None]*w).to(out)
                return out
            mh=M.register_forward_hook(hook)
            try:
                x0=F.rms_norm(model.transformer.wte(ids),(1152,));x=x0;v=None
                for block in model.transformer.h:x,v=block(x,v,x0)
                return x,cap
            finally:mh.remove()
        try:
            for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
                selected=[r for r in rows if r['split']==cohort];storage={k:[] for k in ('h','u','m')}
                for start in range(0,len(selected),4):
                    ids=torch.tensor([r['tokens'][:-1] for r in selected[start:start+4]],device='cuda');h,cap=body(ids)
                    storage['h'].append(h.cpu())
                    for k,v in cap.items():storage[k].append(v.cpu())
                    if start==0:
                        q=C.scalar(cap['u'],Q,beta);expected=P.path(h.double()-q[...,None]*w,energy(cap['u']),w,U)[0]
                        online,_=body(ids,True);bridges[cohort+'_online_energy']=bridge(logits(online),expected)
                        if cohort=='FW_HOLDOUT':
                            ref=facade.forward_with_dispatch(model,ids,lambda e:e.block.attn(e.state,e.first_value),lambda e:e.block.mlp(e.state),require_production=False)
                            bridges['independent_facade']=bridge(logits(h),ref)
                data={k:torch.cat(v).reshape(-1,1152) for k,v in storage.items()};targets=np.array([r['tokens'][1:] for r in selected]).ravel();common=np.isin(targets,top)
                losses={k:[] for k in ('native','remove','energy','mean')};sq_error=0.;sq_norm=0.;effect_error=0.;effect_norm=0.;native_error=0.;native_norm=0.;native_max=0.;qe_sum=0.;q_sum=0.
                for start in range(0,len(targets),256):
                    h=data['h'][start:start+256].cuda();u=data['u'][start:start+256].cuda();q=C.scalar(u,Q,beta);qe=energy(u);g=h.double()-q[:,None]*w
                    sq_error+=float((qe-q).square().sum());sq_norm+=float(q.square().sum());qe_sum+=float(qe.sum());q_sum+=float(q.sum())
                    scalars={'native':q,'remove':torch.zeros_like(q),'energy':qe,'mean':torch.full_like(q,meanq)}
                    z={k:P.path(g,s,w,U)[0] for k,s in scalars.items()};actual=logits(h).double()
                    native_error+=float((z['native']-actual).square().sum());native_norm+=float(actual.square().sum());native_max=max(native_max,float((z['native']-actual).abs().max()))
                    target=torch.tensor(targets[start:start+256],device='cuda')
                    for k,zs in z.items():finite &= bool(torch.isfinite(zs).all());losses[k].extend(F.cross_entropy(zs,target,reduction='none').cpu().tolist())
                    def ss(a):return float((a-a.mean(-1,keepdim=True)).square().sum())
                    effect_error+=ss(z['energy']-z['native']);effect_norm+=ss(z['remove']-z['native'])
                bridges[cohort+'_native_formula']=dict(max_abs=native_max,relative=(native_error/native_norm)**.5)
                loss={k:np.array(v) for k,v in losses.items()};means={k:dict(all=float(v.mean()),frequent=float(v[common].mean()),rare=float(v[~common].mean())) for k,v in loss.items()};effects={k:{c:means[k][c]-means['native'][c] for c in means[k]} for k in means if k!='native'}
                scalar_rel=(sq_error/sq_norm)**.5;effect_rel=(effect_error/effect_norm)**.5 if effect_norm>1e-20 else None
                reports[cohort]=dict(rows=len(selected),scalar_relative_error=scalar_rel,effect_relative_error=effect_rel,q_mean=q_sum/len(targets),energy_q_mean=qe_sum/len(targets),mean_ce=means,ce_effects=effects,sufficient=bool(scalar_rel<=.10 and effect_rel is not None and effect_rel<=.10 and abs(effects['energy']['all'])<=.01))
                for i,r in enumerate(selected):
                    sl=slice(i*256,(i+1)*256);mask=common[sl]
                    per_row.append(dict(row_id=r['row_id'],split=cohort,frequent_count=int(mask.sum()),rare_count=int((~mask).sum()),ce_sums={k:dict(all=float(v[sl].sum()),frequent=float(v[sl][mask].sum()),rare=float(v[sl][~mask].sum())) for k,v in loss.items()}))
        finally:counter.remove()
    replay=max(abs(reports[c]['mean_ce'][arm][cls]-parent['reports'][c]['mean_ce'][{'remove':'remove_ref'}.get(arm,arm)][cls]) for c in reports for arm in ('native','remove','mean') for cls in ('all','frequent','rare'))
    instrument=counts==[18,70] and finite and Qreplay['relative']<=1e-10 and beta_error<=1e-10 and gram_bridge['relative']<=1e-10 and all(v['max_abs']<=1e-3 and v['relative']<=1e-5 for v in bridges.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_global_weight_compatibility':bool(cert['relative_anisotropic_residual']<=1e-10),'pred_c_native_sufficiency':all(v['sufficient'] for v in reports.values()),'pred_d_unchanged_control_replay':bool(replay<=1e-10)}
    instrument=instrument and predictions['pred_d_unchanged_control_replay']
    out=dict(terminal='complete' if instrument else 'invalid',predictions=predictions,coefficient_fit=cert,gram_bridge=gram_bridge,Q_replay=Qreplay,beta_replay_error=beta_error,reports=reports,bridges=bridges,per_row=per_row,unchanged_control_max_CE_error=replay,price=dict(body_forwards=counts[0],sequences=counts[1],native_parameters=sum(p.numel() for p in model.parameters()),weight_saving=0,reader_parameters=U.numel(),gram_artifact_bytes=artifact.stat().st_size),runner_sha256=digest(RUNNER),binding_sha256=digest(BINDING),gram_sha256=digest(artifact),checkpoint_sha256=checkpoint.weights_sha256,wall_seconds=time.perf_counter()-tic,scope='Weight-defined vocabulary-spread computation tested globally and on reused native contexts; no activation fit, entropy interpretation or independent extraction claim')
    atomic_create_json(OUT,out);print(json.dumps({k:out[k] for k in ('predictions','coefficient_fit','gram_bridge','reports','bridges','price','wall_seconds')},indent=2));assert instrument


if __name__=='__main__':main()
