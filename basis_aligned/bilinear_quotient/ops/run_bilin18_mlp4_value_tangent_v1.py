"""Receiver-only Jacobian-vector prediction of finite MLP4-to-localV9 response.
A: parent/primal/identity1e-3/1e-5,CPUcentraldiff1e-8/1e-6,48/864counts.
B: everycell finite-response full-logit ANDmargin relativeerror<=.10.
C: full/tangent value effects>1e-8. No fits,dosechange or adoption; nativecostpriced.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_finite_response_prediction pred_c_live_effects
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_mlp4_value_lineage_native_v1 as R
import value_response_jvp as J
from circuit_fast_screen_managed_runner import atomic_create_json

P=R.P;N=R.N;C=R.C;E=R.E;POLY=R.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_MLP4_VALUE_TANGENT_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP4_VALUE_TANGENT_V1_RESULT.json'
FILES={'prior':PRIOR,'jvp':Path(J.__file__),'parent_runner':Path(R.__file__),'parent_result':R.OUT}
EXPECTED={'prior': '29596f415d1dd204f374ff1e5a02516485e5313f2fb2f8c51e5d082eaa2663d0', 'jvp': '4e2cceeee9894ec2b496df4608e680afc1067171b0f2c6d76f4e738c0397d076', 'parent_runner': '7919d9bbc758d037cf212027175d88aa4e942fb773f90c8ca1fcfeea03331f24', 'parent_result': '1c8c41842c41f1975386cc41b5327effe0d85d52e7a2b2ed4fb007b4e45eb80d'}


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in R.FILES.items()}==R.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(E.S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    controls=J.controls();assert controls['passed']
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':48,'sequence_evaluations':864,'jvp_evaluations':8,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;attn=model.transformer.h[9].attn;mlp=model.transformer.h[4].mlp
    selected=lambda value:value.view(*value.shape[:2],9,128)[:,:,[1,4]]
    logits=lambda out,rows:P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double()
    parent=json.loads(R.OUT.read_text());counts=[0,0];audits=[];reports={};unselected=True;finite=True;jvps=0
    def counter(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=attn.register_forward_hook(counter)
    # Forward-mode differentiation is supported under no_grad, but NOT inference_mode.
    with torch.no_grad():
        try:
            for panel,rows in splits.items():
                outputs={k:[] for k in ('native','full','tangent','identity')};native=[];fields={k:[] for k in ('full','tangent','identity')};local=[]
                lengths=[len(r['base_ids']) for r in rows];assert all(len(r['donor_ids'])==n for r,n in zip(rows,lengths))
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side)
                    with C.capture(model) as record:out=backend.native(batch,capture=True)
                    native.append(record);outputs['native'].append(logits(out,rows))
                for i,side in enumerate(('base','donor')):
                    batch=P.das._batch(backend,rows,side=side)
                    with C.replace_mlp(mlp,native[1-i]['mlp_output'],lengths):
                        with C.capture(model) as record:backend.native(batch,capture=True)
                    assert all(torch.equal(record['mlp_output'][j,:n],native[1-i]['mlp_output'][j,:n]) for j,n in enumerate(lengths))
                    fields['full'].append(selected(record['value']))
                    source=native[i]['mlp_output'].clone();delta=native[1-i]['mlp_output']-source
                    for j,n in enumerate(lengths):delta[j,n:]=0
                    primal,tangent=J.response(backend,batch,source,delta);jvps+=1
                    base_value=selected(native[i]['value']);audits.append({'panel':panel,'side':side,'kind':'jvp_primal_value',**P.agree(primal.double(),base_value.double())})
                    fields['tangent'].append(base_value+tangent);fields['identity'].append(base_value)
                    exact=fields['full'][-1].double()-base_value.double();pred=tangent.double();valid=lambda x:torch.cat([x[j,:n] for j,n in enumerate(lengths)])
                    dn=float(valid(exact).norm());local.append({'side':side,'local_value_relative_error':float(valid(pred-exact).norm())/max(dn,1e-30),'local_value_signed_projection':float((valid(pred)*valid(exact)).sum())/max(dn**2,1e-30)})
                for arm in ('full','tangent','identity'):
                    for i,side in enumerate(('base','donor')):
                        batch=P.das._batch(backend,rows,side=side);mask=[]
                        with C.replace_value(attn,fields[arm][i],lengths,audit=mask):out=backend.native(batch,capture=True)
                        assert len(mask)==1;unselected=unselected and all(mask);outputs[arm].append(logits(out,rows))
                audits.extend({'panel':panel,'kind':'identity_full_logits',**P.agree(a,b)} for a,b in zip(outputs['identity'],outputs['native']))
                finite=finite and all(bool(z.isfinite().all()) for zs in outputs.values() for z in zs)
                ix=torch.arange(len(rows),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rows],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rows],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                cells={}
                for i,side in enumerate(('base','donor')):
                    full=center(outputs['full'][i])-center(outputs['native'][i]);pred=center(outputs['tangent'][i])-center(outputs['native'][i])
                    mf=margin(outputs['full'][i])-margin(outputs['native'][i]);mp=margin(outputs['tangent'][i])-margin(outputs['native'][i]);fn=float(full.norm());mn=float(mf.norm())
                    error=float((pred-full).norm());merror=float((mp-mf).norm())
                    previous=parent['reports'][panel]['cells'][side]['arms']['value_full']
                    audits.append({'panel':panel,'side':side,'kind':'parent_value_margin',**P.agree(mf,torch.tensor(previous['margin_effects'],device='cuda',dtype=torch.float64))})
                    norm_error=abs(fn-previous['effect_norm']);audits.append({'panel':panel,'side':side,'kind':'parent_value_effect_norm','max_abs':norm_error,'passed':norm_error<=1e-5*max(previous['effect_norm'],1e-8)})
                    cells[side]={'relative_effect_error':error/fn if fn>1e-8 else None,'relative_margin_error':merror/mn if mn>1e-8 else None,'signed_effect_projection':float((pred*full).sum())/max(fn**2,1e-30),'signed_margin_projection':float((mp*mf).sum())/max(mn**2,1e-30),'full_effect_norm':fn,'predicted_effect_norm':float(pred.norm()),'full_margin_effects':mf.cpu().tolist(),'predicted_margin_effects':mp.cpu().tolist(),'passed':(error<=.10*fn if fn>1e-8 else error<=1e-8) and (merror<=.10*mn if mn>1e-8 else merror<=1e-8)}
                reports[panel]={'cells':cells,'local_value_errors':local,'row_ids':[r['row_id'] for r in rows]}
                print(json.dumps({'panel':panel,'cells':cells,'local':local,'counts':counts}),flush=True)
        finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    a=finite and unselected and restored and counts==[48,864] and jvps==8 and all(v['passed'] for v in audits)
    cells=[c for r in reports.values() for c in r['cells'].values()]
    predictions={'pred_a_instrument':a,'pred_b_finite_response_prediction':all(c['passed'] for c in cells),'pred_c_live_effects':all(c['full_effect_norm']>1e-8 and c['predicted_effect_norm']>1e-8 for c in cells)}
    result={'terminal':'invalid' if not a else 'value_tangent_complete','predictions':predictions,'reports':reports,'audits':audits,'controls':controls,'hooks_restored':restored,'unselected_values_unchanged':unselected,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'forward_mode_jvp_evaluations':jvps,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0,'adoption':False},'scope':'Receiver-only derivative, paired MLP4 source change, native value-reader background. Existing partialpath on opened rows, no newtext OOD/extraction/structural-reduction claim.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
