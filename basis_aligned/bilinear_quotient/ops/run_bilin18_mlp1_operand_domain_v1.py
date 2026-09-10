"""MLP1 symmetric-function extension versus native independent operand edits.

A: exact16/512 counts/hashes, hook/input restoration, FP64 algebra1e-8/1e-9,
native local/replay1e-3/1e-5, parent baseline. B: each L/R reference and temporal,
iswas/P panel KLmean<=.001,p99<=.01,zero top1 disagreement. C: centered full-logit
causal-vector relative error<=.01 (zero-norm absolute1e-8). No fit/update/saving.
Null restricts the intervention API, not tied-input symmetric execution.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_distribution_prediction pred_c_causal_prediction
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_mlp1_dual_reader_native_v1 as P
import bilinear_operand_intervention as M
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();N=P.N;POLY=P.POLY
PRIOR=POLY/'BILIN18_MLP1_OPERAND_DOMAIN_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP1_OPERAND_DOMAIN_V1_RESULT.json'
FILES={'prior':PRIOR,'primitive':Path(M.__file__),'parent_runner':Path(P.__file__),'parent_result':P.OUT}
EXPECTED={'prior': '52386c84a2fc50577fda7902a6dc7f91fa762b7c4be7df7e6af60a86be5dd4a6', 'primitive': 'd7b84b55c7e6b3acf75e149c253e58d24110702033f6a4a0be3c117d3c21fa56', 'parent_runner': '12f00093193f40dbc5c141f5208fe71189e0e911ccb6c397c241a39e7038001f', 'parent_result': '516ae48ba2e4e4a3f3f6e5b735db974aa7f9bc29c534180bf7760cc0beffcba5'}
ARMS=('zero','left','right','both','donor_output','symmetric')


def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(v) for k,v in P.FILES.items()}==P.EXPECTED
    authorities=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in authorities}==authorities
    manifest=json.loads(P.ROWS.read_text());rows={k:manifest[k] for k in ('targets','controls')}
    assert {k:P.value_sha(v) for k,v in rows.items()}==manifest['row_sha256']
    assert len(rows['targets'])==48 and len(rows['controls'])==16
    parent=json.loads(P.OUT.read_text());assert parent['predictions']['pred_a_instrument']
    controls=M.controls();assert controls['passed']
    dry={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':16,'sequence_evaluations':512,'arms':ARMS,'controls':controls}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    module=backend.model.transformer.h[1].mlp;assert not backend.model.config.gated
    count=[0,0];audits=[];reports={};input_equal=True;finite=True
    def counted(_m,args,_out):count[0]+=1;count[1]+=args[0].shape[0]
    counter=module.register_forward_hook(counted)
    with torch.inference_mode():
        try:
            for population,rs in rows.items():
                captures={k:[] for k in ('input','left','right')};handles=[]
                handles.append(module.register_forward_pre_hook(lambda _m,args:captures['input'].append(args[0].detach().clone())))
                for key,target in [('left',module.Left),('right',module.Right)]:
                    def capture(_m,_args,out,key=key):captures[key].append(out.detach().clone())
                    handles.append(target.register_forward_hook(capture))
                try:batch,bo,bc,do,dc=P.V.cap(backend,rs)
                finally:
                    for handle in handles:handle.remove()
                assert all(len(v)==2 for v in captures.values())
                stops=[int(v)+1 for v in batch.semantic_positions]
                lb,ld=captures['left'];rb,rd=captures['right']
                yb=bc['mlp'][1].to('cuda');yd=dc['mlp'][1].to('cuda')
                left_native=module.Down(ld*rb)+module.Down_bias
                right_native=module.Down(lb*rd)+module.Down_bias
                symmetric=(left_native+right_native)/2
                w=module.Down.weight.detach().double();bias=module.Down_bias.detach().double()
                left64=(ld.double()*rb.double())@w.T+bias
                right64=(lb.double()*rd.double())@w.T+bias
                sym64=(left64+right64)/2;skew64=(left64-right64)/2
                flat=lambda x:P.flatten_valid(x,stops)
                audits.extend([{'population':population,'kind':'fp64_left',**P.agree(sym64+skew64,left64,1e-8,1e-9)},
                    {'population':population,'kind':'fp64_right',**P.agree(sym64-skew64,right64,1e-8,1e-9)}])
                logits={}
                for name,out in [('base',bo),('donor',do)]:
                    logits[name]=P.das.head_logits(backend,P.atlas.states(torch,backend,out,rs)).double()
                specs={'zero':{'output':yb},'left':{'left':ld},'right':{'right':rd},'both':{'left':ld,'right':rd},
                       'donor_output':{'output':yd},'symmetric':{'output':symmetric}}
                local_reference={'zero':yb.double(),'left':left64,'right':right64,'both':yd.double(),'donor_output':yd.double(),'symmetric':sym64}
                for arm in ARMS:
                    current=[]
                    with M.scoped(module,stops=stops,**specs[arm]):
                        def capture_edited(_m,args,out):current.append((args[0].detach().clone(),out.detach().clone()))
                        h=module.register_forward_hook(capture_edited)
                        try:out=backend.native(batch,capture=True)
                        finally:h.remove()
                    assert len(current)==1
                    current_input,current_output=current[0]
                    input_equal=input_equal and torch.equal(current_input,captures['input'][0])
                    audits.append({'population':population,'kind':'native_local','arm':arm,**P.agree(flat(current_output),flat(local_reference[arm]))})
                    logits[arm]=P.das.head_logits(backend,P.atlas.states(torch,backend,out,rs)).double()
                    assert len(module._forward_hooks)==1 and not module.Left._forward_hooks and not module.Right._forward_hooks
                audits.extend([{'population':population,'kind':'zero_replay',**P.agree(logits['zero'],logits['base'])},
                               {'population':population,'kind':'both_donor_output_replay',**P.agree(logits['both'],logits['donor_output'])}])
                finite=finite and all(bool(z.isfinite().all()) for z in logits.values())
                ix=torch.arange(len(rs),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rs],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rs],device='cuda')
                margins={k:z[ix,ans]-z[ix,foil] for k,z in logits.items()};contrast=margins['donor']-margins['base']
                previous=torch.tensor([r['native_margin_change'] for r in parent['reports'][population]['rows']],device='cuda',dtype=torch.float64)
                audits.append({'population':population,'kind':'parent_native_margin',**P.agree(contrast,previous)})
                masks={'ALL':torch.ones(len(rs),device='cuda',dtype=torch.bool)}
                if population=='targets':masks={'temporal':ix<24,'iswas':ix>=24}
                panels={}
                centered={k:z-z.mean(-1,keepdim=True) for k,z in logits.items()}
                for group,mask in masks.items():
                    comparison={}
                    for reference in ('left','right'):
                        lp=logits[reference][mask].log_softmax(-1);candidate=logits['symmetric'][mask]
                        kl=(lp.exp()*(lp-candidate.log_softmax(-1))).sum(-1)
                        target=centered[reference][mask]-centered['base'][mask]
                        estimate=centered['symmetric'][mask]-centered['base'][mask]
                        den=float(target.norm());error=float((estimate-target).norm())
                        comparison[reference]={'mean_kl':float(kl.mean()),'p99_kl':float(torch.quantile(kl,.99)),
                            'max_kl':float(kl.max()),'top1_disagreements':int((candidate.argmax(-1)!=logits[reference][mask].argmax(-1)).sum()),
                            'causal_reference_norm':den,'causal_error_norm':error,'causal_relative_error':error/den if den>1e-8 else None,
                            'distribution_pass':bool(kl.mean()<=.001 and torch.quantile(kl,.99)<=.01 and torch.equal(candidate.argmax(-1),logits[reference][mask].argmax(-1))),
                            'causal_pass':error<=.01*den if den>1e-8 else error<=1e-8}
                    den=float(contrast[mask].square().sum())
                    panels[group]={'symmetric_vs_native_operand':comparison,'native_margin_projection':{k:float((margins[k][mask]-margins['base'][mask])@contrast[mask])/den if den else None for k in ARMS},'count':int(mask.sum())}
                vf=flat(skew64);sf=flat(sym64-bias)
                reports[population]={'panels':panels,'local_skew_rms':float(vf.square().mean().sqrt()),'local_skew_to_symmetric_norm':float(vf.norm()/sf.norm()),
                    'rows':[{'row_id':r['row_id'],'native_margin_change':float(contrast[i]),'arm_margin_change':{k:float(margins[k][i]-margins['base'][i]) for k in ARMS}} for i,r in enumerate(rs)]}
                print(json.dumps({'population':population,'panels':panels,'forwards':count}),flush=True)
        finally:counter.remove()
    restored=not module._forward_hooks and not module.Left._forward_hooks and not module.Right._forward_hooks
    cells=[v for report in reports.values() for panel in report['panels'].values() for v in panel['symmetric_vs_native_operand'].values()]
    a=bool(finite and restored and input_equal and count==[16,512] and all(v['passed'] for v in audits))
    b=all(v['distribution_pass'] for v in cells);c=all(v['causal_pass'] for v in cells)
    result={'terminal':'invalid' if not a else 'symmetric_operand_extension_screen_pass' if b and c else 'symmetric_function_loses_native_operand_interventions',
        'predictions':{'pred_a_instrument':a,'pred_b_distribution_prediction':b,'pred_c_causal_prediction':c},
        'audits':audits,'controls':controls,'reports':reports,'inputs_unchanged':input_equal,'hooks_restored':bool(restored),
        'authority_sha256':observed,'runner_sha256':N.sha(RUNNER),
        'price':{'model_forwards':count[0],'sequence_evaluations':count[1],'model_updates':0,'fit_updates':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'adoption':False},
        'scope':'Opened paired contexts; explicit independent operand interventions, not a rejection of tied-input symmetric rewrites or all possible intervention translations.',
        'wall_seconds':time.perf_counter()-tic}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
