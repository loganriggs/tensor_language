"""Static bias-preserving weight removals of the two MLP1 reader quadratics.

A: hashes/count16/restoration, FP64 weight-output1e-8/1e-9, coefficient identities
1e-8, deployed output-subtraction1e-3/1e-5, native parent/input replay.
B: own abs signed retention<=.1 and RMS<=.25; other contrast error<=.1 and no flips.
C: joint suppression both tasks, all P meanKL<=.001,p99<=.01 and zero endpoint flips.
16 forwards/512 sequences; no fitting or persistent weight updates, all weights priced.
No dose/rank/bias-offset/site rescue. This tests a homogeneous weight-defined removal.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selective_single_removals pred_c_joint_and_controls
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_mlp1_dual_reader_native_v1 as P
import quadratic_reader_weight_removal as M
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();POLY=P.POLY;N=P.N;V=P.V;atlas=P.atlas;das=P.das
PRIOR=POLY/'BILIN18_MLP1_WEIGHT_DEFINED_REMOVAL_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP1_WEIGHT_DEFINED_REMOVAL_V1_RESULT.json'
FILES={'prior':PRIOR,'primitive':Path(M.__file__),'parent_result':P.OUT,'parent_runner':Path(P.__file__)}
EXPECTED={'prior': '8ceeea9fdd8801bd62477fa35b2773c6294150cfce0ef3d8b7c7465f9bb9f888', 'primitive': '519de1ac4a8419ff0f220f1c711ff7b2560819fdc5d73edfe2c63eabf78d31ce', 'parent_result': '516ae48ba2e4e4a3f3f6e5b735db974aa7f9bc29c534180bf7760cc0beffcba5', 'parent_runner': '12f00093193f40dbc5c141f5208fe71189e0e911ccb6c397c241a39e7038001f'}
ARMS={'native':[],'remove_A':list(range(4)),'remove_B':list(range(4,8)),'remove_AB':list(range(8))}


def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(v) for k,v in P.FILES.items()}==P.EXPECTED
    authorities=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in authorities}==authorities
    manifest=json.loads(P.ROWS.read_text());parent=json.loads(P.OUT.read_text());assert parent['predictions']['pred_a_instrument']
    rows={k:manifest[k] for k in ('targets','controls')}
    assert {k:P.value_sha(v) for k,v in rows.items()}==manifest['row_sha256']
    assert len(rows['targets'])==48 and len(rows['controls'])==16
    dry={'dryrun':True,'model_loaded':False,'gpu_accessed':False,'model_forwards':16,'sequence_evaluations':512,'fit_updates':0,'model_updates':0,'arms':list(ARMS)}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    module=backend.model.transformer.h[1].mlp;assert not backend.model.config.gated
    saved=json.loads(N.READERS.read_text());c=torch.cat([torch.tensor(saved['physical_readers'][t],device='cuda',dtype=torch.float64).T for t in N.TASKS])
    d=torch.linalg.solve(c@c.T,c).T;original=module.Down.weight.detach().clone();w=original.double();bias=module.Down_bias.detach().clone();b=bias.double()
    left=module.Left.weight.detach().double();right=module.Right.weight.detach().double()
    controls=M.controls();assert controls['passed']
    edited={arm:M.remove_from_down(w,c,d,ids) for arm,ids in ARMS.items() if ids}
    algebra=[]
    for arm,ids in ARMS.items():
        if not ids:continue
        held=[i for i in range(8) if i not in ids]
        algebra.append({'kind':'removed_weight_reader','arm':arm,'max_abs':float((c[ids]@edited[arm]).abs().max())})
        if held:algebra.append({'kind':'opposite_weight_reader','arm':arm,'max_abs':float((c[held]@edited[arm]-c[held]@w).abs().max())})
    for first,last in [('remove_A','remove_B'),('remove_B','remove_A')]:
        sequential=M.remove_from_down(edited[first],c,d,ARMS[last])
        algebra.append({'kind':'commuting_weight_edits','arm':first+'/'+last,'max_abs':float((sequential-edited['remove_AB']).abs().max())})
    count=[0];sequences=[0]
    def counted(_m,args,_out):count[0]+=1;sequences[0]+=int(args[0].shape[0])
    counter=module.register_forward_hook(counted);reference={};outputs={};audits=[];input_equal=True;restored=True;finite=True
    with torch.inference_mode():
        try:
            for arm,ids in ARMS.items():
                outputs[arm]={}
                with nullcontext() if arm=='native' else M.at_down(module,edited[arm]):
                    for population,rs in rows.items():
                        inputs=[]
                        def capture_input(_m,args):inputs.append(args[0].detach().clone())
                        handle=module.register_forward_pre_hook(capture_input)
                        try:batch,bo,bc,do,dc=V.cap(backend,rs)
                        finally:handle.remove()
                        assert len(inputs)==2;stops=[int(x)+1 for x in batch.semantic_positions]
                        outputs[arm][population]=[]
                        for side,(native_out,cache,n) in enumerate(zip((bo,do),(bc,dc),inputs)):
                            y=P.flatten_valid(cache['mlp'][1].to('cuda'),stops)
                            if arm=='native':
                                nv=P.flatten_valid(n,stops).double();phi=(nv@left.T)*(nv@right.T)
                                reference[population,side]={'input':n,'native_output':y.double(),'phi':phi,'ideal_output':phi@w.T+b}
                            else:
                                ref=reference[population,side];input_equal=input_equal and torch.equal(n,ref['input'])
                                expected=ref['ideal_output']-((ref['ideal_output']-b)@c[ids].T)@d[:,ids].T
                                compiled=ref['phi']@edited[arm].T+b
                                audits.append({'kind':'fp64_weight_output','arm':arm,'population':population,'side':side,**P.agree(compiled,expected,1e-8,1e-9)})
                                deployed_expected=ref['native_output']-((ref['native_output']-b)@c[ids].T)@d[:,ids].T
                                audits.append({'kind':'deployed_weight_output','arm':arm,'population':population,'side':side,**P.agree(y.double(),deployed_expected,1e-3,1e-5)})
                            state=atlas.states(torch,backend,native_out,rs);logits=das.head_logits(backend,state).double()
                            finite=finite and bool(logits.isfinite().all());outputs[arm][population].append(logits)
                restored=restored and torch.equal(module.Down.weight,original) and torch.equal(module.Down_bias,bias)
                print(json.dumps({'arm':arm,'forwards':count[0],'weights_restored':restored}),flush=True)
        finally:counter.remove()
    reports={}
    for population,rs in rows.items():
        baseline=outputs['native'][population];ix=torch.arange(len(rs),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rs],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rs],device='cuda')
        native_margins=[z[ix,ans]-z[ix,foil] for z in baseline];native_contrast=native_margins[1]-native_margins[0]
        old_rows=parent['reports'][population]['rows'];previous=torch.tensor([r['native_margin_change'] for r in old_rows],device='cuda',dtype=torch.float64)
        audits.append({'kind':'parent_native_contrast','population':population,**P.agree(native_contrast,previous)})
        for side,key in enumerate(('native_base_top1','native_donor_top1')):
            assert baseline[side].argmax(-1).tolist()==[r[key] for r in old_rows]
        masks={'ALL':torch.ones(len(rs),device='cuda',dtype=torch.bool)}
        if population=='targets':masks.update({'temporal':torch.arange(48,device='cuda')<24,'iswas':torch.arange(48,device='cuda')>=24})
        reports[population]={}
        for arm in ARMS:
            logits=outputs[arm][population];margins=[z[ix,ans]-z[ix,foil] for z in logits];contrast=margins[1]-margins[0]
            kl=torch.stack([(base.log_softmax(-1).exp()*(base.log_softmax(-1)-edit.log_softmax(-1))).sum(-1) for base,edit in zip(baseline,logits)])
            flips=torch.stack([base.argmax(-1)!=edit.argmax(-1) for base,edit in zip(baseline,logits)])
            panels={}
            for group,mask in masks.items():
                target=native_contrast[mask];value=contrast[mask];den=float(target.square().sum());ks=kl[:,mask].flatten()
                panels[group]={'signed_contrast_retention':float(value@target)/den if den else None,
                    'contrast_rms_retention':float(value.norm())/float(target.norm()) if den else None,
                    'contrast_relative_error':float((value-target).norm())/float(target.norm()) if den else None,
                    'endpoint_teacher_kl_mean':float(ks.mean()),'endpoint_teacher_kl_p99':float(torch.quantile(ks,.99)),
                    'endpoint_teacher_kl_max':float(ks.max()),'endpoint_top1_changes':int(flips[:,mask].sum()),'pair_count':int(mask.sum())}
            reports[population][arm]={'panels':panels,'rows':[{'row_id':r['row_id'],'native_contrast':float(native_contrast[i]),'contrast':float(contrast[i]),
                'endpoint_kl':[float(kl[s,i]) for s in range(2)],'endpoint_top1_changes':[bool(flips[s,i]) for s in range(2)]} for i,r in enumerate(rs)]}
    def suppressed(cell):return cell['signed_contrast_retention'] is not None and abs(cell['signed_contrast_retention'])<=.1 and cell['contrast_rms_retention']<=.25
    decisions={};bpass=True
    for arm,own,other in [('remove_A','temporal','iswas'),('remove_B','iswas','temporal')]:
        owncell=reports['targets'][arm]['panels'][own];othercell=reports['targets'][arm]['panels'][other]
        own_ok=suppressed(owncell);other_ok=othercell['contrast_relative_error']<=.1 and othercell['endpoint_top1_changes']==0
        decisions[arm]={'own_suppressed':own_ok,'other_preserved':other_ok};bpass=bpass and own_ok and other_ok
    joint=all(suppressed(reports['targets']['remove_AB']['panels'][t]) for t in N.TASKS)
    ppass=all(v['panels']['ALL']['endpoint_teacher_kl_mean']<=.001 and v['panels']['ALL']['endpoint_teacher_kl_p99']<=.01 and v['panels']['ALL']['endpoint_top1_changes']==0 for arm,v in reports['controls'].items() if arm!='native')
    a=finite and restored and input_equal and count[0]==16 and sequences[0]==512 and all(z['passed'] for z in audits) and all(z['max_abs']<=1e-8 for z in algebra)
    result={'terminal':'invalid' if not a else 'weight_defined_removal_screen_pass' if bpass and joint and ppass else 'task_reader_quadratics_not_independently_removable',
        'predictions':{'pred_a_instrument':a,'pred_b_selective_single_removals':bpass,'pred_c_joint_and_controls':joint and ppass},
        'decisions':decisions,'joint_suppression':joint,'p_controls_preserved':ppass,'audits':audits,'weight_algebra':algebra,'controls':controls,
        'inputs_unchanged':input_equal,'original_weights_and_bias_restored':restored,'reports':reports,
        'authority_sha256':observed,'runner_sha256':N.sha(RUNNER),
        'price':{'model_forwards':count[0],'sequence_evaluations':sequences[0],'fit_updates':0,'model_updates':0,'temporary_weight_intervention_arms':3,
                 'temporary_edited_weight_scalars':sum(v.numel() for v in edited.values()),'extra_fp64_weight_output_evaluations':12,
                 'native_parameters':sum(p.numel() for p in backend.model.parameters()),'adoption':False},
        'scope':'Fixed homogeneous quadratic reader components, bias unchanged; no fitted offset, donor-reference dependence, or smaller-model claim.',
        'wall_seconds':time.perf_counter()-tic}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','decisions','joint_suppression','p_controls_preserved','price','wall_seconds')},indent=2))
    print(json.dumps({p:{arm:v['panels'] for arm,v in arms.items()} for p,arms in reports.items()},indent=2))

if __name__=='__main__':main()
