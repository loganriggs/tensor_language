"""Fixed attention-times-pre-attention numerator test for both MLP1 task readers.

A: hashes/count19, FP64 algebra1e-8/1e-9, norm1e-4/1e-6,
FP32 reader/logit/parent replay1e-3/1e-5, controls/hooks. B: cross signed effect
>=90% full with effect-vector relative error<=.15 both tasks; live full>=.01.
C: original ordinary P mean KL+1e-6 and zero flips, both edit labels.
19 forwards/528 sequences, no fits/updates. All native weights/background charged.
No fallback component, independent-norm path, or attention-input knockout claim.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_cross_supplies_both_tasks pred_c_original_control_limit
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_mlp1_dual_reader_native_v1 as P
import mlp1_attention_cross_program as M
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();POLY=P.POLY;N=P.N;V=P.V;atlas=P.atlas;das=P.das
PRIOR=POLY/'BILIN18_MLP1_ATTENTION_CROSS_PROGRAM_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP1_ATTENTION_CROSS_PROGRAM_V1_RESULT.json'
FILES={'prior':PRIOR,'primitive':Path(M.__file__),'parent_result':P.OUT,'parent_runner':Path(P.__file__)}
EXPECTED={'prior': 'da9c88164f2f6a1ab73fddc5f85eae0b5e48e099c6186c67ac67e662f9d84bfc', 'primitive': '326bbf3d7547cfc8d51f4eabd48f6116e6c9bb5acbc08650a791133f5291ebae', 'parent_result': '516ae48ba2e4e4a3f3f6e5b735db974aa7f9bc29c534180bf7760cc0beffcba5', 'parent_runner': '12f00093193f40dbc5c141f5208fe71189e0e911ccb6c397c241a39e7038001f'}
ARMS=('full','pre','cross','attention','sum')


def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(v) for k,v in P.FILES.items()}==P.EXPECTED
    authorities=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in authorities}==authorities
    manifest=json.loads(P.ROWS.read_text());parent=json.loads(P.OUT.read_text())
    assert parent['predictions']['pred_a_instrument'] and parent['predictions']['pred_b_own_effect_retained']
    assert {k:P.value_sha(manifest[k]) for k in ('targets','controls')}==manifest['row_sha256']
    assert len(manifest['targets'])==48 and len(manifest['controls'])==16
    for r in manifest['targets']+manifest['controls']:
        assert len(r['base_ids'])==len(r['donor_ids']) and r['base_semantic_position']==r['donor_semantic_position']==len(r['base_ids'])-1
    dry={'dryrun':True,'model_loaded':False,'gpu_accessed':False,'model_forwards':19,'sequence_evaluations':528,
         'fit_updates':0,'model_updates':0,'arms':ARMS,'control_batch_size':16}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;module=model.transformer.h[1].mlp;assert not model.config.gated
    controls=M.controls();assert controls['passed']
    saved=json.loads(N.READERS.read_text());c=torch.cat([torch.tensor(saved['physical_readers'][t],device='cuda',dtype=torch.float64).T for t in N.TASKS])
    d=torch.linalg.solve(c@c.T,c).T;left=module.Left.weight.detach().double();right=module.Right.weight.detach().double()
    e=c@module.Down.weight.detach().double();bias=c@module.Down_bias.detach().double();eps=torch.finfo(torch.float32).eps
    count=[0];sequences=[0]
    def counted(_m,args,_out):count[0]+=1;sequences[0]+=int(args[0].shape[0])
    counter=module.register_forward_hook(counted);source={};audits=[];restored=True;finite=True;reports={}
    with torch.inference_mode():
        try:
            for name in ('targets','controls'):
                rows=manifest[name]
                with M.capture_native_streams(model) as frames:batch,bo,bc,do,dc=V.cap(backend,rows)
                assert len(frames)==2;stops=[int(p)+1 for p in batch.semantic_positions];pieces=[]
                for frame in frames:
                    u,a=frame['pre'].double(),frame['attention'].double()
                    n=(u+a)/((u+a).square().mean(-1,keepdim=True)+eps).sqrt()
                    parts=M.components(u,a,left,right,e,eps);pieces.append(parts)
                    direct=((n@left.T)*(n@right.T))@e.T+bias
                    summed=sum(parts.values())+bias
                    f=lambda z:P.flatten_valid(z,stops)
                    audits.append({'kind':'component_algebra','population':name,**P.agree(f(summed),f(direct),1e-8,1e-9)})
                    audits.append({'kind':'normalized_input','population':name,**P.agree(f(n),f(frame['normalized_input']),1e-4,1e-6)})
                    audits.append({'kind':'native_reader','population':name,**P.agree(f(summed),f(frame['output']).double()@c.T,1e-3,1e-5)})
                baselogits=das.head_logits(backend,atlas.states(torch,backend,bo,rows)).double()
                donorlogits=das.head_logits(backend,atlas.states(torch,backend,do,rows)).double()
                source[name]={'rows':rows,'batch':batch,'stops':stops,'donor_output':dc['mlp'][1].to('cuda'),
                    'delta':{term:pieces[1][term]-pieces[0][term] for term in ('pre','cross','attention')},
                    'base_logits':baselogits,'donor_logits':donorlogits}
            for name,key,labels in [('targets','targets',['A']*24+['B']*24),('control_A','controls',['A']*16),('control_B','controls',['B']*16)]:
                ctx=source[key];rs=ctx['rows'];stops=ctx['stops'];logits={'base':ctx['base_logits'],'donor':ctx['donor_logits']}
                for arm in ARMS:
                    values=sum(ctx['delta'].values()) if arm=='sum' else None if arm=='full' else ctx['delta'][arm]
                    def hook(_m,_args,output):
                        changed=output.clone()
                        for i,(stop,label) in enumerate(zip(stops,labels)):
                            selected=list(range(4)) if label=='A' else list(range(4,8))
                            if arm=='full':
                                changed[i:i+1]=P.edit.apply(output[i:i+1],ctx['donor_output'][i:i+1],c,d,selected,[stop])
                            else:
                                delta=values[i,:stop,selected]@d[:,selected].T
                                changed[i,:stop]=(output[i,:stop].double()+delta).to(output.dtype)
                        return changed
                    handle=module.register_forward_hook(hook)
                    try:out=backend.native(ctx['batch'],capture=True)
                    finally:handle.remove()
                    restored=restored and len(module._forward_hooks)==1
                    logits[arm]=das.head_logits(backend,atlas.states(torch,backend,out,rs)).double()
                audits.append({'kind':'full_vs_compiled','population':name,**P.agree(logits['sum'],logits['full'])})
                finite=finite and all(bool(x.isfinite().all()) for x in logits.values())
                ix=torch.arange(len(rs),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rs],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rs],device='cuda')
                margin={arm:x[ix,ans]-x[ix,foil] for arm,x in logits.items()}
                native=margin['donor']-margin['base'];delta={arm:margin[arm]-margin['base'] for arm in ARMS}
                lp=logits['base'].log_softmax(-1)
                kl={arm:(lp.exp()*(lp-logits[arm].log_softmax(-1))).sum(-1) for arm in ARMS}
                flips={arm:logits[arm].argmax(-1)!=logits['base'].argmax(-1) for arm in ARMS}
                masks={'ALL':torch.ones(len(rs),dtype=torch.bool,device='cuda')}
                if name=='targets':masks.update({'temporal':torch.arange(48,device='cuda')<24,'iswas':torch.arange(48,device='cuda')>=24})
                panels={}
                for group,mask in masks.items():
                    full=delta['full'][mask];nv=native[mask];den=float(nv.square().sum());records={}
                    for arm in ARMS:
                        v=delta[arm][mask];fullnorm=float(full.norm())
                        records[arm]={'signed_native_projection':float(v@nv)/den if den>0 else None,
                            'margin_change_rms':float(v.square().mean().sqrt()),'relative_effect_error_to_full':float((v-full).norm())/fullnorm if fullnorm>0 else None,
                            'mean_kl':float(kl[arm][mask].mean()),'max_kl':float(kl[arm][mask].max()),'top1_flips':int(flips[arm][mask].sum())}
                    panels[group]={'arms':records,'native_margin_change_rms':float(nv.square().mean().sqrt()),'rows':int(mask.sum())}
                # Compare the same registered per-task/P statistics to the parent.
                group_tasks=[('temporal','A'),('iswas','B')] if name=='targets' else [('ALL',name[-1])]
                for group,label in group_tasks:
                    previous=parent['reports']['targets']['panels'][group]['arms']['dual_'+label] if name=='targets' else parent['reports']['controls']['panels']['ALL']['arms']['dual_'+label]
                    for field in ('signed_native_projection','margin_change_rms','mean_kl'):
                        audits.append({'kind':'parent_'+field,'population':name+'/'+group,**P.agree(torch.tensor(panels[group]['arms']['full'][field],dtype=torch.float64),torch.tensor(previous[field],dtype=torch.float64))})
                    assert panels[group]['arms']['full']['top1_flips']==previous['top1_flips']
                reports[name]={'panels':panels,'rows':[{'row_id':r['row_id'],'edit_label':labels[i],'native_margin_change':float(native[i]),
                    'margin_changes':{arm:float(delta[arm][i]) for arm in ARMS},'kl':{arm:float(kl[arm][i]) for arm in ARMS},
                    'top1_flips':{arm:bool(flips[arm][i]) for arm in ARMS}} for i,r in enumerate(rs)]}
                print(json.dumps({'population':name,'panels':panels,'forwards':count[0]}),flush=True)
        finally:counter.remove()
    a=finite and restored and len(module._forward_hooks)==0 and count[0]==19 and sequences[0]==528 and all(z['passed'] for z in audits)
    decisions={};b=True;cpass=True
    for task in N.TASKS:
        arms=reports['targets']['panels'][task]['arms'];full=arms['full']['signed_native_projection'];cross=arms['cross']['signed_native_projection'];error=arms['cross']['relative_effect_error_to_full']
        passed=full is not None and cross is not None and full>=.01 and cross>=.9*full and error is not None and error<=.15
        decisions[task]={'full_signed_effect':full,'cross_signed_effect':cross,'cross_relative_effect_error':error,'passed':passed};b=b and passed
    for label in ('A','B'):
        candidate=reports['control_'+label]['panels']['ALL']['arms']['cross'];baseline=parent['reports']['controls']['panels']['ALL']['arms']['ordinary_'+label]
        passed=candidate['mean_kl']<=baseline['mean_kl']+1e-6 and candidate['top1_flips']==0
        decisions['control_'+label]={'cross_kl':candidate['mean_kl'],'ordinary_parent_kl':baseline['mean_kl'],'top1_flips':candidate['top1_flips'],'passed':passed};cpass=cpass and passed
    result={'terminal':'invalid' if not a else 'attention_cross_program_screen_pass' if b and cpass else 'attention_cross_program_null',
        'predictions':{'pred_a_instrument':a,'pred_b_cross_supplies_both_tasks':b,'pred_c_original_control_limit':cpass},
        'decisions':decisions,'audits':audits,'controls':controls,'reports':reports,'authority_sha256':observed,'runner_sha256':N.sha(RUNNER),
        'price':{'model_forwards':count[0],'sequence_evaluations':sequences[0],'model_updates':0,'fit_updates':0,
                 'native_parameters':sum(p.numel() for p in model.parameters()),'reader_scalars':c.numel(),'writer_scalars':d.numel(),'folded_reader_down_scalars':e.numel(),
                 'extra_stream_factor_maps':16,'component_down_contractions':12,'direct_oracle_contractions':12,'reader_down_weight_folds':1,'adoption':False},
        'scope':'Expanded numerator-output node, shared full normalization retained; not attention-input knockout or independently extracted circuit.',
        'wall_seconds':time.perf_counter()-tic}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','decisions','price','wall_seconds')},indent=2))

if __name__=='__main__':main()
