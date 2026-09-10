"""Direct residual-lineage folding versus full MLP4-induced localV9 change.
A: hashes/recurrence/oracles1e-3/1e-5,FP64fold1e-8/1e-9,48/864 counts.
B:value_full signedprojection>=.10 both full-logit/margin allcells.
C:directfold relative effect error<=.10 bothframes allcells; D:editslive.
All weights retained, no fit or adoption; norm inputs and full background priced.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_value_carrier pred_c_direct_lineage_sufficient pred_d_live_edits
import json
import math
import os
from pathlib import Path
import signal
import time
import run_bilin18_l9_query_partition_native_v1 as Q
import value_lineage_capture as C
import mlp_value_lineage_fold as F
from circuit_fast_screen_managed_runner import atomic_create_json

P=Q.P;N=Q.N;E=Q.E;POLY=Q.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_MLP4_VALUE_LINEAGE_NATIVE_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP4_VALUE_LINEAGE_NATIVE_V1_RESULT.json'
FILES={'prior':PRIOR,'capture':Path(C.__file__),'fold':Path(F.__file__),'parent_runner':Path(Q.__file__),'parent_result':Q.OUT}
EXPECTED={'prior': '26959e78b77a19efe1599c9db3fe94100471ef1c7851764f231ae769e735736c', 'capture': 'd2f9f35d244ba951e76a0f49b358c32afcdfe533733d6a4fbf89cc3f8dc076ea', 'fold': '1037515db1a43149f49031a864230a12710196244deba948c75deede212b78f4', 'parent_runner': '2305f1f4f45daea71f8238ef7e43693fb529259ca47dae76efd384041e4142ab', 'parent_result': '92da8b2ab6a580953e45feece9ca0d5694c2c090293ab1c0493ee70b3ff60449'}


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in Q.FILES.items()}==Q.EXPECTED
    assert {k:N.sha(p) for k,p in Q.A.FILES.items()}==Q.A.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(E.S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    assert all(len(r['base_ids'])==len(r['donor_ids']) and r['base_semantic_position']==r['donor_semantic_position']==len(r['base_ids'])-1 for rows in splits.values() for r in rows)
    controls={'capture':C.controls(),'fold':F.controls()};assert all(v['passed'] for v in controls.values())
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':48,'sequence_evaluations':864,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    attn=backend.model.transformer.h[9].attn;mlp=backend.model.transformer.h[4].mlp;eps=torch.finfo(torch.float32).eps
    reader=torch.cat([attn.c_v.weight[h*128:(h+1)*128].detach().double() for h in (1,4)])
    left=mlp.Left.weight.detach().double();right=mlp.Right.weight.detach().double();down=mlp.Down.weight.detach().double();folded_weight=reader@down
    gamma=math.prod(float(b.lambdas[0]) for b in backend.model.transformer.h[5:10])
    counts=[0,0];audits=[];reports={};recurrence=True;unselected=True;finite=True
    def counter(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=attn.register_forward_hook(counter);parent=json.loads(Q.OUT.read_text())
    selected=lambda value:value.view(*value.shape[:2],9,128)[:,:,[1,4]]
    logits=lambda out,rows:P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double()
    with torch.inference_mode():
        try:
            for panel,rows in splits.items():
                outputs={k:[] for k in ('native','full_mlp4','value_full','direct_fold','value_identity','raw_direct')};native=[];hybrid=[]
                lengths=[len(r['base_ids']) for r in rows]
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side)
                    with C.capture(backend.model) as r:out=backend.native(batch,capture=True)
                    native.append(r);outputs['native'].append(logits(out,rows));recurrence=recurrence and r['recurrence_bitwise']
                for i,side in enumerate(('base','donor')):
                    batch=P.das._batch(backend,rows,side=side)
                    with C.replace_mlp(mlp,native[1-i]['mlp_output'],lengths):
                        with C.capture(backend.model) as r:out=backend.native(batch,capture=True)
                    hybrid.append(r);outputs['full_mlp4'].append(logits(out,rows));recurrence=recurrence and r['recurrence_bitwise']
                    assert all(torch.equal(r['mlp_output'][j,:n],native[1-i]['mlp_output'][j,:n]) for j,n in enumerate(lengths))
                fields=[];local_scales=[]
                for i,side in enumerate(('base','donor')):
                    receiving=native[i];donor=native[1-i];u=receiving['residual'].double()
                    mask=torch.zeros_like(u[:,:,:1])
                    for j,n in enumerate(lengths):mask[j,:n]=1
                    delta=(donor['mlp_output'].double()-receiving['mlp_output'].double())*mask
                    parts=F.value_change(u,delta,gamma,reader,eps)
                    nb=receiving['mlp_input'].double();nd=donor['mlp_input'].double();mid=(nb+nd)/2;dn=nd-nb
                    feature_delta=((mid@left.T)*(dn@right.T)+(dn@left.T)*(mid@right.T))*mask
                    folded_content=gamma*(feature_delta@folded_weight.T)/parts['new_scale']
                    mlp64_delta=(((nd@left.T)*(nd@right.T)-(nb@left.T)*(nb@right.T))*mask)@down.T
                    oracle=gamma*(mlp64_delta@reader.T)/parts['new_scale']
                    audits.append({'panel':panel,'side':side,'kind':'folded_content_fp64',**P.agree(folded_content,oracle,atol=1e-8,rtol=1e-9)})
                    base_value=selected(receiving['value']).double()
                    folded=base_value+(folded_content+parts['normalization']).view_as(base_value)
                    raw=(torch.nn.functional.rms_norm(u+gamma*delta,(1152,),eps=eps)@reader.T).view_as(base_value)
                    audits.append({'panel':panel,'side':side,'kind':'fold_vs_direct_value',**P.agree(folded,raw)})
                    fields.append({'value_full':selected(hybrid[i]['value']),'direct_fold':folded,'value_identity':base_value,'raw_direct':raw})
                    local_scales.append({'side':side,'direct_residual_change_norm':float((gamma*delta).norm()),'actual_residual_change_norm':float((hybrid[i]['residual']-receiving['residual']).norm()),'native_value_change_norm':float((selected(hybrid[i]['value']).double()-base_value).norm()),'direct_value_change_norm':float((folded-base_value).norm()),'normalization_value_term_norm':float(parts['normalization'].norm())})
                for arm in ('value_full','direct_fold','value_identity','raw_direct'):
                    for i,side in enumerate(('base','donor')):
                        batch=P.das._batch(backend,rows,side=side);mask_audit=[]
                        with C.replace_value(attn,fields[i][arm],lengths,audit=mask_audit):out=backend.native(batch,capture=True)
                        assert len(mask_audit)==1;unselected=unselected and all(mask_audit)
                        outputs[arm].append(logits(out,rows))
                for candidate,reference in (('value_identity','native'),('direct_fold','raw_direct')):
                    audits.extend({'panel':panel,'kind':candidate+'_full_logits',**P.agree(a,b)} for a,b in zip(outputs[candidate],outputs[reference]))
                finite=finite and all(bool(z.isfinite().all()) for zs in outputs.values() for z in zs)
                ix=torch.arange(len(rows),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rows],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rows],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                target=center(outputs['native'][1])-center(outputs['native'][0]);mtarget=margin(outputs['native'][1])-margin(outputs['native'][0]);cells={}
                audits.append({'panel':panel,'kind':'parent_native_contrast',**P.agree(mtarget,torch.tensor(parent['reports'][panel]['native_margin_contrasts'],device='cuda',dtype=torch.float64))})
                for i,side in enumerate(('base','donor')):
                    effects={k:center(v[i])-center(outputs['native'][i]) for k,v in outputs.items() if k!='native'}
                    meffects={k:margin(v[i])-margin(outputs['native'][i]) for k,v in outputs.items() if k!='native'};sign=1 if i==0 else -1
                    projection=lambda a,b:float((a*b).sum())*sign/max(float(b.square().sum()),1e-30)
                    ref=effects['value_full'];mref=meffects['value_full'];rn=float(ref.norm());mn=float(mref.norm())
                    error=float((effects['direct_fold']-ref).norm());merror=float((meffects['direct_fold']-mref).norm())
                    arm_report={arm:{'full_logit_signed_projection':projection(effects[arm],target),'margin_signed_projection':projection(meffects[arm],mtarget),'effect_norm':float(effects[arm].norm()),'margin_effects':meffects[arm].cpu().tolist()} for arm in ('full_mlp4','value_full','direct_fold')}
                    b=all(arm_report['value_full'][k]>=.10 for k in ('full_logit_signed_projection','margin_signed_projection')) and float(target.norm())>1e-8 and float(mtarget.norm())>1e-8
                    c=(error<=.10*rn if rn>1e-8 else error<=1e-8) and (merror<=.10*mn if mn>1e-8 else merror<=1e-8)
                    cells[side]={'arms':arm_report,'direct_full_logit_relative_error':error/rn if rn>1e-8 else None,'direct_margin_relative_error':merror/mn if mn>1e-8 else None,'direct_vs_full_value_signed_projection':float((effects['direct_fold']*ref).sum())/max(rn**2,1e-30),'carrier_pass':b,'direct_lineage_pass':c}
                reports[panel]={'cells':cells,'local_changes':local_scales,'native_margin_contrasts':mtarget.cpu().tolist(),'row_ids':[r['row_id'] for r in rows]}
                print(json.dumps({'panel':panel,'cells':cells,'counts':counts}),flush=True)
        finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=finite and recurrence and unselected and restored and counts==[48,864] and all(v['passed'] for v in audits)
    cells=[c for r in reports.values() for c in r['cells'].values()]
    predictions={'pred_a_instrument':a,'pred_b_value_carrier':all(c['carrier_pass'] for c in cells),'pred_c_direct_lineage_sufficient':all(c['direct_lineage_pass'] for c in cells),'pred_d_live_edits':all(c['arms'][arm]['effect_norm']>1e-8 for c in cells for arm in ('value_full','direct_fold'))}
    result={'terminal':'invalid' if not a else 'mlp4_value_lineage_complete','predictions':predictions,'gamma4':gamma,'reports':reports,'audits':audits,'controls':controls,'recurrence_bitwise':recurrence,'unselected_values_unchanged':unselected,'hooks_restored':restored,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'native_parameters':sum(p.numel() for p in backend.model.parameters()),'folded_value_down_entries':folded_weight.numel(),'actual_weight_saving':0,'adoption':False},'scope':'Opened rows, all valid MLP4 token outputs swapped, selected H1/H4 localV9 only. Direct norm still uses native full MLP output delta/residual; all initialization and opaque weights charged.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','gamma4','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
