"""Native query-source interaction algebra and complete paired Q/Q2 swaps.
A: hashes/oracles1e-3/1e-5, localclosure1e-8/1e-9,24forwards/432seq.
B: everypanel/direction full-logit AND margin signed projection>=.10.
C: numerator-cross localinteraction error<=.10 all endpoint/paired panels.
D: query swaps live>1e-8. Native weights retained; no fits/adoption.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_complete_query_carrier pred_c_numerator_cross_explanation pred_d_live_query_swaps
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import run_bilin18_l9_query_source_atlas_v1 as A
import query_partition_norm_math_v1 as Q
from circuit_fast_screen_managed_runner import atomic_create_json

E=A.E;P=A.P;N=A.N;M=A.M;POLY=A.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_L9_QUERY_PARTITION_NATIVE_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_L9_QUERY_PARTITION_NATIVE_V1_RESULT.json'
BANK=POLY/'BILIN18_L9_QUERY_PARTITION_NATIVE_V1_BANK.npz'
FILES={'prior':PRIOR,'partition':Path(Q.__file__),'parent_runner':Path(A.__file__),'parent_result':A.OUT}
EXPECTED={'prior': 'dd7aaa63482789d99f3d420f508ff3e54b4bf196e9aab87add30df7a275e5031', 'partition': '5854b1a02b0f1f72cc74236547cae39858a54e511e49ac2e058950d3003a703f', 'parent_runner': '0b0b925ff5cfbc66a2c851d53d70a8441edf3cbf326ff43c94e9013df2108c31', 'parent_result': '65bcdef7f9052a4af3c21cecf6688e7c087aa00f8d2586037aa4a95dfe78836d'}


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in A.FILES.items()}==A.EXPECTED
    assert {k:N.sha(p) for k,p in E.FILES.items()}==E.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(E.S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    controls={'source':M.controls(),'query_hooks':E.M.controls()};assert all(r['passed'] for r in controls.values())
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':24,'sequence_evaluations':432,'controls':controls}));return
    assert not OUT.exists() and not BANK.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    attn=backend.model.transformer.h[9].attn;eps=torch.finfo(torch.float32).eps
    output_weight=torch.stack([attn.c_proj.weight[:,h*128:(h+1)*128].double() for h in (1,4)],dim=1)
    arrays={'output_weight':output_weight.detach().cpu().numpy()};audits=[];reports={};counts=[0,0];finite=True
    def counter(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    def statistics(parts):
        total=parts['total'];cross=parts['numerator_cross_at_full_norm'];rest=parts['normalization_remainder']
        norm=float(total.norm());error=float(rest.norm());dot=lambda x,y:float((x*y).sum())
        return {'total_norm':norm,'cross_norm':float(cross.norm()),'norm_remainder_norm':error,'cross_relative_error':error/norm if norm>1e-8 else None,'cross_signed_projection':dot(cross,total)/norm**2 if norm>1e-8 else None,'norm_signed_projection':dot(rest,total)/norm**2 if norm>1e-8 else None,'cross_norm_cosine':dot(cross,rest)/max(float(cross.norm()*rest.norm()),1e-30),'passed':error<=.10*norm if norm>1e-8 else error<=1e-8}
    parent=json.loads(A.OUT.read_text());handle=attn.register_forward_hook(counter)
    with torch.inference_mode():
        try:
            for panel,rows in splits.items():
                outputs={k:[] for k in ('native','identity','swap')};captures=[];pieces=[]
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side);positions=batch.semantic_positions
                    with M.capture(backend.model,positions) as r:out=backend.native(batch,capture=True)
                    captures.append(r);outputs['native'].append(P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double())
                    args=tuple(r[k] for k in ('sources','readers','keys','values','cos','sin'))+(positions,)
                    bank=M.prepare(*args,eps);ones=torch.ones(len(rows),19,device='cuda',dtype=torch.float64)
                    audits.append({'panel':panel,'side':side,'kind':'source_lineage',**P.agree(torch.nn.functional.rms_norm(r['sources'].sum(1),(1152,),eps=eps),r['normalized_input'].double())})
                    audits.append({'panel':panel,'side':side,'kind':'native_read',**P.agree(M.evaluate(bank,ones),r['native_read'])})
                    local=[]
                    for i in range(19):
                        left=torch.zeros_like(ones);left[:,i]=1
                        part=Q.partition(bank,left,ones-left)
                        sl=torch.einsum('bs,bijst->bijt',left,bank['factors']);sr=torch.einsum('bs,bijst->bijt',ones-left,bank['factors'])
                        ds=torch.einsum('bs,bijst,bt->bij',ones,bank['norm'],ones)+bank['eps2']
                        oracle=torch.einsum('bjt,btjh->bjh',(sl[:,0]*sr[:,1]+sr[:,0]*sl[:,1])/bank['width']**2,bank['values'])/(ds[:,0]*ds[:,1]).sqrt()[:,:,None]
                        audits.append({'panel':panel,'side':side,'kind':A.LABELS[i]+'_cross_oracle',**P.agree(part['numerator_cross_at_full_norm'],oracle,atol=1e-8,rtol=1e-9)})
                        audits.append({'panel':panel,'side':side,'kind':A.LABELS[i]+'_closure',**P.agree(part['total'],part['numerator_cross_at_full_norm']+part['normalization_remainder'],atol=1e-8,rtol=1e-9)})
                        local.append({k:torch.einsum('bjh,djh->bd',v,output_weight) for k,v in part.items()})
                    pieces.append({k:torch.stack([v[k] for v in local]) for k in local[0]})
                    for key,value in bank.items():arrays[panel+'__'+side+'__'+key]=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                for arm in ('identity','swap'):
                    for i,side in enumerate(('base','donor')):
                        batch=P.das._batch(backend,rows,side=side);local=[]
                        source=captures[i if arm=='identity' else 1-i]['normalized_input']
                        with E.M.install(attn,source,batch.semantic_positions,audit=local):out=backend.native(batch,capture=True)
                        outputs[arm].append(P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double())
                        assert len(local)==2
                        for r in local:
                            finite=finite and r['unselected_unchanged'];audits.append({'panel':panel,'kind':arm+'_projection',**P.agree(r['actual'],r['expected'])})
                audits.extend({'panel':panel,'kind':'identity_full_logits',**P.agree(a,b)} for a,b in zip(outputs['identity'],outputs['native']))
                finite=finite and all(bool(z.isfinite().all()) for zs in outputs.values() for z in zs)
                ix=torch.arange(len(rows),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rows],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rows],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                target=center(outputs['native'][1])-center(outputs['native'][0]);mtarget=margin(outputs['native'][1])-margin(outputs['native'][0]);swaps={}
                for i,side in enumerate(('base','donor')):
                    effect=center(outputs['swap'][i])-center(outputs['native'][i]);meffect=margin(outputs['swap'][i])-margin(outputs['native'][i]);sign=1 if i==0 else -1
                    tn=float(target.norm());en=float(effect.norm());mn=float(mtarget.norm())
                    lp=float((effect*target).sum())*sign/max(tn**2,1e-30);mp=float((meffect*mtarget).sum())*sign/max(mn**2,1e-30)
                    swaps[side]={'full_logit_signed_projection':lp,'margin_signed_projection':mp,'effect_norm':en,'effect_relative_norm':en/max(tn,1e-30),'effect_cosine':float((effect*target).sum())*sign/max(en*tn,1e-30),'margin_effects':meffect.cpu().tolist(),'passed':tn>1e-8 and mn>1e-8 and lp>=.10 and mp>=.10}
                audits.append({'panel':panel,'kind':'parent_native_contrast',**P.agree(mtarget,torch.tensor([r['native_contrast'] for r in parent['reports'][panel]['rows']],device='cuda',dtype=torch.float64))})
                endpoint={k:torch.stack([p[k] for p in pieces]) for k in pieces[0]};paired={k:pieces[1][k]-pieces[0][k] for k in pieces[0]}
                per_source={label:{'endpoint':statistics({k:v[:,j] for k,v in endpoint.items()}),'paired':statistics({k:v[j] for k,v in paired.items()})} for j,label in enumerate(A.LABELS)}
                report={'swaps':swaps,'partition_endpoint':statistics(endpoint),'partition_paired':statistics(paired),'sources':per_source,'native_margin_contrasts':mtarget.cpu().tolist(),'row_ids':[r['row_id'] for r in rows]};reports[panel]=report
                print(json.dumps({'panel':panel,'swaps':swaps,'partition_endpoint':report['partition_endpoint'],'partition_paired':report['partition_paired']}),flush=True)
        finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and 'squared_attention' not in attn.__dict__
    a=finite and restored and counts==[24,432] and all(x['passed'] for x in audits)
    predictions={'pred_a_instrument':a,'pred_b_complete_query_carrier':all(s['passed'] for r in reports.values() for s in r['swaps'].values()),'pred_c_numerator_cross_explanation':all(r[k]['passed'] for r in reports.values() for k in ('partition_endpoint','partition_paired')),'pred_d_live_query_swaps':all(s['effect_norm']>1e-8 for r in reports.values() for s in r['swaps'].values())}
    with BANK.open('xb') as f:np.savez_compressed(f,**arrays)
    result={'terminal':'invalid' if not a else 'native_query_partition_complete','predictions':predictions,'reports':reports,'audits':audits,'controls':controls,'hooks_restored':restored,'source_labels':A.LABELS,'row_sha256':{k:manifest['row_sha256'][k] for k in splits},'bank_path':str(BANK),'bank_sha256':N.sha(BANK),'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'native_parameters':sum(p.numel() for p in backend.model.parameters()),'conditional_bank_entries':sum(v.size for v in arrays.values()),'conditional_bank_file_bytes':BANK.stat().st_size,'actual_weight_saving':0,'adoption':False},'scope':'Opened rows; full native query-input swaps and local source/rest interaction partition with fixed native keys/values. No independent token execution or semantic factor identification.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
