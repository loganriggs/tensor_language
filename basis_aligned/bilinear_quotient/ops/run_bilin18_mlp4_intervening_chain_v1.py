"""Intervening attention/MLP factorial during an actual MLP4 source swap.
A: identity/parent/rawdirect1e-3/1e-5, clamps exact,80/1440 counts.
B: MLP-only value effect error<=.10 full-logit ANDmargin allcells.
C: attention-only same criterion. D:all nonresidual effects>1e-8.
Existing partialpath, native background/weightscharged, no fits/adoption.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_mlp_chain pred_c_attention_chain pred_d_live_effects
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_mlp4_value_lineage_native_v1 as R
import intervening_write_clamp as K
from circuit_fast_screen_managed_runner import atomic_create_json

P=R.P;N=R.N;C=R.C;E=R.E;POLY=R.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_MLP4_INTERVENING_CHAIN_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP4_INTERVENING_CHAIN_V1_RESULT.json'
FILES={'prior':PRIOR,'clamp':Path(K.__file__),'parent_runner':Path(R.__file__),'parent_result':R.OUT}
EXPECTED={'prior': 'b7b2e492fdce8746b57652b738892ecea4447ca4775106acb4c273180ae934ab', 'clamp': 'b2a4db0f94590c5eb7cd08bcb8fbc2f8ba81431d714c088d170c9ae703b48f2d', 'parent_runner': '7919d9bbc758d037cf212027175d88aa4e942fb773f90c8ca1fcfeea03331f24', 'parent_result': '1c8c41842c41f1975386cc41b5327effe0d85d52e7a2b2ed4fb007b4e45eb80d'}
CONFIGS={'full':(),'mlp_only':('attention',),'attention_only':('mlp',),'residual_only':('attention','mlp')}


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in R.FILES.items()}==R.EXPECTED
    assert {k:N.sha(p) for k,p in R.Q.FILES.items()}==R.Q.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(E.S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    controls=K.controls();assert controls['passed']
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':80,'sequence_evaluations':1440,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;attn=model.transformer.h[9].attn;mlp=model.transformer.h[4].mlp
    reader=torch.cat([attn.c_v.weight[h*128:(h+1)*128].detach().double() for h in (1,4)])
    parent=json.loads(R.OUT.read_text());assert parent['predictions']['pred_a_instrument'];gamma=parent['gamma4'];eps=torch.finfo(torch.float32).eps
    selected=lambda value:value.view(*value.shape[:2],9,128)[:,:,[1,4]]
    logits=lambda out,rows:P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double()
    counts=[0,0];audits=[];reports={};clamps_exact=True;unselected=True;recurrence=True;finite=True
    def counter(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=attn.register_forward_hook(counter)
    with torch.inference_mode():
        try:
            for panel,rows in splits.items():
                outputs={k:[] for k in ('native','identity',*CONFIGS)};fields={k:[] for k in CONFIGS};native=[];middle=[]
                lengths=[len(r['base_ids']) for r in rows]
                assert all(len(r['donor_ids'])==n for r,n in zip(rows,lengths))
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side)
                    with K.capture(model) as writes:
                        with C.capture(model) as record:out=backend.native(batch,capture=True)
                    native.append(record);middle.append(writes);outputs['native'].append(logits(out,rows));recurrence=recurrence and record['recurrence_bitwise']
                for config,kinds in CONFIGS.items():
                    for i,side in enumerate(('base','donor')):
                        batch=P.das._batch(backend,rows,side=side)
                        with C.replace_mlp(mlp,native[1-i]['mlp_output'],lengths):
                            with K.clamp(model,middle[i],lengths,kinds):
                                with K.capture(model) as writes:
                                    with C.capture(model) as record:out=backend.native(batch,capture=True)
                        fields[config].append(selected(record['value']));recurrence=recurrence and record['recurrence_bitwise']
                        clamps_exact=clamps_exact and all(torch.equal(record['mlp_output'][j,:n],native[1-i]['mlp_output'][j,:n]) for j,n in enumerate(lengths))
                        clamps_exact=clamps_exact and all(torch.equal(writes[k][j,:n],middle[i][k][j,:n]) for k in writes if k[0] in kinds for j,n in enumerate(lengths))
                        if config=='residual_only':
                            u=native[i]['residual'].double();delta=native[1-i]['mlp_output'].double()-native[i]['mlp_output'].double()
                            raw=(torch.nn.functional.rms_norm(u+gamma*delta,(1152,),eps=eps)@reader.T).view_as(fields[config][-1])
                            audits.append({'panel':panel,'side':side,'kind':'residual_only_raw_value',**P.agree(torch.cat([raw[j,:n] for j,n in enumerate(lengths)]),torch.cat([fields[config][-1][j,:n].double() for j,n in enumerate(lengths)]))})
                for config in CONFIGS:
                    for i,side in enumerate(('base','donor')):
                        batch=P.das._batch(backend,rows,side=side);local=[]
                        with C.replace_value(attn,fields[config][i],lengths,audit=local):out=backend.native(batch,capture=True)
                        assert len(local)==1;unselected=unselected and all(local);outputs[config].append(logits(out,rows))
                for i,side in enumerate(('base','donor')):
                    batch=P.das._batch(backend,rows,side=side)
                    with K.clamp(model,middle[i],lengths,('attention','mlp')):out=backend.native(batch,capture=True)
                    outputs['identity'].append(logits(out,rows))
                audits.extend({'panel':panel,'kind':'identity_clamp_full_logits',**P.agree(a,b)} for a,b in zip(outputs['identity'],outputs['native']))
                finite=finite and all(bool(z.isfinite().all()) for zs in outputs.values() for z in zs)
                ix=torch.arange(len(rows),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rows],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rows],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                cells={}
                for i,side in enumerate(('base','donor')):
                    effects={k:center(outputs[k][i])-center(outputs['native'][i]) for k in CONFIGS};meffects={k:margin(outputs[k][i])-margin(outputs['native'][i]) for k in CONFIGS}
                    reference=effects['full'];mreference=meffects['full'];rn=float(reference.norm());mn=float(mreference.norm());arms={}
                    for config in CONFIGS:
                        effect=effects[config];meffect=meffects[config];error=float((effect-reference).norm());merror=float((meffect-mreference).norm())
                        arms[config]={'effect_norm':float(effect.norm()),'relative_effect_error':error/rn if rn>1e-8 else None,'relative_margin_error':merror/mn if mn>1e-8 else None,'signed_effect_projection':float((effect*reference).sum())/max(rn**2,1e-30),'signed_margin_projection':float((meffect*mreference).sum())/max(mn**2,1e-30),'margin_effects':meffect.cpu().tolist(),'passed':(error<=.10*rn if rn>1e-8 else error<=1e-8) and (merror<=.10*mn if mn>1e-8 else merror<=1e-8)}
                    previous=parent['reports'][panel]['cells'][side]['arms']['value_full']
                    audits.append({'panel':panel,'side':side,'kind':'parent_value_margin',**P.agree(mreference,torch.tensor(previous['margin_effects'],device='cuda',dtype=torch.float64))})
                    norm_error=abs(rn-previous['effect_norm']);audits.append({'panel':panel,'side':side,'kind':'parent_value_effect_norm','max_abs':norm_error,'passed':norm_error<=1e-5*max(previous['effect_norm'],1e-8)})
                    cells[side]={'arms':arms}
                reports[panel]={'cells':cells,'row_ids':[r['row_id'] for r in rows]}
                print(json.dumps({'panel':panel,'cells':cells,'counts':counts}),flush=True)
        finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    a=finite and recurrence and clamps_exact and unselected and restored and counts==[80,1440] and all(v['passed'] for v in audits)
    cells=[c for r in reports.values() for c in r['cells'].values()]
    predictions={'pred_a_instrument':a,'pred_b_mlp_chain':all(c['arms']['mlp_only']['passed'] for c in cells),'pred_c_attention_chain':all(c['arms']['attention_only']['passed'] for c in cells),'pred_d_live_effects':all(c['arms'][arm]['effect_norm']>1e-8 for c in cells for arm in ('full','mlp_only','attention_only'))}
    result={'terminal':'invalid' if not a else 'intervening_chain_complete','predictions':predictions,'reports':reports,'audits':audits,'controls':controls,'recurrence_bitwise':recurrence,'clamps_exact':clamps_exact,'unselected_values_unchanged':unselected,'hooks_restored':restored,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0,'adoption':False},'scope':'Existing partial localV9 path, opened rows, actual intervening computation clamps duringMLP4swap. Earlier has full-logit carrier miss remains; no whole-task sufficiency or independent extraction claim.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
