#!/usr/bin/env python3
# BQGATE: live causal-prefix binding;384forwards6144seq,0fits.
"""A instrument; B prefix necessity .25; C new/D inherited/E composition .10."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import source_stage_native_v2 as N
import prefix_memory_port_v1 as M
import causal_prefix_binding_v1 as B
import mixed_state_projector_v1 as Q
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'CAUSAL_PREFIX_BINDING_V1_BINDING.json';OUT=POLY/'CAUSAL_PREFIX_BINDING_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MATURE_VALUE_MLP8_CONSUMERS_V1_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    for start in (0,16):
        for early,late in [(2,4),(4,2)]:B.groups_and_signs(bank['corners'][start:start+16],early,late)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':384,'sequences':6144,'fits':0,'patched_attention_calls':2880}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2);model=backend.model
    q_np=Q.projector(bank['corners'],2,4);ids=bank['reader_ids'];counts=[0,0];reports=[];patch_count=0;tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for world in bank['worlds']:
                foil=1 if world['foil']=='himself' else 2
                early,late=(2,4) if world['layout']=='original' else (4,2)
                native=N.run(backend,world,ids);z0=native['logits'].cpu().numpy();arms={};audits=[];first_ok=True
                for mode in ('zero','full','new','inherited','joint'):
                    def context(start,end):
                        return B.remove(model,M.slice_batch(native['memory'],start,end),bank['corners'][start:end],early,late,mode,audits)
                    result=N.run(backend,world,ids,attention_context=context)
                    arms[mode]=result['logits'].cpu().numpy();first_ok &= torch.equal(result['first'],native['first'])
                patch_count+=len(audits)
                bridges={'native':S.bridge(z0[:,ids],lookup[world['world_id']]['native_logits']),
                    'zero_full':S.bridge(arms['zero'],z0),'joint_full':S.bridge(arms['joint'],arms['full'])}
                effects={k:z0-v for k,v in arms.items() if k in ('full','new','inherited')}
                def norms(a,b):
                    ar,br=a[:,ids],b[:,ids]
                    return {'margin':S.ratio(ar[:,0]-ar[:,foil],br[:,0]-br[:,foil]),
                        'readers':S.ratio(ar-ar.mean(-1,keepdims=True),br-br.mean(-1,keepdims=True)),
                        'vocabulary':S.ratio(a-a.mean(-1,keepdims=True),b-b.mean(-1,keepdims=True))}
                total=q_np@effects['full'];native_mixed=q_np@z0
                necessity=norms(q_np@arms['full'],native_mixed);materiality=norms(total,native_mixed)
                errors={k:norms(q_np@effects[k]-total,total) for k in ('new','inherited')}
                interaction=effects['full']-effects['new']-effects['inherited']
                composition={'mixed':norms(q_np@interaction,total),'full':norms(interaction,effects['full'])}
                live=all(v is not None and v>=.10 for v in materiality.values())
                reports.append({'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'early_factor_index':early,'late_factor_index':late,'native_logits':z0[:,ids].tolist(),
                    'arm_logits':{k:v[:,ids].tolist() for k,v in arms.items()},'bridges':bridges,
                    'necessity_errors':necessity,'materiality':materiality,'branch_errors':errors,'composition_errors':composition,
                    'prefix_necessity_passed':all(S.passed(v,.25) for v in necessity.values()),
                    'new_passed':live and all(S.passed(v,.10) for v in errors['new'].values()),
                    'inherited_passed':live and all(S.passed(v,.10) for v in errors['inherited'].values()),
                    'composition_passed':all(S.passed(v,.10) for obj in composition.values() for v in obj.values()),
                    'maximum_local_closure_relative':max(a['closure_relative'] for a in audits),
                    'instrument_passed':first_ok and len(audits)==90 and all(a['prefix_factors_bitwise'] and a['late_factor_absent_bitwise'] and a['prefix_output_bitwise'] and a['finite'] for a in audits)
                        and all(np.isfinite(v).all() for v in arms.values())
                        and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    restored &= all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[384,6144] and patch_count==2880 and all(r['instrument_passed'] for r in reports),
        'pred_b_prefix_necessity':all(r['prefix_necessity_passed'] for r in reports),
        'pred_c_new':all(r['new_passed'] for r in reports),'pred_d_inherited':all(r['inherited_passed'] for r in reports),
        'pred_e_composition':all(r['composition_passed'] for r in reports)}
    result={'terminal':'causal_prefix_binding_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'patched_attention_calls':patch_count,'fits':0,
            'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Live conditional early-memory/late-query operation removals at all later prefix readers, both word orders. Native counterfactual producers and all weights retained; no independent semantic extraction.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','prefix_necessity_passed','new_passed','inherited_passed','composition_passed','branch_errors')} for r in reports]}))
    assert preds['pred_a_instrument']


if __name__=='__main__':main()
