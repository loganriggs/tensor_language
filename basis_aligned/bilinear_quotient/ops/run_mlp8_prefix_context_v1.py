#!/usr/bin/env python3
# BQGATE: source by prefix-memory by recipient;320forwards5120seq,0fits.
"""A instrument; B memory; C remaining context; D interaction, <=.10 every pair."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import source_stage_native_v1 as N
import source_stage_interface_v1 as T
import mixed_state_projector_v1 as Q
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MLP8_PREFIX_CONTEXT_V1_BINDING.json';OUT=POLY/'MLP8_PREFIX_CONTEXT_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MLP8_SOURCE_STAGE_INTERCHANGE_V1_RESULT.json').read_text())
    assert parent['predictions']['pred_a_instrument']
    lookup={r['pair_id']:r for r in parent['reports']};worlds={w['world_id']:w for w in bank['worlds']}
    pairs=[(w,worlds['fronted_pp:'+w['world_id']]) for w in bank['worlds'] if w['layout']=='original']
    assert len(pairs)==16 and all(T.validate_pair(*pair) for pair in pairs)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':320,'sequences':5120,'fits':0,'patched_attention_calls':2304}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2);model=backend.model
    q_np=Q.projector(bank['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    ids=bank['reader_ids'];counts=[0,0];reports=[];tic=time.perf_counter();patch_count=0
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for original,fronted in pairs:
                pair=[original,fronted];old=lookup[original['world_id']];foil=1 if original['foil']=='himself' else 2
                native=[N.run(backend,w,ids) for w in pair]
                components=[torch.einsum('ij,jtd->itd',q,n['mlp'].double()) for n in native]
                prefix_ratios=[float(c[:,:-3].norm()/c.norm().clamp_min(1e-30)) for c in components]
                cube=np.empty((2,2,2,32,native[0]['logits'].shape[-1]));audits=[];first_ok=True
                for producer in (0,1):
                    for memory in (0,1):
                        for recipient in (0,1):
                            c=components[recipient];mapped=torch.zeros_like(c);mapped[:,-3:]=components[producer][:,-3:]
                            result=N.run(backend,pair[recipient],ids,c-mapped,native[recipient],
                                (native[memory]['memory'],native[recipient]['memory']))
                            cube[producer,memory,recipient]=result['logits'].cpu().numpy()
                            audits.extend(result['memory_audit']);first_ok &= torch.equal(result['first'],native[recipient]['first'])
                patch_count+=len(audits);bridges={}
                for producer in (0,1):
                    for recipient in (0,1):
                        reference=np.asarray(old['effect_logits'])[producer,recipient]+np.asarray(old['removed_logits'])[recipient]
                        bridges[f'parent_{producer}{recipient}']=S.bridge(cube[producer,recipient,recipient][:,ids],reference)
                for recipient in (0,1):
                    bridges[f'identity_full_{recipient}']=S.bridge(cube[recipient,recipient,recipient],native[recipient]['logits'].cpu().numpy())
                effects=cube[1]-cube[0];projected=np.einsum('ij,krjv->kriv',q_np,effects)
                def representations(x):
                    small=x[:,ids]
                    return {'margin':small[:,0]-small[:,foil],
                        'readers':small-small.mean(-1,keepdims=True),'vocabulary':x-x.mean(-1,keepdims=True)}
                views={(k,r):representations(projected[k,r]) for k in (0,1) for r in (0,1)}
                errors={'memory':{},'remaining_context':{},'interaction':{}}
                for metric in ('margin','readers','vocabulary'):
                    e00,e01,e10,e11=[views[k][metric] for k in [(0,0),(0,1),(1,0),(1,1)]]
                    errors['memory'][metric]=[S.ratio(e10-e11,e11),S.ratio(e01-e00,e00)]
                    errors['remaining_context'][metric]=[S.ratio(e10-e00,e00),S.ratio(e01-e11,e11)]
                    denom=max(np.linalg.norm(e00),np.linalg.norm(e11))
                    errors['interaction'][metric]=float(np.linalg.norm(e11-e10-e01+e00)/denom) if denom else None
                reports.append({'pair_id':original['world_id'],'foil':original['foil'],'errors':errors,
                    'cube_logits':cube[...,ids].tolist(),'same_source_edit_effect_logits':effects[...,ids].tolist(),
                    'bridges':bridges,'prefix_relative_norm':prefix_ratios,'patched_attention_calls':len(audits),
                    'memory_change_norm_range':[min(a['memory_read_change_norm'] for a in audits),max(a['memory_read_change_norm'] for a in audits)],
                    'memory_passed':all(S.passed(v,.10) for a in errors['memory'].values() for v in a),
                    'remaining_context_passed':all(S.passed(v,.10) for a in errors['remaining_context'].values() for v in a),
                    'interaction_passed':all(S.passed(v,.10) for v in errors['interaction'].values()),
                    'instrument_passed':first_ok and max(prefix_ratios)<=1e-8 and np.isfinite(cube).all()
                        and len(audits)==144 and all(a['prefix_factors_bitwise'] and a['prefix_output_bitwise'] and a['finite'] for a in audits)
                        and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    restored &= all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[320,5120] and patch_count==2304 and all(r['instrument_passed'] for r in reports),
        'pred_b_memory':all(r['memory_passed'] for r in reports),
        'pred_c_remaining_context':all(r['remaining_context_passed'] for r in reports),
        'pred_d_interaction':all(r['interaction_passed'] for r in reports)}
    result={'terminal':'prefix_context_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'patched_attention_calls':patch_count,
            'prefix_contractions':2*patch_count,'factor_bank_fp32_scalars_per_pair':14929920,'phase_bf16_scalars_per_pair':6912,
            'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Same source-edit response with independently swapped native prefix memory and recipient context. Live queries and within-suffix computation; native factors and all weights retained, no independent semantic circuit.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'pairs':[{k:r[k] for k in ('pair_id','memory_passed','remaining_context_passed','interaction_passed','errors')} for r in reports]}))
    assert preds['pred_a_instrument']


if __name__=='__main__':main()
