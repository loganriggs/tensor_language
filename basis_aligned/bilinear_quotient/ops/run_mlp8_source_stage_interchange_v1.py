#!/usr/bin/env python3
# BQGATE: MLP8 source stage interchange;256 forwards4096 sequences,0fits.
"""A instrument; B producer; C reader; D interaction, <=.10 all readouts/pairs."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as Q
import module_output_delta_v1 as D
import source_stage_interface_v1 as T
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MLP8_SOURCE_STAGE_INTERCHANGE_V1_BINDING.json'
OUT=POLY/'MLP8_SOURCE_STAGE_INTERCHANGE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MATURE_VALUE_MLP8_CONSUMERS_V1_RESULT.json').read_text())
    assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']};worlds={r['world_id']:r for r in bank['worlds']}
    pairs=[(w,worlds['fronted_pp:'+w['world_id']]) for w in bank['worlds'] if w['layout']=='original']
    assert len(pairs)==16 and all(T.validate_pair(*pair) for pair in pairs)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':256,'sequences':4096,'fits':0}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;mlp=model.transformer.h[8].mlp;attn=model.transformer.h[9].attn
    q_np=Q.projector(bank['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    ids=bank['reader_ids'];counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for original,fronted in pairs:
                pair=[original,fronted];foil=1 if original['foil']=='himself' else 2
                def run(world,delta=None,native=None):
                    chunks=[];audit=[]
                    for start in (0,16):
                        rows=world['rows'][start:start+16];length=world['length']
                        b=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                            (ids[0],)*16,(ids[foil],)*16,(length-1,)*16)
                        record={};handles=[]
                        try:
                            handles.append(mlp.register_forward_hook(lambda _m,_a,out:record.update(mlp=out.detach().clone())))
                            handles.append(attn.register_forward_hook(lambda _m,_a,out:record.update(first=out[1].detach().clone())))
                            if delta is None:
                                with G.capture(model) as g:backend.native(b,capture=False)
                            else:
                                with D.subtract(mlp,native['mlp'][start:start+16],delta[start:start+16],audit):
                                    with G.capture(model) as g:backend.native(b,capture=False)
                            record['logits']=G.endpoint_logits(g['raw_logits'],b).double()
                        finally:
                            for h in handles:h.remove()
                        chunks.append(record)
                    return {k:torch.cat([r[k] for r in chunks]) for k in chunks[0]}
                native=[run(w) for w in pair]
                components=[torch.einsum('ij,jtd->itd',q,r['mlp'].double()) for r in native]
                prefix_ratio=[float(c[:,:-3].norm()/c.norm().clamp_min(1e-30)) for c in components]
                removed=[run(w,c,n) for w,c,n in zip(pair,components,native)]
                swaps=[];identities=[]
                for recipient in (0,1):
                    donor=1-recipient;c=components[recipient]
                    mapped=torch.zeros_like(c);mapped[:,-3:]=components[donor][:,-3:]
                    self_mapped=torch.zeros_like(c);self_mapped[:,-3:]=c[:,-3:]
                    swaps.append(run(pair[recipient],c-mapped,native[recipient]))
                    identities.append(run(pair[recipient],c-self_mapped,native[recipient]))
                z=[n['logits'].cpu().numpy() for n in native];base=[r['logits'].cpu().numpy() for r in removed]
                effects=np.empty((2,2,32,len(z[0][0])))
                for r in (0,1):
                    effects[r,r]=z[r]-base[r]
                    effects[1-r,r]=swaps[r]['logits'].cpu().numpy()-base[r]
                projected=np.einsum('ij,abjv->abiv',q_np,effects)
                bridges={}
                for r,w in enumerate(pair):
                    old=lookup[w['world_id']]
                    bridges[f'native_{r}']=S.bridge(z[r][:,ids],old['native_logits'])
                    bridges[f'removal_{r}']=S.bridge(base[r][:,ids],old['arm_logits']['full'])
                    bridges[f'identity_{r}']=S.bridge(identities[r]['logits'].cpu().numpy(),z[r])
                def representations(x):
                    small=x[:,ids]
                    return {'margin':small[:,0]-small[:,foil],
                            'readers':small-small.mean(-1,keepdims=True),
                            'vocabulary':x-x.mean(-1,keepdims=True)}
                views={(p,r):representations(projected[p,r]) for p in (0,1) for r in (0,1)}
                errors={'producer':{},'reader':{},'interaction':{}}
                for metric in ('margin','readers','vocabulary'):
                    e00,e01,e10,e11=[views[k][metric] for k in [(0,0),(0,1),(1,0),(1,1)]]
                    errors['producer'][metric]=[S.ratio(e10-e11,e11),S.ratio(e01-e00,e00)]
                    errors['reader'][metric]=[S.ratio(e10-e00,e00),S.ratio(e01-e11,e11)]
                    denom=max(np.linalg.norm(e00),np.linalg.norm(e11))
                    errors['interaction'][metric]=float(np.linalg.norm(e11-e10-e01+e00)/denom) if denom else None
                first_ok=all(torch.equal(r['first'],native[i]['first']) for i in (0,1) for r in [removed[i],swaps[i],identities[i]])
                reports.append({'pair_id':original['world_id'],'foil':original['foil'],'errors':errors,
                    'effect_logits':effects[...,ids].tolist(),'native_logits':[v[:,ids].tolist() for v in z],
                    'removed_logits':[v[:,ids].tolist() for v in base],'bridges':bridges,'prefix_relative_norm':prefix_ratio,
                    'producer_passed':all(S.passed(v,.10) for a in errors['producer'].values() for v in a),
                    'reader_passed':all(S.passed(v,.10) for a in errors['reader'].values() for v in a),
                    'task_producer_passed':all(S.passed(v,.10) for k in ('margin','readers') for v in errors['producer'][k]),
                    'task_reader_passed':all(S.passed(v,.10) for k in ('margin','readers') for v in errors['reader'][k]),
                    'interaction_passed':all(S.passed(v,.10) for v in errors['interaction'].values()),
                    'instrument_passed':first_ok and max(prefix_ratio)<=1e-8 and np.isfinite(effects).all()
                        and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    preds={'pred_a_instrument':restored and counts==[256,4096] and all(r['instrument_passed'] for r in reports),
        'pred_b_producer':all(r['producer_passed'] for r in reports),
        'pred_c_reader':all(r['reader_passed'] for r in reports),
        'pred_d_interaction':all(r['interaction_passed'] for r in reports)}
    result={'terminal':'source_stage_interchange_complete' if preds['pred_a_instrument'] else 'invalid',
        'predictions':preds,'reports':reports,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),
        'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Full MLP8 mixed-source interchange at three information stages with all native consumers live. Previously opened layouts; native counterfactuals retained, no independent extraction.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'pairs':[{k:r[k] for k in ('pair_id','producer_passed','reader_passed','interaction_passed','errors')} for r in reports]}))
    assert preds['pred_a_instrument']


if __name__=='__main__':main()
