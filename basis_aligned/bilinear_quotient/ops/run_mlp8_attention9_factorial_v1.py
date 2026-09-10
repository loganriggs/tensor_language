#!/usr/bin/env python3
# BQGATE: full MLP8 source by whole attention9 response; 384 forwards/6144 sequences.
"""A instrument; B attention dominance; C bypass dominance; D interaction (.10).
All native weights and counterfactual sources retained; zero fits or structural saving.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from contextlib import nullcontext
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as Q
import module_output_delta_v1 as D
import attention_output_delta_v1 as A
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
import mlp8_consumer_effect_geometry_v1 as E
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MLP8_ATTENTION9_FACTORIAL_V1_BINDING.json';OUT=POLY/'MLP8_ATTENTION9_FACTORIAL_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MATURE_VALUE_MLP8_CONSUMERS_V1_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'model_loaded':False,'gpu_accessed':False,'forwards':384,'sequences':6144,'fits':0}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;mlp=model.transformer.h[8].mlp;attn=model.transformer.h[9].attn
    q_np=Q.projector(bank['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for world in bank['worlds']:
                foil=1 if world['foil']=='himself' else 2
                def batch(start):
                    rows=world['rows'][start:start+16]
                    return P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                        (bank['reader_ids'][0],)*16,(bank['reader_ids'][foil],)*16,(world['length']-1,)*16)
                def run(b,capture=False):
                    record={};handles=[]
                    try:
                        if capture:
                            handles.append(mlp.register_forward_hook(lambda _m,_a,out:record.update(mlp=out.detach().clone())))
                            handles.append(attn.register_forward_hook(lambda _m,_a,out:record.update(write=out[0].detach().clone(),first=out[1].detach().clone())))
                        with G.capture(model) as g:backend.native(b,capture=False)
                        record['logits']=G.endpoint_logits(g['raw_logits'],b)[:,bank['reader_ids']].double()
                    finally:
                        for h in handles:h.remove()
                    return record
                def join(chunks):return {k:torch.cat([r[k] for r in chunks]) for k in chunks[0]}
                native=join([run(batch(start),True) for start in (0,16)])
                delta=torch.einsum('ij,jtd->itd',q,native['mlp'].double());source_audit=[];changed=[]
                for start in (0,16):
                    with D.subtract(mlp,native['mlp'][start:start+16],delta[start:start+16],source_audit):
                        changed.append(run(batch(start),True))
                edited=join(changed);states=[native,edited]
                first_equal=torch.equal(native['first'],edited['first']);assert first_equal
                cube=[];clamp_audit=[]
                for source in (0,1):
                    row=[]
                    for writer in (0,1):
                        chunks=[]
                        for start in (0,16):
                            source_delta=delta[start:start+16] if source else torch.zeros_like(delta[start:start+16])
                            expected=states[source]['write'][start:start+16]
                            target=states[writer]['write'][start:start+16]
                            with D.subtract(mlp,native['mlp'][start:start+16],source_delta,source_audit):
                                with A.subtract(attn,expected,expected.double()-target.double(),clamp_audit):
                                    chunks.append(run(batch(start))['logits'])
                            assert torch.equal((expected.double()-clamp_audit[-1]).to(expected),target)
                        row.append(torch.cat(chunks).cpu().tolist())
                    cube.append(row)
                cube=np.array(cube);z0=cube[0,0];old=lookup[world['world_id']]
                bridges={'native':S.bridge(native['logits'].cpu().numpy(),old['native_logits']),
                    'source':S.bridge(edited['logits'].cpu().numpy(),old['arm_logits']['full']),
                    'identity_native':S.bridge(cube[0,0],native['logits'].cpu().numpy()),
                    'identity_edited':S.bridge(cube[1,1],edited['logits'].cpu().numpy())}
                total,attention,bypass,interaction=E.factorial_effects(cube)
                def norms(x,y):return {'margin':S.ratio(x[:,0]-x[:,foil],y[:,0]-y[:,foil]),
                    'readers':S.ratio(x-x.mean(-1,keepdims=True),y-y.mean(-1,keepdims=True))}
                tq=q_np@total;materiality=norms(tq,q_np@z0)
                errors={k:norms(q_np@v-tq,tq) for k,v in [('attention',attention),('bypass',bypass)]}
                interactions={'mixed':norms(q_np@interaction,tq),'full':norms(interaction,total)}
                live=all(v is not None and v>=.10 for v in materiality.values())
                reports.append({'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'cube_logits':cube.tolist(),'bridges':bridges,'first_value_unchanged':first_equal,
                    'dominance_errors':errors,'source_materiality':materiality,'interaction_errors':interactions,
                    'attention_passed':live and all(S.passed(v,.10) for v in errors['attention'].values()),
                    'bypass_passed':live and all(S.passed(v,.10) for v in errors['bypass'].values()),
                    'interaction_passed':all(S.passed(v,.10) for obj in interactions.values() for v in obj.values()),
                    'instrument_passed':np.isfinite(cube).all() and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    preds={'pred_a_instrument':restored and counts==[384,6144] and all(r['instrument_passed'] for r in reports),
        'pred_b_attention':all(r['attention_passed'] for r in reports),
        'pred_c_bypass':all(r['bypass_passed'] for r in reports),
        'pred_d_interaction':all(r['interaction_passed'] for r in reports)}
    result={'terminal':'source_attention_factorial_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Full MLP8 mixed source removal crossed with whole attention9 write; native source-consistent background and live suffix. No independent circuit or new OOD evidence.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','attention_passed','bypass_passed','interaction_passed','dominance_errors')} for r in reports]}));assert preds['pred_a_instrument']

if __name__=='__main__':main()
