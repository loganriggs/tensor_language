#!/usr/bin/env python3
# BQGATE: value x mean-routing x recipient; 448 forwards/7168 sequences; no fits.
"""A native factor bridge; B value-following .10; C routing-following .10;
D interaction .10, both readouts, every pair and recipient. Preserve nulls.
All545902902nativeparameters retained; no independent inputs or structural saving.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mature_value_route_native_v1 as M
import symmetric_factor_interaction_v1 as F
import attention_output_delta_v1 as A
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
import cross_context_effect_attribution_v1 as C
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MATURE_VALUE_ROUTE_V1_BINDING.json';OUT=POLY/'MATURE_VALUE_ROUTE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_COMMON_INTERCHANGE_V1_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':448,'sequences':7168,'fits':0,'pairs':16}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    qs=F.projectors(source['corners']).to('cuda');q=qs[3];mean=qs[0];q_np=q.cpu().numpy()
    attn=backend.model.transformer.h[9].attn;weight=attn.c_proj.weight.double();counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    count_handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    def run(world,state,component):
        delta=state['observed_common'].clone()
        if component is not None:delta[:,-2:]-=component
        chunks=[];audit=[];foil=1 if world['foil']=='himself' else 2
        for start in (0,16):
            rows=world['rows'][start:start+16]
            batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                (source['reader_ids'][0],)*16,(source['reader_ids'][foil],)*16,(world['length']-1,)*16)
            with A.subtract(attn,state['native_write'][start:start+16],delta[start:start+16],audit):
                with G.capture(backend.model) as captured:backend.native(batch,capture=False)
            chunks.append(G.endpoint_logits(captured['raw_logits'],batch)[:,source['reader_ids']].double())
        return torch.cat(chunks).cpu().numpy()
    try:
        with torch.inference_mode():
            for i in range(16):
                worlds=source['worlds'][2*i:2*i+2];old=parent['reports'][i]
                assert old['world_id']==worlds[0]['world_id'] and worlds[1]['world_id']=='fronted_pp:'+worlds[0]['world_id']
                for a,b in zip(worlds[0]['rows'],worlds[1]['rows'],strict=True):
                    assert a['factors']==b['factors'] and a['ids'][-2:]==b['ids'][-2:]
                states=[M.capture(backend,w,q,mean,source['reader_ids']) for w in worlds]
                assert all(s['instrument']['passed'] for s in states),'Native factor bridge before hybrid outcomes'
                removed=[run(w,s,None) for w,s in zip(worlds,states,strict=True)]
                bridges=[]
                for r in (0,1):
                    bridges.extend([S.bridge(states[r]['native_logits'],old['native_logits'][r]),S.bridge(removed[r],old['removed_logits'][r])])
                assert all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges)
                effects=np.zeros((2,2,2,32,3))
                for phase,choices in [('homogeneous',[(0,0),(1,1)]),('hybrid',[(0,1),(1,0)])]:
                    for value,routing in choices:
                        component=M.contract(states[routing]['pattern'],states[value]['value'],weight,1.65625)
                        for recipient in (0,1):
                            z=run(worlds[recipient],states[recipient],component)
                            effects[value,routing,recipient]=z-removed[recipient]
                            if phase=='homogeneous':
                                target=old['native_logits'][recipient] if value==recipient else old['cross_logits'][recipient]
                                bridges.append(S.bridge(z,target))
                    if phase=='homogeneous':
                        assert all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges),'Homogeneous output bridge before hybrid outcomes'
                mixed=np.einsum('ij,vprjc->vpric',q_np,effects);foil=1 if worlds[0]['foil']=='himself' else 2
                score=[]
                for recipient in (0,1):
                    m=mixed[:,:,recipient]
                    objects={'margin':m[...,0]-m[...,foil],'readers':m-m.mean(-1,keepdims=True)}
                    scores={name:C.analyse(x) for name,x in objects.items()}
                    interaction={name:S.ratio(np.asarray(scores[name]['interaction']),x[0,0] if np.linalg.norm(x[0,0])>=np.linalg.norm(x[1,1]) else x[1,1]) for name,x in objects.items()}
                    score.append({'recipient':worlds[recipient]['layout'],'scores':scores,'interaction_ratios':interaction,
                        'value_following_passed':all(s['producer_following_passed'] for s in scores.values()),
                        'routing_following_passed':all(s['reader_following_passed'] for s in scores.values()),
                        'small_interaction_passed':all(S.passed(v,.10) for v in interaction.values())})
                reports.append({'world_id':worlds[0]['world_id'],'foil':worlds[0]['foil'],'effects':effects.tolist(),
                    'recipient_scores':score,'bridges':bridges,'instruments':[s['instrument'] for s in states],
                    'instrument_passed':bool(np.isfinite(effects).all()),
                    'value_following_passed':all(s['value_following_passed'] for s in score),
                    'routing_following_passed':all(s['routing_following_passed'] for s in score),
                    'small_interaction_passed':all(s['small_interaction_passed'] for s in score)})
    finally:count_handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[448,7168] and all(r['instrument_passed'] for r in reports),
        'pred_b_value_following':all(r['value_following_passed'] for r in reports),
        'pred_c_routing_following':all(r['routing_following_passed'] for r in reports),
        'pred_d_small_interaction':all(r['small_interaction_passed'] for r in reports)}
    result={'terminal':'mature_value_route_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Native value x mean-routing x recipient factorial on opened layouts; information-stage alignment of three sources. No independent production or semantic circuit promotion.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','value_following_passed','routing_following_passed','small_interaction_passed')} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
