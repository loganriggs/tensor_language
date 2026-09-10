#!/usr/bin/env python3
# BQGATE: genuine two-position component interchange; 256 forwards/4096 sequences, no fits.
"""A instrument; B producer-following .10; C reader-following .10; D interaction .10.
Both directions and both mixed readouts in every pair. No gain/head/row rescue.
All 545902902 native parameters retained; no independent producer extraction.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import live_value_factorial_executor_v2 as V
import attention_output_delta_v1 as A
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
import common_write_transport_v1 as T
import cross_context_effect_attribution_v1 as C
import mixed_state_projector_v1 as Q
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_COMMON_INTERCHANGE_V1_BINDING.json';OUT=POLY/'THIRD_NOUN_COMMON_INTERCHANGE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_COMMON_SUFFIX_V1_RESULT.json').read_text())
    assert parent['predictions']['pred_b_common_fidelity'] and parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':256,'sequences':4096,'fits':0,'pairs':16}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    q_np=Q.projector(source['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    attn=backend.model.transformer.h[9].attn;counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    count_handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    def capture(world):
        writes=[];h=attn.register_forward_hook(lambda _m,_a,out:writes.append(out[0].detach().clone()))
        try:base=V.run_world(backend,world,q,source['reader_ids'])
        finally:h.remove()
        assert len(writes)==4
        native=torch.cat(writes[:2]);d=native.double()-torch.cat(writes[2:]).double()
        mixed=torch.einsum('ij,jtd->itd',q,d);common=torch.zeros_like(mixed)
        common[:,world['common_positions']]=mixed[:,world['common_positions']]
        purity=float((common-torch.einsum('ij,jtd->itd',q,common)).norm()/common.norm().clamp_min(1e-30))
        assert purity<=1e-10
        return base,native,common
    def run(world,native,delta):
        chunks=[];audit=[];foil=1 if world['foil']=='himself' else 2
        for start in (0,16):
            rows=world['rows'][start:start+16]
            batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                (source['reader_ids'][0],)*16,(source['reader_ids'][foil],)*16,(world['length']-1,)*16)
            with A.subtract(attn,native[start:start+16],delta[start:start+16],audit):
                with G.capture(backend.model) as captured:backend.native(batch,capture=False)
            chunks.append(G.endpoint_logits(captured['raw_logits'],batch)[:,source['reader_ids']].double())
        return torch.cat(chunks).cpu().numpy()
    try:
        with torch.inference_mode():
            for index in range(0,32,2):
                worlds=source['worlds'][index:index+2]
                assert worlds[0]['layout']=='original' and worlds[1]['layout']=='fronted_pp'
                assert worlds[1]['world_id']=='fronted_pp:'+worlds[0]['world_id']
                states=[capture(w) for w in worlds]
                removed=[run(w,s[1],s[2]) for w,s in zip(worlds,states,strict=True)]
                bridges=[]
                for w,s,z in zip(worlds,states,removed,strict=True):
                    old=lookup[w['world_id']]
                    bridges.extend([S.bridge(s[0]['native_logits'],old['native_logits']),S.bridge(z,old['arm_logits']['mixed'])])
                assert all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges),'Diagonal replay before cross interpretation'
                swaps=[];effects=np.zeros((2,2,32,3))
                for recipient in (0,1):
                    donor=1-recipient;s=states[recipient]
                    mapped=T.transport(states[donor][2],worlds[donor],worlds[recipient])
                    z=run(worlds[recipient],s[1],s[2]-mapped);swaps.append(z)
                    effects[recipient,recipient]=np.array(s[0]['native_logits'])-removed[recipient]
                    effects[donor,recipient]=z-removed[recipient]
                mixed=np.einsum('ij,prjc->pric',q_np,effects)
                foil=1 if worlds[0]['foil']=='himself' else 2
                objects={'margin':mixed[...,0]-mixed[...,foil],'readers':mixed-mixed.mean(-1,keepdims=True)}
                scores={k:C.analyse(v) for k,v in objects.items()}
                interaction={k:S.ratio(np.asarray(scores[k]['interaction']),
                    v[0,0] if np.linalg.norm(v[0,0])>=np.linalg.norm(v[1,1]) else v[1,1]) for k,v in objects.items()}
                a,b=[s[2][:,w['common_positions']].reshape(-1) for s,w in zip(states,worlds,strict=True)]
                write_cos=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
                reports.append({'world_id':worlds[0]['world_id'],'foil':worlds[0]['foil'],'bridges':bridges,
                    'instrument_passed':all(s[0]['instrument']['passed'] for s in states) and bool(np.isfinite(effects).all()),
                    'effects':effects.tolist(),'removed_logits':[z.tolist() for z in removed],
                    'native_logits':[s[0]['native_logits'] for s in states],'cross_logits':[z.tolist() for z in swaps],
                    'scores':scores,'interaction_ratios':interaction,'common_write_cosine':write_cos,
                    'fronted_original_write_norm_ratio':float(b.norm()/a.norm().clamp_min(1e-30)),
                    'producer_following_passed':all(s['producer_following_passed'] for s in scores.values()),
                    'reader_following_passed':all(s['reader_following_passed'] for s in scores.values()),
                    'small_interaction_passed':all(S.passed(x,.10) for x in interaction.values())})
    finally:count_handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[256,4096] and all(r['instrument_passed'] for r in reports),
        'pred_b_producer_following':all(r['producer_following_passed'] for r in reports),
        'pred_c_reader_following':all(r['reader_following_passed'] for r in reports),
        'pred_d_small_interaction':all(r['small_interaction_passed'] for r in reports)}
    result={'terminal':'common_interchange_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'True component replacement on opened paired layouts; producer includes value production and L9 routing, recipient includes background and downstream readers. Not independent extraction.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','producer_following_passed','reader_following_passed','small_interaction_passed','interaction_ratios','common_write_cosine')}|
        {'producer_margin_errors':r['scores']['margin']['producer_following_errors'],'reader_margin_errors':r['scores']['margin']['reader_following_errors']} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
