#!/usr/bin/env python3
# BQGATE: raw mixed input removal with native RMS recomputation; 320 forwards/5120 sequences.
"""A instrument; B raw necessity .10; C regeneration .10, both readouts every world.
All545902902nativeweights retained; no independently generated residual state.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mature_value_route_native_v2 as M
import rms_mixed_input_partition_v1 as R
import symmetric_factor_interaction_v1 as F
import attention_output_delta_v1 as A
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MATURE_VALUE_RAW_REMOVAL_V1_BINDING.json';OUT=POLY/'MATURE_VALUE_RAW_REMOVAL_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_COMMON_SUFFIX_V1_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':320,'sequences':5120,'fits':0}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    projections=F.projectors(source['corners']).to('cuda');q=projections[3];q_np=q.cpu().numpy()
    attn=backend.model.transformer.h[9].attn;counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for world in source['worlds']:
                state=M.capture(backend,world,q,projections[0],source['reader_ids'])
                x=state['raw_inputs'];u=state['normalized_inputs']
                assert state['instrument']['passed'] and state['recurrence_bitwise']
                raw_prime=(x.double()-torch.einsum('ij,jtd->itd',q,x.double())).to(x)
                raw_impurity=float(torch.einsum('ij,jtd->itd',q,raw_prime.double()).norm()/x.double().norm())
                assert raw_impurity<=1e-6
                uprime=torch.nn.functional.rms_norm(raw_prime,(raw_prime.size(-1),)).double()
                def compile_common(normalized):
                    mixed=torch.einsum('ij,jtd->itd',q,normalized)
                    values=(mixed[:,-3:]@attn.c_v.weight.double().T).view(32,3,9,128)
                    return M.contract(state['pattern'],values,attn.c_proj.weight.double(),1.65625)
                compiled=compile_common(u);regenerated=compile_common(uprime)
                observed=state['observed_common'][:,-2:]
                component_bridge=float((compiled-observed).norm()/observed.norm().clamp_min(1e-30))
                assert component_bridge<=1e-4
                # Subtract C-Cprime to install B+Cprime; the baseline is B=A-C.
                components={'full':observed,'raw':observed-regenerated,'identity':observed-compiled}
                arms={};foil=1 if world['foil']=='himself' else 2
                for arm,component in components.items():
                    delta=torch.zeros_like(state['native_write'],dtype=torch.float64);delta[:,-2:]=component
                    chunks=[];audit=[]
                    for start in (0,16):
                        rows=world['rows'][start:start+16]
                        batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                            (source['reader_ids'][0],)*16,(source['reader_ids'][foil],)*16,(world['length']-1,)*16)
                        with A.subtract(attn,state['native_write'][start:start+16],delta[start:start+16],audit):
                            with G.capture(backend.model) as captured:backend.native(batch,capture=False)
                        chunks.append(G.endpoint_logits(captured['raw_logits'],batch)[:,source['reader_ids']].double())
                    arms[arm]=torch.cat(chunks).cpu().tolist()
                old=lookup[world['world_id']]
                bridges={'native':S.bridge(state['native_logits'],old['native_logits']),
                         'full':S.bridge(arms['full'],old['arm_logits']['mixed']),
                         'identity':S.bridge(arms['identity'],state['native_logits'])}
                z0=np.array(state['native_logits']);effect={k:q_np@(z0-np.asarray(z)) for k,z in arms.items()}
                full=effect['full'];raw=effect['raw'];d=raw-full
                def norms(a,b):
                    return {'margin':S.ratio(a[:,0]-a[:,foil],b[:,0]-b[:,foil]),
                            'readers':S.ratio(a-a.mean(-1,keepdims=True),b-b.mean(-1,keepdims=True))}
                errors=norms(d,full);remaining=norms(raw,full)
                report={'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'native_logits':state['native_logits'],'arm_logits':arms,'bridges':bridges,
                    'raw_mixed_impurity_relative':raw_impurity,'component_bridge_relative':component_bridge,
                    'regenerated_write_norm_ratio':float(regenerated.norm()/observed.norm()),
                    'fidelity_errors':errors,'intervention_effect_ratios':remaining,
                    'raw_necessity_passed':all(S.passed(v,.10) for v in errors.values()),
                    'regeneration_passed':all(S.passed(v,.10) for v in remaining.values()),
                    'instrument_passed':all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values()) and all(np.isfinite(z).all() for z in arms.values())}
                reports.append(report)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[320,5120] and all(r['instrument_passed'] for r in reports),
        'pred_b_raw_necessity':all(r['raw_necessity_passed'] for r in reports),
        'pred_c_regeneration':all(r['regeneration_passed'] for r in reports)}
    result={'terminal':'raw_mixed_removal_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Raw mixed input removed and native RMS recomputed inside the partial value producer; native routing/background retained. No independent residual production or new OOD evidence.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','raw_necessity_passed','regeneration_passed','fidelity_errors')} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
