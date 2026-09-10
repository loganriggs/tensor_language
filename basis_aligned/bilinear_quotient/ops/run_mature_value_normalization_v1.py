#!/usr/bin/env python3
# BQGATE: raw inherited versus RMS-generated mixed input; 320 forwards/5120 sequences.
"""A instrument; B raw fidelity .10; C normalization fidelity .10;
D composition .10 on both readouts, every world. No rank/norm/head selection.
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
BINDING=POLY/'MATURE_VALUE_NORMALIZATION_V1_BINDING.json';OUT=POLY/'MATURE_VALUE_NORMALIZATION_V1_RESULT.json'
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
                epsilon=torch.finfo(state['raw_inputs'].dtype).eps
                parts=R.partition(state['raw_inputs'],projections,epsilon)
                u=state['normalized_inputs'];norm_bridge=float((parts['normalized']-u).norm()/u.norm().clamp_min(1e-30))
                purity=max(float((parts[k]-torch.einsum('ij,jtd->itd',q,parts[k])).norm()/parts['full'].norm().clamp_min(1e-30)) for k in ('raw','normalization'))
                assert state['instrument']['passed'] and state['recurrence_bitwise'] and norm_bridge<=1e-6 and parts['closure_relative']<=1e-10 and purity<=1e-10
                components={}
                for arm,term in [('full','full'),('mixed','raw'),('spill','normalization')]:
                    values=(parts[term][:,-3:]@attn.c_v.weight.double().T).view(32,3,9,128)
                    components[arm]=M.contract(state['pattern'],values,attn.c_proj.weight.double(),1.65625)
                observed=state['observed_common'][:,-2:]
                component_bridge=float((components['full']-observed).norm()/observed.norm().clamp_min(1e-30))
                assert component_bridge<=1e-4
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
                         'full':S.bridge(arms['full'],old['arm_logits']['mixed'])}
                report={'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'native_logits':state['native_logits'],'arm_logits':arms,
                    'arm_meanings':{'full':'complete normalized mixed input','mixed':'inherited raw-state branch','spill':'normalization-generated branch'},
                    'bridges':bridges,'normalized_input_bridge_relative':norm_bridge,'component_bridge_relative':component_bridge,
                    'recurrence_bitwise':state['recurrence_bitwise'],'input_partition_closure':parts['closure_relative'],
                    'input_partition_purity':purity,'native_epsilon':epsilon,
                    'instrument_passed':all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values()) and all(np.isfinite(z).all() for z in arms.values())}
                report.update(S.score(source['corners'],q_np,report))
                z0=np.array(state['native_logits']);effect={k:q_np@(z0-np.asarray(z)) for k,z in arms.items()}
                errors={}
                for arm in ('mixed','spill'):
                    d=effect[arm]-effect['full'];full=effect['full']
                    errors[arm]={'margin':S.ratio(d[:,0]-d[:,foil],full[:,0]-full[:,foil]),
                                 'readers':S.ratio(d-d.mean(-1,keepdims=True),full-full.mean(-1,keepdims=True))}
                report.update(branch_fidelity_errors=errors,
                    raw_fidelity_passed=all(S.passed(v,.10) for v in errors['mixed'].values()),
                    normalization_fidelity_passed=all(S.passed(v,.10) for v in errors['spill'].values()))
                reports.append(report)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[320,5120] and all(r['instrument_passed'] for r in reports),
        'pred_b_raw_origin':all(r['raw_fidelity_passed'] for r in reports),
        'pred_c_normalization_origin':all(r['normalization_fidelity_passed'] for r in reports),
        'pred_d_composition':all(r['composition_passed'] for r in reports)}
    result={'terminal':'value_normalization_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Inherited versus normalization-generated mixed value branches with native routing retained. No direct raw-state edit, independent residual production, or new OOD evidence.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','raw_fidelity_passed','normalization_fidelity_passed','composition_passed','branch_fidelity_errors')} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
