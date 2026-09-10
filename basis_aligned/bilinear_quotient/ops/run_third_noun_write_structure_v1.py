#!/usr/bin/env python3
# BQGATE: frozen mixed-write structure transfer; 266 forwards/4256 sequences; zero fits.
"""A instrument; B transfer .05/.25; C fidelity .10; D gender .25; E composition .10.

Every world must pass separately; nulls retained with no operator/text retuning.
All 545902902 parameters retained, zero structural saving, native inputs/suffix.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import circuit_fast_screen_producer as P
import attention_write_factorial_executor_v1 as E
import mixed_state_projector_v1 as Q
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_WRITE_STRUCTURE_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_WRITE_STRUCTURE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_WRITE_STRUCTURE_V1_ROWS.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':266,'sequences':4256,'fits':0,'audit':source['audit']}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    q_np=Q.projector(source['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    counts=[0,0];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        reference=E.run_world(backend,source['reference'],q,source['reader_ids'],full_replay=True)
        old=parent['reports'][0];assert old['world_id']==reference['world_id']
        bridges={'native':E.bridge(reference['native_logits'],old['native_logits'])}
        bridges.update({k:E.bridge(reference['arm_logits'][k],old['arm_logits'][k]) for k in ('full','mixed','spill')})
        assert reference['instrument_passed'] and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values()), 'Reference must pass before new outcomes'
        reports=[]
        for world in source['worlds']:
            report=E.run_world(backend,world,q,source['reader_ids'])
            report.update(E.score(source['corners'],q_np,report));reports.append(report)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[266,4256] and all(r['instrument_passed'] for r in reports),
           'pred_b_structure_transfer':all(r['transfer_passed'] for r in reports),
           'pred_c_fidelity':all(r['fidelity_passed'] for r in reports),
           'pred_d_gender':all(r['gender_passed'] for r in reports),
           'pred_e_composition':all(r['composition_passed'] for r in reports)}
    result={'terminal':'structure_transfer_complete' if preds['pred_a_instrument'] else 'invalid',
        'predictions':preds,'reports':reports,'reference':reference,'reference_bridges':bridges,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],
        'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Frozen mixed attention-write intervention on structurally changed text; native four-corner inputs and full native suffix retained, no independent extraction.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'reference_bridges':bridges,
        'worlds':[{k:r[k] for k in ('world_id','native_mixed_rms','transfer_passed','fidelity_passed','gender_passed','composition_passed','mixed_fidelity_error','mixed_gender_ratio')}|
                  {'projection':None if r['metrics']['mixed'] is None else r['metrics']['mixed']['live_natural_mixed_projection'],
                   'spill':None if r['metrics']['mixed'] is None else r['metrics']['mixed']['nonmixed_to_mixed_margin_ratio']} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
