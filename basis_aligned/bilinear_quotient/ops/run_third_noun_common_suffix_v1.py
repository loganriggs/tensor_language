#!/usr/bin/env python3
# BQGATE: causal common-position screen, 320 forwards/5120 sequences; no fits.
"""A instrument; B common fidelity .10; C noun fidelity .10; D composition .10;
E common spill .25; F common gender .25. All worlds, no position/gain selection.
All 545902902 native parameters retained. Null closes unqualified suffix swap.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_write_position_executor_v1 as E
import attention_write_factorial_executor_v1 as S
import mixed_state_projector_v1 as Q
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_COMMON_SUFFIX_V1_BINDING.json';OUT=POLY/'THIRD_NOUN_COMMON_SUFFIX_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parents=[json.loads((POLY/name).read_text()) for name in ('THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json','THIRD_NOUN_WRITE_STRUCTURE_V1_RESULT.json')]
    lookup={r['world_id']:r for p in parents for r in p['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':320,'sequences':5120,'fits':0,'worlds':len(source['worlds'])}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    q_np=Q.projector(source['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        for world in source['worlds']:
            report=E.run_world(backend,world,q,source['reader_ids']);old=lookup[world['world_id']]
            bridges={'native':S.bridge(report['native_logits'],old['native_logits']),
                     'full_mixed':S.bridge(report['arm_logits']['full'],old['arm_logits']['mixed'])}
            report['instrument_passed'] &= all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
            report['bridges']=bridges;score=S.score(source['corners'],q_np,report)
            z0=np.array(report['native_logits']);zf=np.array(report['arm_logits']['full']);zn=np.array(report['arm_logits']['spill'])
            foil=1 if world['foil']=='himself' else 2
            ef=q_np@((z0[:,0]-z0[:,foil])-(zf[:,0]-zf[:,foil]))
            en=q_np@((z0[:,0]-z0[:,foil])-(zn[:,0]-zn[:,foil]))
            noun_error=S.ratio(ef-en,ef);m=score['metrics']['mixed']
            report.update(score,common_fidelity_passed=score['fidelity_passed'],noun_fidelity_error=noun_error,
                noun_fidelity_passed=S.passed(noun_error,.10),
                common_selectivity_passed=m is not None and m['nonmixed_to_mixed_margin_ratio']<=.25)
            reports.append(report)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[320,5120] and all(r['instrument_passed'] for r in reports),
           'pred_b_common_fidelity':all(r['common_fidelity_passed'] for r in reports),
           'pred_c_noun_fidelity':all(r['noun_fidelity_passed'] for r in reports),
           'pred_d_composition':all(r['composition_passed'] for r in reports),
           'pred_e_common_selectivity':all(r['common_selectivity_passed'] for r in reports),
           'pred_f_common_gender':all(r['gender_passed'] for r in reports)}
    result={'terminal':'common_suffix_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Position partition on opened layouts. Common to/action interface versus remaining noun position; no cross-layout swap or independent extracted producer.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','mixed_fidelity_error','noun_fidelity_error','composition_margin_error','composition_reader_error','common_selectivity_passed','mixed_gender_ratio')} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
