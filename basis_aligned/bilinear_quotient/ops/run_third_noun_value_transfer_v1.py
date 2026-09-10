#!/usr/bin/env python3
# BQGATE: frozen value operator, fresh lexical/reader panels, seventy forwards, no fits.
"""A instrument; B male transfer; C female transfer; D gender selectivity."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import live_value_factorial_executor_v1 as E
import mixed_state_projector_v1 as Q
from factorial_effect_metrics_v1 import measure
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_VALUE_TRANSFER_V1_BINDING.json';OUT=POLY/'THIRD_NOUN_VALUE_TRANSFER_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_VALUE_TRANSFER_V1_ROWS.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_L9_VALUE_LIVE_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':70,'sequences':1120,'fits':0,'audit':source['audit']}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    tic=time.perf_counter();counts=[0,0]
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    q_np=Q.projector(source['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    try:
        reference=E.run_world(backend,source['reference'],q,source['reader_ids'],identity=True)
        reference_bridges={}
        for arm in ('native','edited'):
            z=np.array(reference[f'{arm}_logits']);observed=z[:,0]-z[:,1];target=np.array(parent['margins'][arm][0])
            reference_bridges[arm]={'max_abs':float(np.max(abs(observed-target))),
                'relative':float(np.linalg.norm(observed-target)/np.linalg.norm(target))}
        reports=[]
        for world in source['worlds']:
            report=E.run_world(backend,world,q,source['reader_ids'])
            z0=np.array(report['native_logits']);z1=np.array(report['edited_logits']);foil=1 if world['foil']=='himself' else 2
            m0=z0[:,0]-z0[:,foil];m1=z1[:,0]-z1[:,foil]
            natural_rms=float(np.sqrt(np.mean((q_np@m0)**2)));effect_norm=float(np.linalg.norm(q_np@(m0-m1)))
            if natural_rms>0 and effect_norm>0:
                metrics=measure(source['corners'],m0,m1)
                transfer=natural_rms>=.05 and metrics['live_natural_mixed_projection']>=.05 and metrics['nonmixed_to_mixed_margin_ratio']<=.25
            else:metrics=None;transfer=False
            dz=z0-z1;number=dz[:,0]-(dz[:,1]+dz[:,2])/2;gender=dz[:,1]-dz[:,2]
            n=float(np.linalg.norm(q_np@number));g=float(np.linalg.norm(q_np@gender))
            ratio=g/n if n>0 else None
            report.update(native_mixed_margin_rms=natural_rms,metrics=metrics,transfer_passed=transfer,
                gender_to_number_effect_ratio=ratio,gender_selectivity_passed=ratio is not None and ratio<=.25)
            reports.append(report)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    a=restored and counts==[70,1120] and reference['instrument']['passed'] and all(r['instrument']['passed'] for r in reports) and all(
        b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in reference_bridges.values())
    preds={'pred_a_instrument':a,'pred_b_male_transfer':all(r['transfer_passed'] for r in reports if r['panel']=='male'),
           'pred_c_female_transfer':all(r['transfer_passed'] for r in reports if r['panel']=='female'),
           'pred_d_gender_selectivity':all(r['gender_selectivity_passed'] for r in reports)}
    result={'terminal':'value_transfer_complete' if a else 'invalid','predictions':preds,'reports':reports,
        'reference':reference,'reference_bridges':reference_bridges,'reader_names':source['reader_names'],'corners':source['corners'],
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Frozen native-input-dependent value operator on fresh lexical and answer-reader panels. Not independent four-corner production, unseen training-domain proof, or unrelated-task circuit selectivity.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'reference_bridges':reference_bridges,'wall_seconds':result['wall_seconds'],
        'worlds':[{k:r[k] for k in ('world_id','native_mixed_margin_rms','transfer_passed','gender_to_number_effect_ratio')}|
                 {'projection':None if r['metrics'] is None else r['metrics']['live_natural_mixed_projection'],
                  'spill':None if r['metrics'] is None else r['metrics']['nonmixed_to_mixed_margin_ratio']} for r in reports]}));assert a


if __name__=='__main__':main()
