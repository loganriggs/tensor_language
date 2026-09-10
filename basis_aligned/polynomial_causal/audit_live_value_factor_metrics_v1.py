"""Timed shared-scorer replay and exact spill spectrum of the livevalue result."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from factorial_effect_metrics_v1 import measure


def main():
    p=Path(__file__).resolve().parent;source=p/'THIRD_NOUN_L9_VALUE_LIVE_V1_RESULT.json'
    stages={};started=datetime.now(timezone.utc).isoformat();tic=time.perf_counter()
    d=json.loads(source.read_text());assert d['predictions']['pred_a_instrument']
    stages['input_read_seconds']=time.perf_counter()-tic;tic=time.perf_counter()
    reports=[];max_replay=0.
    for w,ref in enumerate(d['reports']):
        r=measure(d['corners'],d['margins']['native'][w],d['margins']['edited'][w],prediction=d['margins']['direct'][w])
        for k in ('live_natural_mixed_projection','nonmixed_to_mixed_margin_ratio','direct_margin_relative_error'):
            max_replay=max(max_replay,abs(r[k]-ref[k]))
        coef=np.array(r['effect_coefficients']);spill=r['spill_masks']
        mask=max(spill,key=lambda i:coef[i]**2)
        r['largest_spill_term']=''.join(x for i,x in enumerate('csoah') if mask&(1<<i)) or 'constant'
        r['largest_spill_fraction']=float(coef[mask]**2/np.sum(coef[spill]**2))
        r['world_id']=ref['world_id'];reports.append(r)
    stages['scientific_analysis_seconds']=time.perf_counter()-tic;tic=time.perf_counter()
    assert max_replay<1e-12 and max(r['parseval_error'] for r in reports)<1e-12
    # Verify the target excludes main effects while retaining contextual interactions.
    corners=np.array(d['corners']);c,s,o,a,h=corners.T
    native=2*o+.5*o*h+.25*c*o*h
    change=.1*o*h+.03*o
    control=measure(corners,native,native-change)
    assert abs(control['nonmixed_to_mixed_margin_ratio']-.3)<1e-12
    assert abs(control['target_effect_rms']-.1)<1e-12
    assert control['parseval_error']<1e-12
    stages['validation_seconds']=time.perf_counter()-tic
    result={'passed':True,'reports':reports,'max_replay_error':max_replay,'model_forwards':0,
        'started_utc':started,'stage_seconds':stages,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256((p/'factorial_effect_metrics_v1.py').read_bytes()).hexdigest(),
        'scope':'Post-outcome exact finite-table spectrum and reusable scorer. Stage timings cover CPU execution, not earlier human/agent design or documentation time. No semantic unit or OOD circuit promoted.'}
    with (p/'LIVE_VALUE_FACTOR_METRICS_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'max_replay_error':max_replay,'stage_seconds':stages,
        'spill':[{'world_id':r['world_id'],'largest_spill_term':r['largest_spill_term'],
                  'largest_spill_fraction':r['largest_spill_fraction'],'remaining_natural_mixed_ratio':r['remaining_natural_mixed_ratio']} for r in reports]},indent=2))


if __name__=='__main__':main()
