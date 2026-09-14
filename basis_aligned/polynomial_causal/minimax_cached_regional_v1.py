"""Frozen minimax support validation; same historical transfer bars as prior fit."""
import json
import signal
import numpy as np
from producer_fit_cached_regional_v1 import main, P


if __name__=='__main__':
    signal.alarm(120)
    main('MINIMAX_ROW_SUPPORT_V1','MINIMAX_CACHED_REGIONAL_V1')
    baseline=json.loads((P/'PRODUCER_BASELINE_CACHED_REGIONAL_V1_RESULT.json').read_text())
    fit=json.loads((P/'MINIMAX_CACHED_REGIONAL_V1_RESULT.json').read_text())
    assert baseline['cache_sha256']==fit['cache_sha256']
    reference=np.array(fit['reference_own_effects'])
    be=np.array(baseline['predicted_own_effects'])-reference
    fe=np.array(fit['predicted_own_effects'])-reference
    improvement=float(1-(fe@fe)/(be@be))
    ratios=[f['own_effect_error']/b['own_effect_error'] for b,f in zip(baseline['cells'],fit['cells'])]
    result=dict(relative_squared_error_improvement=improvement,group_error_ratios=ratios,
                pred_d=improvement>=.05 and max(ratios)<=1.01,
                scope='Frozen minimax support, historical native cache, no fit/freshOOD/adoption.')
    out=P/'MINIMAX_CACHED_COMPARISON_V1_RESULT.json';assert not out.exists()
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));signal.alarm(0)
