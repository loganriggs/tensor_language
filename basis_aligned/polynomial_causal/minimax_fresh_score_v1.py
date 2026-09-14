"""Frozen fresh-syntax comparison, preregistered before managed native capture."""
import json
import signal
import numpy as np
from producer_fit_cached_regional_v1 import main, P


if __name__=='__main__':
    signal.alarm(120)
    for name,prefix in [('SPARSE_INTERACTION_EXECUTOR_V1','MINIMAX_FRESH_BASELINE_V1'),
                        ('MINIMAX_ROW_SUPPORT_V1','MINIMAX_FRESH_FIT_V1')]:
        main(name,prefix,cache_prefix='MINIMAX_FRESH_CACHE_V1',rows_prefix='MINIMAX_FRESH_CACHE_V1')
    baseline=json.loads((P/'MINIMAX_FRESH_BASELINE_V1_RESULT.json').read_text())
    fit=json.loads((P/'MINIMAX_FRESH_FIT_V1_RESULT.json').read_text())
    assert baseline['cache_sha256']==fit['cache_sha256']
    reference=np.array(fit['reference_own_effects'])[24:]
    be=np.array(baseline['predicted_own_effects'])[24:]-reference
    fe=np.array(fit['predicted_own_effects'])[24:]-reference
    improvement=float(1-(fe@fe)/(be@be))
    ratios=[f['own_effect_error']/b['own_effect_error'] for b,f in zip(baseline['cells'][1:],fit['cells'][1:])]
    result=dict(relative_squared_error_improvement=improvement,fresh_group_error_ratios=ratios,
                pred_b=all(c['own_effect_error']<=.1 for c in fit['cells'][1:]),
                pred_c=all(c['compact_effect_error']<=.05 and c['material_compact_sign_reversals']==0 and c['material_own_sign_reversals']==0 for c in fit['cells'][1:]),
                pred_d=improvement>=.05 and max(ratios)<=1.01,
                scope='96freshsyntax prefixes only in comparison,24anchors separately reported in scorers; existing endpoints/cities. No corpusOOD or discoveredselectivity. Historical Dfailure retained.')
    out=P/'MINIMAX_FRESH_COMPARISON_V1_RESULT.json';assert not out.exists()
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));signal.alarm(0)
