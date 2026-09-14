"""Frozen explicit-pair sparse native contrast tests; prior own/compact/comparison bars."""
import json
import signal
import numpy as np
from producer_fit_cached_regional_v1 import main,P


if __name__=='__main__':
    signal.alarm(120)
    comparisons=[]
    for cache,rows,prefix,baseline,start in [
        ('COMPOSED_LAST_BLOCK_STATES_V1',None,'PAIR_FRAME_HISTORICAL_V1','PRODUCER_BASELINE_CACHED_REGIONAL_V1',0),
        ('MINIMAX_FRESH_CACHE_V1','MINIMAX_FRESH_CACHE_V1','PAIR_FRAME_FRESH_V1','MINIMAX_FRESH_BASELINE_V1',1)]:
        main('PAIR_FRAME_SPARSE_V1',prefix,cache,rows)
        base=json.loads((P/(baseline+'_RESULT.json')).read_text())
        fit=json.loads((P/(prefix+'_RESULT.json')).read_text())
        ref=np.array(fit['reference_own_effects'])[24*start:]
        be=np.array(base['predicted_own_effects'])[24*start:]-ref
        fe=np.array(fit['predicted_own_effects'])[24*start:]-ref
        improvement=float(1-(fe@fe)/(be@be))
        ratios=[f['own_effect_error']/b['own_effect_error'] for b,f in zip(base['cells'][start:],fit['cells'][start:])]
        cells=fit['cells'][start:]
        comparisons.append(dict(panel=cache,relative_squared_error_improvement=improvement,group_error_ratios=ratios,
                                pred_b=all(c['own_effect_error']<=.1 for c in cells),
                                pred_c=all(c['compact_effect_error']<=.05 and c['material_compact_sign_reversals']==0 and c['material_own_sign_reversals']==0 for c in cells),
                                pred_d=improvement>=.05 and max(ratios)<=1.01))
    out=P/'PAIR_FRAME_REGIONAL_V1_RESULT.json';assert not out.exists()
    with out.open('x') as f:json.dump(dict(panels=comparisons,scope='Frozen weight-only pair-frame sparse program, native conditional task contrasts; no fit/adoption or corpusOOD.'),f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(comparisons))
