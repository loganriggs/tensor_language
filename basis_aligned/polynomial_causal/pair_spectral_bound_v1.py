"""Exact scalar-contrast error norms; frozen weights and descriptive native split."""
import json
import signal
import time
from datetime import datetime, timezone
import torch
from retained_objective_context_v1 import Contexts, P


@torch.no_grad()
def main():
    out=P/'PAIR_SPECTRAL_BOUND_V1_RESULT.json';assert not out.exists()
    signal.alarm(120);torch.set_num_threads(2);start=time.monotonic();model=Contexts()
    rowdata=json.loads((P/'MINIMAX_FRESH_CACHE_V1_ROWS.json').read_text())['rows'][24:]
    pairs=list(zip(model.ids[::2],model.ids[1::2]))
    rows=[]
    for name,prefix in [('SPARSE_INTERACTION_EXECUTOR_V1','MINIMAX_FRESH_BASELINE_V1'),
                        ('MINIMAX_ROW_SUPPORT_V1','MINIMAX_FRESH_FIT_V1')]:
        error=model.decode(name)-model.tensor
        native=json.loads((P/(prefix+'_RESULT.json')).read_text())
        difference=torch.tensor(native['predicted_own_effects'],dtype=torch.float64)[24:]-torch.tensor(native['reference_own_effects'],dtype=torch.float64)[24:]
        stats=[]
        for index,pair in enumerate(pairs):
            matrix=error[2*index]-error[2*index+1]
            u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
            witness=float(u[:,0]@matrix@vh[0])
            selected=torch.tensor([(r['uk_id'],r['us_id'])==pair for r in rowdata])
            stats.append(dict(pair=list(pair),spectral_bound=float(s[0]),
                              witness_relative_error=abs(witness-float(s[0]))/float(s[0]),
                              fresh_rows=int(selected.sum()),fresh_selected_error_energy=float(difference[selected].square().sum())))
        rows.append(dict(name=name,pairs=stats))
    ratios=[b['spectral_bound']/a['spectral_bound'] for a,b in zip(rows[0]['pairs'],rows[1]['pairs'])]
    result=dict(rows=rows,bound_ratios=ratios,pred_a=all(p['witness_relative_error']<=1e-10 for r in rows for p in r['pairs']),
                pred_b=max(ratios)<=1,seconds=time.monotonic()-start,
                scope='Exact scalar-reader absolute bilinear error norms, not native relative-error guarantees. Descriptive pair split, no fitting.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
