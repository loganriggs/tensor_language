"""Same-cache score-boundary discriminator, registered board10:20UTC.

Prediction: >=5% worse all12 normalized raw-error energy for frozen fit.
No fitting, reselection or fresh/OOD evidence.
"""
import json
import signal
import time
from datetime import datetime, timezone
import torch
from retained_objective_context_v1 import Contexts, P
from shared_local_fineweb_v1 import digest


@torch.no_grad()
def main():
    out=P/'MINIMAX_FRESH_SCORE_BOUNDARY_V1_RESULT.json'
    assert not out.exists()
    signal.alarm(120);start=time.monotonic();torch.set_num_threads(2)
    model=Contexts()
    path=P/'MINIMAX_FRESH_CACHE_V1_ARTIFACT.pt'
    assert digest(path)==json.loads((P/'MINIMAX_FRESH_CACHE_V1_RESULT.json').read_text())['artifact_sha']
    data=torch.load(path,weights_only=True)
    z=data['linear_parts'][:,3].double()
    state=data['linear_parts'][:,2].double()
    a=torch.linalg.lstsq(model.writer,(state-z).T).solution.T
    eps=torch.finfo(torch.float32).eps
    denominator=(state.square().mean(-1)+eps)*(data['states'][:,2].square().mean(-1)+eps).sqrt()
    rowdata=json.loads((P/'MINIMAX_FRESH_CACHE_V1_ROWS.json').read_text())['rows']
    lookup={v:k for k,v in enumerate(model.ids)}
    uk=torch.tensor([lookup[r['uk_id']] for r in rowdata]);us=torch.tensor([lookup[r['us_id']] for r in rowdata])
    scores=[]
    for name,prefix in [('SPARSE_INTERACTION_EXECUTOR_V1','MINIMAX_FRESH_BASELINE_V1'),
                        ('MINIMAX_ROW_SUPPORT_V1','MINIMAX_FRESH_FIT_V1')]:
        error=model.decode(name)-model.tensor
        value=torch.einsum('oih,ni,nh->no',error,z,a)/denominator[:,None]
        receipt=json.loads((P/(prefix+'_RESULT.json')).read_text())
        capped=torch.tensor(receipt['predicted_own_effects'])-torch.tensor(receipt['reference_own_effects'])
        scores.append(dict(name=name,all12_raw_energy=float(value[24:].square().sum()),
                           all6_raw_contrast_energy=float((value[24:,::2]-value[24:,1::2]).square().sum()/2),
                           selected_raw_contrast_energy=float((value[torch.arange(120),uk]-value[torch.arange(120),us])[24:].square().sum()),
                           selected_softcap_contrast_energy=float(capped[24:].double().square().sum())))
    ratios={key:scores[1][key]/scores[0][key] for key in scores[0] if key!='name'}
    result=dict(utc=datetime.now(timezone.utc).isoformat(),scores=scores,fit_over_baseline=ratios,
                pred_a=ratios['all12_raw_energy']>=1.05,seconds=time.monotonic()-start,
                scope='96fresh syntax rows; frozen native cache score-boundary diagnostic only, no fit or corpusOOD.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
