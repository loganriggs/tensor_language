"""Post-result row-cluster uncertainty; preserves registered point-estimate gates."""
import json
from pathlib import Path
import numpy as np
import torch
from induction_context_transport_v2 import digest


def main():
    p=Path(__file__).parent;source=p/'CALIBRATION_TWO_READERS_V2_RESULT.json'
    r=json.loads(source.read_text());rng=np.random.default_rng(9111135);reports={}
    for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
        rows=[x for x in r['per_row'] if x['split']==cohort];n=len(rows)
        draw=rng.integers(n,size=(4000,n));cohort_out={}
        for cls in ('frequent','rare'):
            counts=np.array([x[cls+'_count'] for x in rows]);den=counts[draw].sum(1);assert np.all(den>0)
            totals={arm:np.array([x['ce_sums'][arm][cls] for x in rows]) for arm in rows[0]['ce_sums']}
            effects={arm:totals[arm]-totals['native'] for arm in totals if arm!='native'}
            effects['interaction']=effects['joint']-effects['numerator']-effects['denominator']
            class_out={}
            for arm,delta in effects.items():
                estimate=float(delta.sum()/counts.sum())
                if arm!='interaction':assert abs(estimate-r['reports'][cohort]['ce_effects'][arm][cls])<1e-12
                sampled=delta[draw].sum(1)/den
                class_out[arm]=dict(estimate=estimate,ci95=np.quantile(sampled,[.025,.975]).tolist(),
                                    positive_row_fraction=float(np.mean(delta[counts>0]>0)))
            cohort_out[cls]=class_out
        cohort_out['resampling_unit']='cached row; original document grouping unknown' if cohort=='FW_HOLDOUT' else 'one sampled Pile document'
        reports[cohort]=cohort_out
    a=torch.load(p/'CALIBRATION_TWO_READERS_V1_PRODUCER.pt',map_location='cpu',weights_only=True)
    b=torch.load(p/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt',map_location='cpu',weights_only=True)
    repeat={k:bool(torch.equal(a[k],b[k])) for k in ('w','Q','beta','random','top20')}
    assert all(repeat.values())
    out=dict(experiment='calibration_two_readers_bootstrap_v1',model_forwards=0,
             bootstrap_draws=4000,seed=9111135,post_result_diagnostic=True,
             original_gates_preserved=True,reports=reports,producer_repeat_bitwise=repeat,
             source_sha256=digest(__file__),input_result_sha256=digest(source))
    with (p/'CALIBRATION_TWO_READERS_BOOTSTRAP_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'producer_repeat_bitwise':repeat,'reports':{c:{cl:{a:reports[c][cl][a] for a in ('joint','interaction')} for cl in ('frequent','rare')} for c in reports}},indent=2))


if __name__=='__main__':main()
