"""Post-native class cancellation audit and next operation's certificate controls."""
import json
from pathlib import Path
import numpy as np
from reader_energy_certificate_v1 import controls
from induction_context_transport_v2 import digest


def main():
    p=Path(__file__).parent;source=p/'CALIBRATION_TOKEN_CONTEXT_V1_RESULT.json';r=json.loads(source.read_text());rng=np.random.default_rng(9111218);reports={}
    for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
        rows=[x for x in r['per_row'] if x['split']==cohort];n=len(rows);draw=rng.integers(n,size=(4000,n));report={}
        for arm in ('no_interaction','context'):
            byclass={}
            for cls in ('all','frequent','rare'):
                counts=np.array([256 if cls=='all' else x[cls+'_count'] for x in rows]);delta=np.array([x['ce_sums'][arm][cls]-x['ce_sums']['native'][cls] for x in rows]);estimate=float(delta.sum()/counts.sum())
                assert abs(estimate-r['reports'][cohort]['ce_effects'][arm][cls])<1e-12
                byclass[cls]=dict(estimate=estimate,ci95=np.quantile(delta[draw].sum(1)/counts[draw].sum(1),[.025,.975]).tolist(),positive_rows=int((delta>0).sum()),rows=n)
            f=sum(x['frequent_count'] for x in rows)/(256*n);rare=byclass['rare']['estimate'];freq=byclass['frequent']['estimate'];replay=f*freq+(1-f)*rare
            assert abs(replay-byclass['all']['estimate'])<1e-12
            byclass['frequent_fraction']=f;byclass['zero_mean_at_frequent_fraction']=rare/(rare-freq) if rare!=freq else None
            report[arm]=byclass
        reports[cohort]=report
    out=dict(experiment='calibration_token_context_audit_v1',model_forwards=0,post_result=True,registered_gates_unchanged=True,reports=reports,bootstrap_draws=4000,seed=9111218,uncertainty_scope='row clusters; FineWeb original document grouping unknown',next_operation_controls=controls(),source_sha256=digest(__file__),input_result_sha256=digest(source))
    with (p/'CALIBRATION_TOKEN_CONTEXT_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
