"""Post-result uncertainty and exact RMS-aware constant-term separation."""
import json
from pathlib import Path
import numpy as np
import torch
from calibration_two_readers_v1 import EPS32,scalar
from calibration_scalar_path_v1 import controls
from induction_context_transport_v2 import digest


def main():
    p=Path(__file__).parent;source=p/'CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json'
    result=json.loads(source.read_text());reports={};rng=np.random.default_rng(9111153)
    for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
        rows=[r for r in result['per_row'] if r['split']==cohort];n=len(rows)
        draws=rng.integers(n,size=(4000,n));local={}
        for arm in ('mean','donor','remove_A','remove_B','remove_ref'):
            delta=np.array([r['ce_sums'][arm]['all']-r['ce_sums']['native']['all'] for r in rows])/256
            estimate=float(delta.mean());assert abs(estimate-result['reports'][cohort]['ce_effects'][arm]['all'])<1e-12
            local[arm]=dict(estimate=estimate,positive_row_count=int((delta>0).sum()),rows=n,min_row_effect=float(delta.min()),max_row_effect=float(delta.max()))
            if arm=='mean':local[arm]['ci95']=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        local['scope']='Mean replacement: row-cluster bootstrap; FineWeb original document grouping unknown. Donor effects descriptive only: cyclic edges share source/recipient rows.'
        reports[cohort]=local
    producer=p/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt';x=torch.load(producer,weights_only=True,map_location='cpu')
    Q=x['Q'].double();beta=x['beta'].double();D=len(Q);alpha=Q.trace()/D;Q0=Q-alpha*torch.eye(D,dtype=torch.float64)
    g=torch.Generator().manual_seed(9111153);raw=torch.randn(12,D,generator=g,dtype=torch.float64)
    raw=raw*torch.logspace(-5,2,12,dtype=torch.float64)[:,None]
    norm2=raw.square().mean(-1);u=raw/(norm2[:,None]+EPS32).sqrt()
    direct=scalar(u,Q,beta)
    split=scalar(u,Q0,beta+Q.trace())-Q.trace()*EPS32/(norm2+EPS32)
    relative=float((direct-split).norm()/direct.norm());assert relative<1e-12
    split_axes=torch.load(p/'CALIBRATION_STABILITY_CONTEXT_V1_AXES.pt',weights_only=True,map_location='cpu')
    angles={k:float(split_axes[k]@x['w']) for k in ('A','B')}
    math=dict(identity='q=beta+tr(Q)+u^T(Q-tr(Q)/D*I)u-tr(Q)*eps/(mean(x^2)+eps), u=RMS(x)',isotropic_alpha=float(alpha),trace_Q=float(Q.trace()),beta=float(beta),constant_limit=float(beta+Q.trace()),fit_mean_q=result['fit_stats']['mean'],traceless_matrix_trace=float(Q0.trace()),quadratic_split_relative_error=relative,split_axes_cosine_to_original=angles,scope='Algebraic decomposition only. A constant limit is not the empirical mean unless the anisotropic quadratic averages to zero; no such distributional assumption is made. No rank approximation.')
    out=dict(experiment='calibration_context_math_audit_v1',model_forwards=0,controls=controls(),post_result=True,registered_gates_unchanged=True,bootstrap_draws=4000,seed=9111153,reports=reports,mathematics=math,source_sha256=digest(__file__),input_result_sha256=digest(source),producer_sha256=digest(producer))
    with (p/'CALIBRATION_CONTEXT_MATH_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
