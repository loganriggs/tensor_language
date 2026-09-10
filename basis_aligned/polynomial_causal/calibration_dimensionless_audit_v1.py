"""Post-native uncertainty and explicit local producer dependency closure."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from calibration_two_readers_v1 import EPS32,fold,scalar
from calibration_dimensionless_donor_v1 import units,read_dimensionless
from induction_context_transport_v2 import digest


def closure_control():
    rng=torch.Generator().manual_seed(9111206)
    rand=lambda *s:torch.randn(*s,generator=rng,dtype=torch.float64)
    m=SimpleNamespace(Left=SimpleNamespace(weight=rand(11,6)),Right=SimpleNamespace(weight=rand(11,6)),Down=SimpleNamespace(weight=rand(6,11)),Down_bias=rand(6))
    a=rand(13,6);w=rand(6);w=w/w.norm();U=rand(17,6)
    rho=(a.square().mean(-1)+EPS32).sqrt();u=a/rho[:,None]
    native_m=((u@m.Left.weight.T)*(u@m.Right.weight.T))@m.Down.weight.T+m.Down_bias
    Q,beta=fold(m,w);q=scalar(u,Q,beta)
    g=rho[:,None]*u+native_m-q[:,None]*w
    h=a+native_m;direct_g=h-(native_m@w/(w@w))[:,None]*w
    tau=q/units(g);z=read_dimensionless(g,tau,w,U)
    native_z=30*torch.tanh((h@U.T)/(30*(h.square().mean(-1,keepdim=True)+EPS32).sqrt()))
    errors=dict(background=float((g-direct_g).abs().max()),readout=float((z-native_z).abs().max()))
    assert max(errors.values())<1e-10
    # Dropping the MLP complement from R is a live falsifier of hidden dependency.
    wrong_tau=q/units(a)
    wrong=read_dimensionless(g,wrong_tau,w,U)
    missing_complement_error=float((wrong-native_z).norm());assert missing_complement_error>.1
    return dict(passed=True,errors=errors,missing_complement_error_norm=missing_complement_error,producer='u=a/sqrt(mean(a²)+eps); q=u^TQ u+beta; g=rho*u+(I-Pw)M(u); tau=q/sqrt(mean(g²))',polynomial_accounting='g degree<=2 in (u,rho), mean(g²) degree<=4; keep native factored DAG, do not materialize quartic coefficient tensor',scope='Exact-arithmetic local interface control; full MLP complement and upstream input still required, no trained extraction claim')


def main():
    p=Path(__file__).parent;source=p/'CALIBRATION_DIMENSIONLESS_DONOR_V1_RESULT.json';r=json.loads(source.read_text());rng=np.random.default_rng(9111206);reports={}
    for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
        rows=[x for x in r['per_row'] if x['split']==cohort];n=len(rows);draws=rng.integers(n,size=(4000,n));out={}
        for arm in ('mean_tau','donor_tau'):
            delta=np.array([x['ce_sums'][arm]['all']-x['ce_sums']['native']['all'] for x in rows])/256
            estimate=float(delta.mean());assert abs(estimate-r['reports'][cohort]['ce_effects'][arm]['all'])<1e-12
            out[arm]=dict(estimate=estimate,positive_rows=int((delta>0).sum()),rows=n,min_row=float(delta.min()),max_row=float(delta.max()))
            if arm=='mean_tau':out[arm]['ci95']=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        out['adjusted_to_raw_donor_damage_ratio']=r['reports'][cohort]['ce_effects']['donor_tau']['all']/r['reports'][cohort]['ce_effects']['donor']['all']
        out['mean_tau_uncertainty_scope']='row bootstrap; FineWeb document grouping unknown; Pile one row per document'
        out['donor_uncertainty_scope']='descriptive; no independent-row CI because cyclic pairs share rows'
        reports[cohort]=out
    out=dict(experiment='calibration_dimensionless_audit_v1',model_forwards=0,registered_gates_unchanged=True,post_result=True,seed=9111206,bootstrap_draws=4000,reports=reports,closure_control=closure_control(),source_sha256=digest(__file__),input_result_sha256=digest(source))
    with (p/'CALIBRATION_DIMENSIONLESS_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
