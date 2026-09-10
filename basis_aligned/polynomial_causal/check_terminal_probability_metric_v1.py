"""Autodiff, finite-dose KL, and softmax-shift controls for terminal curvature."""
import json
from pathlib import Path
import torch
from terminal_probability_metric_v1 import terminal_scores_and_tangent,kl_quadratic
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(91162138)
    h=torch.randn(8,7,dtype=torch.float64);delta=torch.randn_like(h);u=torch.randn(11,7,dtype=torch.float64);eps=1.1920928955078125e-7
    scores,tangent=terminal_scores_and_tangent(h,delta,u,eps)
    def f(x):return 30*torch.tanh(((x@u.T)/(x.square().mean(-1,keepdim=True)+eps).sqrt())/30)
    _,jvp=torch.autograd.functional.jvp(f,h,delta)
    error=float((jvp-tangent).abs().max());assert error<=1e-10
    logp=scores.log_softmax(-1);p=logp.exp();proxy=kl_quadratic(p,tangent);doses={}
    for dose in [.1,.01,.001]:
        actual=(p*(logp-f(h+dose*delta).log_softmax(-1))).sum(-1)
        rel=float((actual-dose*dose*proxy).norm()/actual.norm());doses[str(dose)]=rel
    assert doses['0.001']<=.01
    shift=float(kl_quadratic(p,torch.ones_like(scores)*3).abs().max());assert shift<=1e-12
    result=dict(schema='terminal.probability.metric.control.v1',passed=True,jvp_max_absolute_error=error,finite_dose_kl_relative_errors=doses,common_score_shift_quadratic_max_absolute_error=shift,scope='Exact RMS/tanh score differential, softmax probability Hessian and small-dose KL only. Finite native replacement errors still require measurement.')
    with (P/'TERMINAL_PROBABILITY_METRIC_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
