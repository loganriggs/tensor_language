"""Planted common optimum with a non-isotropic producer metric, condition100."""
import json
from pathlib import Path
import torch
from coupled_producer_pymanopt_v1 import fit
from coupled_producer_routing_objective_v1 import producer_numerator


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1433)
    d,r=13,3;q=torch.linalg.qr(torch.randn(d,d)).Q
    g=q@torch.diag(torch.logspace(0,2,d))@q.T;a=torch.linalg.qr(torch.randn(d,r)).Q
    m,_=producer_numerator(a.T,g);s=a@a.T+.01*torch.eye(d);s/=s.trace()
    mu=float(torch.linalg.eigvalsh(s)[-r:].sum());rows=[]
    for seed in [1439,1447]:
        torch.manual_seed(seed);e,report=fit(torch.linalg.qr(torch.randn(d,r)).Q,g,m,s,1.,mu,seconds=10)
        rows.append(dict(seed=seed,**report,subspace_overlap=float(torch.linalg.svdvals(a.T@e).square().mean())))
    out=dict(instrument_passed=all(x['converged'] and abs(x['score']-1)<1e-10 and x['subspace_overlap']>1-1e-10 for x in rows),
             condition=float(torch.linalg.cond(g)),fits=rows,
             scope='Non-isotropic metric, condition100, planted same optimal sharing/routing subspace. No native evidence.')
    with Path(__file__).with_name('COUPLED_PRODUCER_PYMANOPT_NONISOTROPIC_V1_CONTROL.json').open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':main()
