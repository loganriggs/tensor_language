"""Exact scalar calibration of structured branches under the product penalty."""
import json
from pathlib import Path
import torch


@torch.no_grad()
def inner(first,second,chunk=256):
    a,b,w=first;c,d,v=second;answer=a.new_zeros(())
    for start in range(0,len(a),chunk):
        sl=slice(start,start+chunk)
        gram=((a[sl]@c.T)*(b[sl]@d.T)+(a[sl]@d.T)*(b[sl]@c.T))/2
        answer+=((w[:,sl].T@v)*gram).sum()
    return answer


@torch.no_grad()
def calibrate(model,native,whitener,penalty=.01,chunk=256):
    matrices=model.matrices()
    branches=[(a,c,whitener@b) for a,c,b in matrices]
    gram=torch.stack([torch.stack([inner(a,b,chunk) for b in branches]) for a in branches])
    cross=torch.stack([inner(native,b,chunk) for b in branches])
    energy=torch.stack([(w.square().sum(0)*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())/2).sum() for a,b,w in branches])
    reg=gram+penalty*torch.diag(energy)
    alpha=torch.linalg.solve(reg,cross)
    error=float((reg@alpha-cross).norm()/cross.norm().clamp_min(1e-30))
    for branch,scale in zip(model.maps,alpha):branch[2].stages[-1].mul_(scale)
    return dict(scales=alpha.tolist(),relative_solve_error=error,condition=float(torch.linalg.cond(reg)))


def control():
    from structured_bilinear_bank_v1 import StructuredBank
    from chunked_bilinear_coefficient_v1 import dense
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(623)
    m=StructuredBank([2,2],branches=2);u=torch.randn(7,4)
    a,b,w=m.factors();targetw=w.detach().clone()
    targetw[:,:4]*=2.;targetw[:,4:]*=-.7
    native=(a.detach(),b.detach(),u@targetw)
    stats=calibrate(m,native,u,penalty=0.,chunk=3)
    aa,bb,ww=m.factors()
    error=float((dense(aa,bb,u@ww)-dense(*native)).norm()/dense(*native).norm())
    result=dict(**stats,relative_function_error=error)
    with Path(__file__).with_name('STRUCTURED_BRANCH_AMPLITUDES_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));assert max(error,stats['relative_solve_error'])<1e-10


if __name__=='__main__':control()
