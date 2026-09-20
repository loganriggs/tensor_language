"""Gaussian linear projection directions via derivative of exact teacher mean.
For fixed M, d_mu E F(mu+delta) = E grad F(mu+delta).
JVPs give B v without a full native Jacobian. No empirical output fitting.
"""
import torch
from gaussian_quartic_mean import native_mean

def directions(teacher,mu,M,vectors):
    return torch.stack([torch.func.jvp(lambda center:native_mean(teacher,center,M),(mu,),(v,))[1] for v in vectors],1)

def check():
    import json
    from pathlib import Path
    from gaussian_low_degree_projection import project
    from implicit_quartic import entries
    torch.set_num_threads(1);torch.manual_seed(2026);torch.set_default_dtype(torch.float64)
    teacher=[torch.randn(*s) for s in [(2,3),(3,5),(3,5),(5,6),(6,4),(6,4)]];mu=torch.randn(4);raw=torch.randn(4,4);M=raw@raw.T+.1*torch.eye(4);v=torch.randn(3,4)
    idx=torch.cartesian_prod(*[torch.arange(4)]*4);H=entries(*teacher,idx).T.reshape(2,4,4,4,4);_,B,_,_=project(H,mu,M);actual=directions(teacher,mu,M,v);reference=B@v.T;error=float((actual-reference).norm()/reference.norm());assert error<1e-12
    out=dict(relative_error=error,scope='Exact Gaussian linear projection directional derivatives, independent dense quartic oracle; native runtime and fitting pending.');(Path(__file__).resolve().parent/'GAUSSIAN_LINEAR_PROJECTION_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':check()
