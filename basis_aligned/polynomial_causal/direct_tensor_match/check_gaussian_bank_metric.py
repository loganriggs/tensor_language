import json
from pathlib import Path
import torch
from gaussian_bank_metric import gram_by_degree
from empirical_quartic_dictionary import features
from dag_square_optimizer import rule


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    x,w=rule(5);x=x[:,:4];rows=[]
    for seed,kind in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
        torch.manual_seed(4200+seed);u=torch.randn(3,2,4);v=torch.randn_like(u)
        if kind=='shared_input':u[1]=u[0]
        if kind=='shared_output':u[2]=u[0];v[2]=v[0]+v[1]
        if kind=='squares':v=u.clone()
        if kind=='cancellation':u[2]=u[0];v[2]=-v[0]
        # Nonidentity law checks the covariance-transform convention too.
        t=torch.randn(4,4);t=t/2+torch.eye(4)
        a,b=u@t,v@t
        parts,metadata=gram_by_degree(a,b)
        phi=features(x@t.T,u,v);reference=phi.T@(w[:,None]*phi)
        got=sum(parts.values());error=float((got-reference).norm()/reference.norm())
        mean_error=float((metadata['mean']-(w[:,None]*phi).sum(0)).norm()/metadata['mean'].norm())
        # PSD of each Hermite block, including rank-deficient/cancelling banks.
        minimum=min(float(torch.linalg.eigvalsh(g).min()/g.norm()) for g in parts.values())
        assert error<1e-11 and mean_error<1e-11 and minimum>-1e-12
        rows.append(dict(family=kind,gram_error=error,mean_error=mean_error,minimum_relative_eigenvalue=minimum))
    result=dict(records=rows,scope='Five explicit bank structures; exact degree8 Gaussian quadrature under nonidentity covariance. Gram correctness, not optimizer or semantic recovery.')
    Path(__file__).with_name('GAUSSIAN_BANK_METRIC_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
