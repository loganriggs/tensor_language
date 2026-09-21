"""Independent exact spherical quadrature for five structures in dimension three."""
import json
from pathlib import Path
import numpy as np
import torch
from spherical_source_moments import moments

def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    # Gauss-Legendre polar integral and uniform azimuth integrate degree-four
    # Cartesian polynomials exactly; no stochastic tolerances are needed.
    u,w=np.polynomial.legendre.leggauss(6);theta=np.arange(16)*2*np.pi/16
    points=[];weights=[]
    for a,b in zip(u,w):
        for t in theta:
            points.append(np.sqrt(3)*np.array([np.sqrt(1-a*a)*np.cos(t),np.sqrt(1-a*a)*np.sin(t),a]));weights.append(b/32)
    x=torch.tensor(np.array(points));w=torch.tensor(weights);rows=[]
    for seed,name in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
        torch.manual_seed(seed);l=torch.randn(5,3);r=torch.randn(5,3);d=torch.randn(4,5)
        if name=='shared_input':l[1:]=l[0]
        if name=='shared_output':d[:,1:]=d[:,0,None]
        if name=='squares':r=l.clone()
        if name=='cancellation':l[1]=l[0];r[1]=r[0];d[:,1]=-d[:,0]
        mean,cov=moments(l,r,d);y=((x@l.T)*(x@r.T))@d.T;reference=(w[:,None]*y).sum(0);dy=y-reference;reference_cov=dy.T@(w[:,None]*dy)
        rows.append(dict(structure=name,mean_error=float((mean-reference).norm()/(1+reference.norm())),covariance_error=float((cov-reference_cov).norm()/(1+reference_cov.norm()))))
    # Isotropic quadratic is constant on the sphere, but Gaussian variance is 2d.
    eye=torch.eye(3);mean,cov=moments(eye,eye,torch.ones(1,3))
    result=dict(rows=rows,radial_mean=float(mean[0]),radial_variance=float(cov[0,0]),gaussian_radial_variance=6.,passed=all(max(r['mean_error'],r['covariance_error'])<1e-12 for r in rows) and abs(float(cov[0,0]))<1e-12)
    assert result['passed'];Path(__file__).with_name('SPHERICAL_SOURCE_MOMENT_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
