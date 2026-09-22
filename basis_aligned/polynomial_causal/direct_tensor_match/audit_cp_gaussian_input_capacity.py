"""Degree-four Gaussian conditional-variance bounds from exact derivative moments."""
import itertools,json,time
from pathlib import Path
import numpy as np
import torch
from gaussian_cp_derivative_gram import gram
from mixed_gaussian_cp import gram_dynamic
P=Path(__file__).resolve().parent

def controls():
    dtype=torch.float64;nodes,weights=np.polynomial.hermite.hermgauss(5);ids=torch.tensor(list(itertools.product(range(5),repeat=3)));z=torch.tensor(nodes*2**.5,dtype=dtype)[ids];w1=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype);w=w1[ids].prod(1);rows=[]
    for seed in range(5):
        torch.manual_seed(13200+seed);f=[torch.randn(5,3,dtype=dtype) for _ in range(4)];b=[torch.randn(5,dtype=dtype) for _ in range(4)];C=torch.randn(2,5,dtype=dtype)
        if seed==1:f[1]=f[0].clone();b[1]=b[0].clone()
        if seed==2:C[1]=C[0]
        if seed==3:f[2:]=[a.clone() for a in f[:2]];b[2:]=[a.clone() for a in b[:2]]
        if seed==4:
            for a,c in zip(f,b):a[1]=a[0];c[1]=c[0]
            C[:,1]=-C[:,0]
        lin=[z@a.T+c for a,c in zip(f,b)];phi=torch.ones_like(lin[0]);jac=0.
        for a in lin:phi*=a
        for i in range(4):
            prod=torch.ones_like(phi)
            for j in range(4):
                if i!=j:prod*=lin[j]
            jac=jac+prod[:,:,None]*f[i][None,:,:]
        J=torch.einsum('nad,va->nvd',jac,C);reference=torch.einsum('nvi,nvj,n->ij',J,J,w);K=gram(f,b,input_coefficients=C);error=float((K-reference).norm()/reference.norm());assert error<1e-10
        values=phi@C.T;tests=[]
        for rank in [0,1,2,3]:
            grid=values.reshape(5**rank,5**(3-rank),2);subids=torch.tensor(list(itertools.product(range(5),repeat=3-rank)),dtype=torch.long);wv=w1[subids].prod(1) if rank<3 else torch.ones(1,dtype=dtype)
            mean=(grid*wv[None,:,None]).sum(1);res=(grid-mean[:,None]).reshape_as(values);variance=float((res.square()*w[:,None]).sum());derivative=float(K[rank:,rank:].trace());tol=1e-10*(1+float(K.trace()));assert derivative/4-tol<=variance<=derivative+tol
            tests.append(dict(rank=rank,conditional_error_squared=variance,discarded_gradient_energy=derivative))
        rows.append(dict(seed=seed,gradient_matrix_quadrature_error=error,conditional_bounds=tests))
    return rows

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checked=controls();cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'];mu=cache['mean'];rows=[]
    for seed in [1001,1002]:
        s=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);f=[a.double() for a in s['factors']];C=s['coefficients'].double()/19054614563.464127;fw=[a@S for a in f];bias=[a@mu for a in f];K=gram(fw,bias,input_coefficients=C);K=(K+K.T)/2;e=torch.linalg.eigvalsh(K);assert e[0]>=-1e-10*e[-1];g=gram_dynamic(fw,bias,fw,bias);energy=((C.T@C)*g).sum();dg=gram(fw,bias);trace_error=float(abs(K.trace()-((C.T@C)*dg).sum())/K.trace());assert trace_error<1e-10
        spectrum=e.clamp_min(0).flip(0);tail=torch.cat([spectrum.flip(0).cumsum(0).flip(0),spectrum.new_zeros(1)]);bounds=[]
        for count in [4,8,16,32,64,128,192,256]:
            rank=min(4*count,len(e));bounds.append(dict(cp_terms=count,input_rank=rank,relative_gaussian_error_lower_bound=float((tail[rank]/(4*energy)).sqrt()),active_subspace_conditional_mean_error_upper_bound=float((tail[rank]/energy).sqrt())))
        necessary={}
        for tol in [.1,.05,.01]:
            rank=int(torch.where(tail<=4*tol*tol*energy)[0][0]);necessary[str(tol)]=dict(necessary_input_rank=rank,necessary_cp_terms=(rank+3)//4)
        rows.append(dict(seed=seed,trace_replay=trace_error,bounds=bounds,necessary=necessary));print(json.dumps(rows[-1]),flush=True)
    result=dict(rows=rows,controls=checked,seconds=time.monotonic()-start,scope='FittedCP512parent underfixedcalibrationmean/covarianceGaussian; polynomialdegree<=4. Exactimplicit E[J_z^T J_z], spectraltaillowerbound/4 forANYrank-rinput-subspace approximation; CPtermsk implyr<=4k. Upperbound appliesconditionalmean onactiveeigensubspace, NOT a k-termCP representation/algorithm. No nativeeffect/OODbound ororiginalnative-tensorcertificate. Floating-point numericalaudit.')
    (P/'CP_GAUSSIAN_INPUT_CAPACITY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
