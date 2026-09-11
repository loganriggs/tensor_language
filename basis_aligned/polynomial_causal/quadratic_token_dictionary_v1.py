"""Sparse nonorthogonal token-function dictionary in exact weight Hilbert coordinates.

Minimize .5||X-A B||²+lambda||A||1 with row norms of B<=1.
Conditional accelerated proximal solves; joint convergence is checked separately.
No token text or activation data. B's rows may be dense quadratic functions.
"""
import json,math
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross


def embed(u,l,r,down):
    gram=product_cross(l,r,l,r)
    metric=down@gram@down.T;metric=(metric+metric.T)/2
    root=torch.linalg.cholesky(metric)
    mean=u.mean(0);x=(u-mean)@root
    scale=(x.square().sum()/len(x)).sqrt()
    native_basis=torch.linalg.solve_triangular(root,down,upper=False)
    return x/scale,root,native_basis,scale,mean


def soft(x,amount):
    return x.sign()*(x.abs()-amount).clamp_min(0)


def ball(x):
    """Column unit balls; dictionary variables are stored transposed here."""
    return x/x.norm(dim=0,keepdim=True).clamp_min(1)


def objective_part(x,gram,cross,penalty):
    return float((.5*x*(x@gram)-x*cross).sum(dtype=torch.float64)
                 +penalty*x.abs().sum(dtype=torch.float64))


def stationarity(x,gram,cross,penalty,kind,lipschitz=None):
    if lipschitz is None:
        lipschitz=max(float(torch.linalg.eigvalsh((gram+gram.T)/2)[-1]),1e-15)
    grad=x@gram-cross
    prox=soft(x-grad/lipschitz,penalty/lipschitz) if kind=='codes' else ball(x-grad/lipschitz)
    return float((lipschitz*(x-prox)).norm()/cross.norm().clamp_min(1e-30))


def conditional(initial,gram,cross,penalty,kind,max_steps=200,tolerance=1e-5):
    gram=(gram+gram.T)/2
    lipschitz=max(float(torch.linalg.eigvalsh(gram)[-1])*1.01,1e-15)
    x=initial.clone();y=x.clone();momentum=1.;value=objective_part(x,gram,cross,penalty)
    resets=0;largest_increase=0.;residual=math.inf
    def step(point):
        proposal=point-(point@gram-cross)/lipschitz
        return soft(proposal,penalty/lipschitz) if kind=='codes' else ball(proposal)
    for iteration in range(max_steps):
        candidate=step(y);candidate_value=objective_part(candidate,gram,cross,penalty)
        if candidate_value>value:
            resets+=1;momentum=1.;candidate=step(x)
            candidate_value=objective_part(candidate,gram,cross,penalty)
            # A roundoff-level objective reversal is not a convergence certificate.
            if candidate_value>value+1e-10*max(1.,abs(value)):
                residual=stationarity(x,gram,cross,penalty,kind,lipschitz)
                return x,dict(steps=iteration+1,relative_stationarity=residual,
                              converged=residual<=tolerance,resets=resets,terminal='nondecreasing_step',
                              maximum_objective_increase=largest_increase)
        largest_increase=max(largest_increase,candidate_value-value)
        new_momentum=(1+math.sqrt(1+4*momentum*momentum))/2
        y=candidate+(momentum-1)/new_momentum*(candidate-x)
        x=candidate;value=candidate_value;momentum=new_momentum
        if (iteration+1)%10==0 or iteration==max_steps-1:
            residual=stationarity(x,gram,cross,penalty,kind,lipschitz)
            if residual<=tolerance:break
    return x,dict(steps=iteration+1,relative_stationarity=residual,converged=residual<=tolerance,
                  resets=resets,terminal='stationarity' if residual<=tolerance else 'step_limit',
                  maximum_objective_increase=largest_increase)


def cycle(x,codes,dictionary,penalty,max_steps=200,tolerance=1e-5):
    gram=dictionary@dictionary.T;cross=x@dictionary.T
    codes,code_report=conditional(codes,gram,cross,penalty,'codes',max_steps,tolerance)
    gram=codes.T@codes;cross=x.T@codes
    transposed,dictionary_report=conditional(dictionary.T,gram,cross,0.,'dictionary',max_steps,tolerance)
    return codes,transposed.T,dict(codes=code_report,dictionary=dictionary_report)


def diagnostics(x,codes,dictionary,penalty):
    residual=x-codes@dictionary
    error=float(residual.square().sum(dtype=torch.float64)/x.square().sum(dtype=torch.float64))
    penalty_value=float(penalty*codes.abs().sum(dtype=torch.float64)/len(x))
    code_station=stationarity(codes,dictionary@dictionary.T,x@dictionary.T,penalty,'codes')
    dictionary_station=stationarity(dictionary.T,codes.T@codes,x.T@codes,0.,'dictionary')
    active=(codes!=0).sum(1).double()
    return dict(objective=.5*float(residual.square().sum(dtype=torch.float64)/len(x))+penalty_value,
                squared_relative_error=error,captured_energy=1-error,penalty_value=penalty_value,
                code_stationarity=code_station,dictionary_stationarity=dictionary_station,
                relative_stationarity=max(code_station,dictionary_station),
                median_active=float(active.median()),mean_active=float(active.mean()),
                p90_active=float(torch.quantile(active,.9)),
                atom_norm_max=float(dictionary.norm(dim=1).max()),nonzero_codes=int((codes!=0).sum()))


def control():
    torch.manual_seed(1021);torch.set_default_dtype(torch.float64)
    u,l,r,down=torch.randn(13,4),torch.randn(9,6),torch.randn(9,6),torch.randn(4,9)
    x,root,basis,scale,mean=embed(u,l,r,down)
    h=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    dense=torch.einsum('vk,kij->vij',(u-mean)@down,h).flatten(1)/scale
    gram_error=float((x@x.T-dense@dense.T).norm()/(dense@dense.T).norm())
    atoms=torch.randn(3,4)
    direct=torch.einsum('ak,kij->aij',atoms@basis,h).flatten(1)
    atom_error=float((direct@direct.T-atoms@atoms.T).norm()/(atoms@atoms.T).norm())
    dictionary=torch.eye(7)[:3];target=torch.zeros(30,7)
    target[torch.arange(30),torch.arange(30)%3]=torch.linspace(.5,1.5,30)*torch.where(torch.arange(30)%2==0,1.,-1.)
    expected=soft(target[:,:3],.1)
    codes,reported=conditional(torch.zeros(30,3),dictionary@dictionary.T,target@dictionary.T,.1,'codes',300,1e-9)
    code_error=float((codes-expected).norm()/expected.norm())
    codes,updated,cycle_report=cycle(target,codes,dictionary,.1,300,1e-9)
    final=diagnostics(target,codes,updated,.1)
    # Independent constrained smooth solve of a nonorthogonal Lasso fixture.
    import numpy as np
    from scipy.optimize import minimize
    nonorth=torch.randn(4,7);nonorth/=nonorth.norm(dim=1,keepdim=True)
    dense_target=torch.randn(5,7);penalty=.13
    fitted,fitted_report=conditional(torch.zeros(5,4),nonorth@nonorth.T,dense_target@nonorth.T,penalty,'codes',2000,1e-9)
    def reference_objective(flat):
        parts=torch.from_numpy(flat).reshape(2,5,4);a=parts[0]-parts[1]
        residual=a@nonorth-dense_target;gradient=residual@nonorth.T
        value=.5*residual.square().sum()+penalty*parts.sum()
        jac=torch.stack([gradient+penalty,-gradient+penalty]).flatten().numpy()
        return float(value),jac
    independent=minimize(reference_objective,np.zeros(40),jac=True,method='L-BFGS-B',bounds=[(0,None)]*40,
                         options={'ftol':1e-14,'gtol':1e-10,'maxiter':2000})
    ours=float(.5*(fitted@nonorth-dense_target).square().sum()+penalty*fitted.abs().sum())
    nonorth_error=abs(ours-independent.fun)/max(1.,abs(independent.fun))
    passed=max(gram_error,atom_error,code_error)<1e-9 and final['relative_stationarity']<1e-9 and final['atom_norm_max']<=1+1e-12 and fitted_report['converged'] and nonorth_error<1e-9
    result=dict(instrument_passed=passed,exact_embedding_gram_error=gram_error,
                atom_function_isometry_error=atom_error,orthogonal_soft_threshold_error=code_error,
                planted_final=final,conditional_report=reported,cycle_report=cycle_report,
                nonorthogonal_reference_objective_error=nonorth_error,nonorthogonal_code_stationarity=fitted_report['relative_stationarity'],
                scope='Exact quadratic embedding and known-optimum sparse-code/dictionary controls. No native fit or global nonconvex recovery claim.')
    Path(__file__).with_name('QUADRATIC_TOKEN_DICTIONARY_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    assert passed,result
    return result


if __name__=='__main__':print(json.dumps(control(),indent=2))
