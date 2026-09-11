"""L1 token codes and unit-Frobenius real-product atoms, exact conditional updates.

Native coefficient tensors stay implicit. Products may leave the native output span.
The spectral atom update is global for fixed codes/other atoms, not jointly global.
"""
import json
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross
from quadratic_token_dictionary_v1 import conditional,stationarity


def form(l,r,coeff):
    q=(l.T*coeff)@r
    return (q+q.T)/2


def unit_product(q):
    """Maximize <q,H> over unit-Frobenius H=sym(l r^T)."""
    ev,vec=torch.linalg.eigh((q+q.T)/2)
    pos=ev[-1].clamp_min(0);neg=(-ev[0]).clamp_min(0)
    magnitude=(pos.square()+neg.square()).sqrt()
    if float(magnitude)==0:
        l=torch.zeros_like(q[0]);l[0]=1
        return l,l.clone(),magnitude
    p=(pos/magnitude).sqrt()*vec[:,-1]
    n=(neg/magnitude).sqrt()*vec[:,0]
    return p+n,p-n,magnitude


def grams(u,l,r,down,pl,pr):
    cross=u@(down@product_cross(l,r,pl,pr))
    return product_cross(pl,pr,pl,pr),cross


def report(u,l,r,down,pl,pr,codes,penalty,target_energy):
    gram,cross=grams(u,l,r,down,pl,pr)
    approx=((codes.T@codes)*gram).sum()
    overlap=(codes*cross).sum()
    error=(target_energy+approx-2*overlap)/target_energy
    active=(codes!=0).sum(1).double()
    return dict(objective=float(.5*error*target_energy/len(u)+penalty*codes.abs().sum()/len(u)),
                captured_energy=float(1-error),median_active=float(active.median()),
                mean_active=float(active.mean()),nonzero_codes=int((codes!=0).sum()),
                code_stationarity=stationarity(codes,gram,cross,penalty,'codes'),
                maximum_atom_norm_error=float((gram.diag()-1).abs().max()))


def atom_sweep(u,l,r,down,pl,pr,codes,update=True):
    """Sequential exact atom minimizations, or simultaneous frozen-point gap audit."""
    weights=codes.T@codes
    coefficients=(codes.T@u)@down
    gains=[];maximum_norm_error=0.
    for j in range(len(pl)):
        if float(weights[j,j])==0:
            gains.append(0.);continue
        other=weights[j].clone();other[j]=0
        residual=form(l,r,coefficients[j])-form(pl,pr,other)
        left,right,optimum=unit_product(residual)
        old_inner=(pl[j]@residual@pr[j])
        gain=float(optimum-old_inner)
        gains.append(gain/len(u))
        if update:
            # Reject a numerical increase; stopping alone is never convergence.
            if gain>=0:
                pl[j]=left;pr[j]=right
    return dict(sum_conditional_gain=sum(max(0.,g) for g in gains),
                maximum_conditional_gain=max(gains),minimum_conditional_gain=min(gains),
                frozen_point=not update)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1031)
    # Known indefinite optimum and PSD case with two positive modes.
    checks=[]
    for diagonal in [[-4.,2.,3.,1.],[4.,3.,2.,1.],[-4.,-3.,-2.,-1.]]:
        q=torch.diag(torch.tensor(diagonal));a,b,opt=unit_product(q)
        dense=(a[:,None]*b[None,:]+b[:,None]*a[None,:])/2
        exact=(max(max(diagonal),0)**2+min(min(diagonal),0)**2)**.5
        checks.extend([abs(float(dense.norm())-1),abs(float((dense*q).sum())-exact),abs(float(opt)-exact)])
    v,d,n,k=17,7,10,4
    u=torch.randn(v,5);l=torch.randn(n,d);r=torch.randn(n,d);down=torch.randn(5,n)
    pl=torch.randn(k,d);pr=torch.randn(k,d)
    norms=product_cross(pl,pr,pl,pr).diag().sqrt();pl/=norms.sqrt()[:,None];pr/=norms.sqrt()[:,None]
    codes=torch.randn(v,k);penalty=.03
    target=torch.stack([form(l,r,c) for c in u@down]);energy=target.square().sum()
    def dense_objective():
        atoms=torch.stack([form(pl[j:j+1],pr[j:j+1],torch.ones(1)) for j in range(k)])
        return float(.5*(target-torch.einsum('vk,kij->vij',codes,atoms)).square().sum()/v+penalty*codes.abs().sum()/v)
    before=report(u,l,r,down,pl,pr,codes,penalty,energy)
    dense_error=abs(before['objective']-dense_objective())/max(1.,before['objective'])
    gram,cross=grams(u,l,r,down,pl,pr)
    codes,code_report=conditional(codes,gram,cross,penalty,'codes',2000,1e-9)
    prior=report(u,l,r,down,pl,pr,codes,penalty,energy)['objective']
    sweep=atom_sweep(u,l,r,down,pl,pr,codes)
    after=report(u,l,r,down,pl,pr,codes,penalty,energy)
    gain_error=abs(prior-after['objective']-sweep['sum_conditional_gain'])/max(1.,prior)
    dense_after_error=abs(after['objective']-dense_objective())/max(1.,after['objective'])
    frozen=atom_sweep(u,l,r,down,pl,pr,codes,False)
    # With one atom, its global conditional update leaves no atom improvement.
    atom_sweep(u,l,r,down,pl[:1],pr[:1],codes[:,:1])
    one=atom_sweep(u,l,r,down,pl[:1],pr[:1],codes[:,:1],False)
    result=dict(instrument_passed=max(checks+[dense_error,gain_error,dense_after_error,abs(one['sum_conditional_gain'])])<1e-8
                and after['objective']<=prior and code_report['converged'],
                spectral_optimum_error=max(checks),dense_objective_error=dense_error,
                dense_after_error=dense_after_error,conditional_gain_identity_error=gain_error,
                one_atom_remaining_gain=one['sum_conditional_gain'],code_report=code_report,
                sweep=sweep,frozen_joint_gap=frozen,
                scope='Exact dense objective, signed inertia, conditional optimum and monotonicity. Not native/global recovery.')
    Path(__file__).with_name('SPARSE_PRODUCT_DICTIONARY_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
