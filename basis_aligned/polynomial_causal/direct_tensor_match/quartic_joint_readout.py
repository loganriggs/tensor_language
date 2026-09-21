"""Exact fixed-feature value + local derivative readout with matched ridge.
Derivative sufficient statistics use physical (unscaled) feature coordinates.
Both losses divide by their own teacher energy; the value RMS scales determine
one fixed ridge geometry, independent of the derivative loss weight.
"""
import torch


def solve(phi,y,derivative_gram,derivative_cross,derivative_energy,weight,ridge_fraction=1e-6):
    if weight<0:raise ValueError('negative derivative weight')
    s=phi.square().mean(0).sqrt().clamp_min(1e-12);z=phi/s;value_energy=y.square().sum()
    G=z.T@z/value_energy+weight*derivative_gram/s[:,None]/s[None,:]/derivative_energy
    X=y.T@z/value_energy+weight*derivative_cross/s/derivative_energy
    ridge=len(phi)*ridge_fraction/value_energy;system=G+ridge*torch.eye(len(s),device=phi.device,dtype=phi.dtype)
    c=torch.linalg.solve(system,X.T).T
    return c/s,dict(normal_residual=float((c@system-X).norm()/X.norm()),scales=s,ridge=ridge)


def controls():
    from empirical_quartic_dictionary import features,readout
    from quartic_bank_derivative import feature_jacobian
    torch.set_default_dtype(torch.float64);torch.manual_seed(5300);rows=[]
    for family in ['independent','shared_input','shared_output','squares','cancellation']:
        u,v=torch.randn(3,2,4),torch.randn(3,2,4);c=torch.randn(5,6)
        if family=='shared_input':u[1]=u[0]
        if family=='shared_output':c[1]=c[0]
        if family=='squares':v=u.clone()
        if family=='cancellation':u[2]=u[0];v[2]=-v[0]
        x=torch.randn(43,4);phi=features(x,u,v);y=phi@c.T+.01*torch.randn(43,5);jac=[feature_jacobian(u,v,z) for z in x[:5]];target=[c@j+.01*torch.randn(5,4) for j in jac];dg=sum(j@j.T for j in jac);dc=sum(t@j.T for t,j in zip(target,jac));energy=sum(t.square().sum() for t in target);s=phi.square().mean(0).sqrt();valenergy=y.square().sum();errors=[]
        for weight in [0.,.1,1.,10.]:
            writer,info=solve(phi,y,dg,dc,energy,weight)
            design=torch.cat([phi/s/valenergy.sqrt(),*(weight**.5*j.T/s/energy.sqrt() for j in jac),(len(phi)*1e-6/valenergy).sqrt()*torch.eye(6)])
            outputs=torch.cat([y/valenergy.sqrt(),*(weight**.5*t.T/energy.sqrt() for t in target),torch.zeros(6,5)])
            reference=torch.linalg.lstsq(design,outputs,driver='gelsd').solution.T/s
            discrepancy=float((phi@(writer-reference).T).norm()/y.norm());assert discrepancy<1e-8 and info['normal_residual']<1e-10;errors.append(discrepancy)
            if weight==0:
                old,sc,_=readout(phi,y);assert float((phi@writer.T-phi/sc@old).norm()/y.norm())<1e-8
        rows.append(dict(family=family,maximum_augmented_solve_error=max(errors)))
    return rows
if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);rows=controls();Path(__file__).with_name('QUARTIC_JOINT_READOUT_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))


def solve_rank(phi,y,derivative_gram,derivative_cross,derivative_energy,weight,rank,ridge_fraction=1e-6):
    """Globally optimal output-rank constrained writer for the fixed joint metric.

    This optimizes a fixed feature dictionary, not arbitrary arithmetic circuits.
    The ridge is included in the Cholesky whitening, so the rank solution and
    unconstrained baseline minimize exactly the same quadratic objective.
    """
    s=phi.square().mean(0).sqrt().clamp_min(1e-12);z=phi/s;ve=y.square().sum()
    G=z.T@z/ve+weight*derivative_gram/s[:,None]/s[None,:]/derivative_energy
    X=y.T@z/ve+weight*derivative_cross/s/derivative_energy
    G=G+len(phi)*ridge_fraction/ve*torch.eye(len(s),device=phi.device,dtype=phi.dtype)
    L=torch.linalg.cholesky((G+G.T)/2)
    Z=torch.linalg.solve_triangular(L,X.T,upper=False).T
    left,singular,_=torch.linalg.svd(Z,full_matrices=False)
    if not 1<=rank<=len(singular):raise ValueError('invalid output rank')
    basis=left[:,:rank];full=torch.cholesky_solve(X.T,L).T
    c=basis@(basis.T@full)
    objective=((c@G)*c).sum()-2*(c*X).sum()
    optimum=-singular[:rank].square().sum()
    certificate=float(abs(objective-optimum)/optimum.abs().clamp_min(1e-30))
    return c/s,basis,dict(spectral_objective_relative_error=certificate)
