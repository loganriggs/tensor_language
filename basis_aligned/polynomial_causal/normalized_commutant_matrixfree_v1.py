"""PSD shifted normalized commutant operator, excluding the trivial identity."""
import json
from pathlib import Path
import torch
from fullu_input_blocks_v1 import sandwich,partition_metrics
from quartic_matrixfree_eigen_v2 import eigenmatrices


class NormalizedCommutant:
    def __init__(self,l,r,g):
        self.g=g
        d=l.shape[1]
        k=sandwich(l,r,g,torch.eye(d,device=l.device,dtype=l.dtype))
        values,self.basis=torch.linalg.eigh(k)
        if float(values.min())<=0:raise ValueError('K must be positive definite; do not silently ridge null directions')
        self.k_values=values
        self.l,self.r=l@self.basis,r@self.basis
        self.mass_root=(values[:,None]+values[None,:]).sqrt()
        self.trivial=torch.diag((2*values).sqrt());self.trivial/=self.trivial.norm()

    def project(self,y):
        return y-(self.trivial*y).sum()*self.trivial

    def action(self,y):
        y=self.project(y)
        out=.5*y+sandwich(self.l,self.r,self.g,y/self.mass_root)/self.mass_root
        return self.project((out+out.T)/2)

    def original_witness(self,y):
        x=self.project(y)/self.mass_root
        return self.basis@x@self.basis.T


def control():
    from normalized_commutant_relaxation_v1 import dense_relaxation,symmetric_basis
    from toy_consumer_commutant_blocks import planted_family,dense_null_family
    torch.set_default_dtype(torch.float64);d=8
    l=torch.eye(d).repeat_interleave(d,0);r=torch.eye(d).repeat(d,1)
    results=[]
    for kind in ('planted','dense'):
        forms=planted_family(seed=120432,block_sizes=(4,4),consumers=10)[0] if kind=='planted' else dense_null_family(seed=120432,dimension=d,consumers=10)
        w=forms.reshape(10,-1);g=w.T@w
        operator=NormalizedCommutant(l,r,g)
        dense,_=dense_relaxation(l,r,g)
        values,matrices,report=eigenmatrices(operator.action,d,k=2,seed=120438,tol=1e-11,ncv=12,max_actions=300)
        order=values.argsort()[::-1].copy();values=values[order];matrices=[matrices[i] for i in order]
        estimates=2*(1-torch.tensor(values))
        error=float((estimates-torch.tensor(dense['eigenvalues'][:2])).abs().max())
        basis=symmetric_basis(d)
        actions=torch.stack([operator.action(b) for b in basis])
        full=torch.einsum('aij,bij->ab',basis,actions)
        symmetry=float((full-full.T).norm()/full.norm())
        spectrum=torch.linalg.eigvalsh((full+full.T)/2)
        _,q=torch.linalg.eigh(operator.original_witness(matrices[0]));p=q[:,:4]@q[:,:4].T
        rounded=partition_metrics(l,r,g,p)
        results.append(dict(kind=kind,relaxed_eigenvalues=estimates.tolist(),dense_eigenvalue_error=error,
                            symmetry_error=symmetry,shifted_spectrum_range=[float(spectrum[0]),float(spectrum[-1])],
                            identity_action_norm=float(operator.action(operator.trivial).norm()),rounded=rounded,solver=report))
    passed=all(x['dense_eigenvalue_error']<1e-9 and x['symmetry_error']<1e-12 and x['identity_action_norm']<1e-12 and x['solver']['status']=='converged' for x in results) and abs(results[0]['rounded']['normalized_cut'])<1e-10
    return dict(results=results,passed=passed)


if __name__=='__main__':
    result=control();Path(__file__).with_name('NORMALIZED_COMMUTANT_MATRIXFREE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert result['passed']
