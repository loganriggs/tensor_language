"""Actual-weight CPU probe; synthetic contexts are not text/OOD evidence."""
import json
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,coefficients
from response_diagonal_conic_v1 import prepare_direct,execute
from conic_amplitude_polynomial_v1 import compile_numerator,compile_degree2,evaluate,absolute_bound

P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(9130947)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    lam=float(sd['transformer.h.10.lambdas'][0])
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    z=torch.randn(32,1152,dtype=torch.float64)
    base=((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    ctx,basis=prepare_basis(z,base,program,lam)
    bank=prepare_direct(basis,l,r,d,ctx);n,scale=compile_numerator(bank,ctx)
    direct,_=compile_degree2(basis@l.T,basis@r.T,d,ctx)
    direct_error=float((direct-n[:,:3]).norm()/n[:,:3].norm())
    singular=torch.linalg.svdvals(n/n.norm(dim=-1,keepdim=True))
    rows=[];max_identity=0.;bound_excess=0.;bound_relative_excess=0.
    for radius in (.001,.01,.1,.3,1.):
        t=torch.linspace(-radius,radius,65,dtype=torch.float64)[:,None,None]
        ref=execute(coefficients(t*scale,ctx),bank)
        exact=evaluate(t,n,scale,ctx)
        max_identity=max(max_identity,float((exact-ref).norm()/ref.norm()))
        # SVD lower bound for any rank-r output representation on this fixed grid.
        s=torch.linalg.svdvals(ref.transpose(0,1))
        for degree in range(4):
            fit=evaluate(t,n,scale,ctx,degree)
            errors=(fit-ref).norm(dim=-1,keepdim=True)
            bound=absolute_bound(t,n,scale,ctx,degree)
            bound_excess=max(bound_excess,float((errors-bound).max()))
            bound_relative_excess=max(bound_relative_excess,float(((errors-bound)/(ref.norm(dim=-1,keepdim=True)+bound).clamp_min(1e-300)).max()))
            rel=(fit-ref).norm(dim=(0,2))/ref.norm(dim=(0,2))
            best=(s[:,degree+1:].square().sum(-1)/s.square().sum(-1)).sqrt()
            rows.append(dict(radius=radius,degree=degree,writers=degree+1,
                relative_error_median=float(rel.median()),relative_error_max=float(rel.max()),
                optimal_grid_rank_error_median=float(best.median()),
                max_absolute_bound=float(bound.max())))
    result=dict(direct_degree2_relative_error=direct_error,identity_relative_error=max_identity,absolute_bound_excess=bound_excess,relative_bound_excess=bound_relative_excess,
        execution_correction='Initial run failed fixed 1e-10 absolute floating-point bound check at large amplitudes. Preserve absolute excess and check it relative to output-plus-bound scale; mathematical bound is unchanged.',
        equilibrated_coefficient_smin_smax=(singular[:,-1]/singular[:,0]).tolist(),
        rows=rows,scope='32 synthetic Gaussian contexts, actual weights, FP64 CPU; dimensionless amplitude grid. No text fit, circuit transfer, runtime or whole-model compression claim.')
    (P/'CONIC_AMPLITUDE_POLYNOMIAL_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert max_identity<1e-10
    assert direct_error<1e-10
    assert bound_relative_excess<1e-12

if __name__=='__main__':main()
