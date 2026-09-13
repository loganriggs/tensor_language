"""Actual-weight diagonal identity/FP32 control; no native adoption claim."""
import json,time
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,prepare_products,coefficients,combine
from response_diagonal_conic_v1 import prepare_diagonal,execute
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(61571);start=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    z=torch.randn(4,1152,dtype=torch.float64)
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    base=((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    ctx,basis=prepare_basis(z,base,program,float(sd['transformer.h.10.lambdas'][0]))
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    pairs,products=prepare_products(basis,l,r,d);bank=prepare_diagonal(products,ctx)
    rows=[];zero=0.;conicmax=0.
    rho0=ctx['perpendicular_rms']+ctx['writer_rms']*ctx['parallel'].square()
    for strength in [-1.,-.2,-1e-4,-1e-8,0.,1e-8,1e-4,.2,1.,2.]:
        u=coefficients(torch.full((4,1),strength,dtype=z.dtype),ctx)
        direct=combine(u,u,pairs,products);pred=execute(u,bank);fp32=execute(u.float(),bank.float()).double()
        e=float((pred-direct).norm()/direct.norm().clamp_min(1e-300));e32=float((fp32-direct).norm()/direct.norm().clamp_min(1e-300))
        lhs=u[:,0:1]*u[:,1:2];rhs=rho0*u[:,1:2].square()+4*ctx['cross_rms']*u[:,1:2]*u[:,2:3]+4*ctx['writer_rms']*u[:,2:3].square()
        conicmax=max(conicmax,float((lhs-rhs).abs().max()))
        if strength==0:zero=float(pred.abs().max())
        rows.append(dict(strength=strength,relative_error=e,fp32_relative_error=e32))
    # The conic identity does not polarize to zero for arbitrary two amplitudes.
    u=coefficients(torch.full((4,1),.2,dtype=z.dtype),ctx);v=coefficients(torch.full((4,1),-.7,dtype=z.dtype),ctx)
    cross=u[:,0:1]*v[:,1:2]+u[:,1:2]*v[:,0:1]-2*rho0*u[:,1:2]*v[:,1:2]-4*ctx['cross_rms']*(u[:,1:2]*v[:,2:3]+u[:,2:3]*v[:,1:2])-8*ctx['writer_rms']*u[:,2:3]*v[:,2:3]
    result=dict(pred_a=max(x['relative_error'] for x in rows)<=1e-10 and zero==0,pred_b=max(x['fp32_relative_error'] for x in rows)<=1e-5,pred_c=float(cross.abs().max())>1e-10,
        rows=rows,conic_max_absolute_residual=conicmax,independent_cross_conic_residual=float(cross.abs().max()),
        stored_product_scalars_before=6*1152,stored_product_scalars_after=5*1152,seconds=time.perf_counter()-start,
        scope='Actual weights, four synthetic contexts; diagonal/self response only. Five-bank generated from existing six-bank, so no preparation FLOP saving established. No native attention/denominator/suffix validation or global five-factor claim.')
    assert result['pred_a'] and result['pred_b'] and result['pred_c']
    (P/'RESPONSE_DIAGONAL_CONIC_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
