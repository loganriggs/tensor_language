"""Registered control: exact response/product <=1e-10, zero edits exact;
FP32 six-product error <=1e-5 vs FP64 on regular signed grid. Compare fair
vectorized ten/six preparation and per-pair execution; timing descriptive.
"""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from directional_mlp_response_context_v1 import evaluate
from response_product_basis_v2 import prepare_basis,coefficients,prepare_products,combine
P=Path(__file__).resolve().parent

def timing(fn):
    fn();fn();samples=[]
    for _ in range(7):
        start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
    return statistics.median(samples)

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(61334);start=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    z=torch.randn(4,1152,dtype=torch.float64);l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    base=((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    lam=float(sd['transformer.h.10.lambdas'][0]);ctx,basis=prepare_basis(z,base,program,lam)
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    pairs,products=prepare_products(basis,l,r,d)
    oldbasis=torch.stack([ctx['direction'].expand_as(z),base,ctx['Jz'],ctx['Jw'].expand_as(z)],1)*lam
    oldpairs,oldproducts=prepare_products(oldbasis,l,r,d)
    def oldcoeff(a):
        rho=ctx['perpendicular_rms']+ctx['writer_rms']*(a-ctx['parallel']).square()
        return torch.cat([-a,a*(2*ctx['cross_rms']-a*ctx['writer_rms'])/rho,-a/rho,a.square()/(2*rho)],-1)
    def cross(x,y):return ((x@l.T)*(y@r.T)+(y@l.T)*(x@r.T))@d.T
    errors=dict(response=0.,product=0.,old_replay=0.,fp32=0.,mixed=0.);norms=dict(response=0.,product=0.,mixed=0.);zero=0.
    strengths=[-1.,-.5,-.2,-.05,0.,.03,.1,.25,.5,.75,1.];inputs=[]
    for a0 in strengths:
        a=torch.full((4,1),a0,dtype=z.dtype);u=coefficients(a,ctx);x=lam*evaluate(a,ctx)
        pred=torch.einsum('bi,bid->bd',u,basis)
        errors['response']+=float((pred-x).square().sum());norms['response']+=float(x.square().sum())
        for b0 in strengths:
            b=torch.full_like(a,b0);v=coefficients(b,ctx);y=lam*evaluate(b,ctx);ref=cross(x,y)
            fitted=combine(u,v,pairs,products);old=combine(oldcoeff(a),oldcoeff(b),oldpairs,oldproducts)
            fp32=combine(u.float(),v.float(),pairs,products.float()).double()
            du=coefficients(a+b,ctx)-u-v
            mixed=du[:,1:]@torch.eye(2,dtype=z.dtype) # retain only two nonlinear coefficients
            mixpred=torch.einsum('bi,bid->bd',mixed,basis[:,1:])
            mixref=lam*(evaluate(a+b,ctx)-evaluate(a,ctx)-evaluate(b,ctx))
            errors['mixed']+=float((mixpred-mixref).square().sum());norms['mixed']+=float(mixref.square().sum())
            for name,actual in [('product',fitted),('old_replay',old),('fp32',fp32)]:errors[name]+=float((actual-ref).square().sum())
            norms['product']+=float(ref.square().sum());inputs.append((u,v,a,b))
            if a0==0 or b0==0:zero=max(zero,float(fitted.abs().max()))
    rel={k:(v/norms[k if k in norms else 'product'])**.5 for k,v in errors.items()}
    assert max(rel[k] for k in ['response','product','mixed','old_replay'])<1e-10 and zero==0 and rel['fp32']<1e-5
    prep6=timing(lambda:prepare_products(basis,l,r,d));prep10=timing(lambda:prepare_products(oldbasis,l,r,d))
    eval6=timing(lambda:[combine(u,v,pairs,products) for u,v,a,b in inputs])
    eval10=timing(lambda:[combine(oldcoeff(a),oldcoeff(b),oldpairs,oldproducts) for u,v,a,b in inputs])
    # Fair contraction-only comparison: precompute coefficient arrays on both sides.
    oldinputs=[(oldcoeff(a),oldcoeff(b)) for u,v,a,b in inputs]
    contraction10=timing(lambda:[combine(u,v,oldpairs,oldproducts) for u,v in oldinputs])
    out=dict(relative_errors=rel,zero_edit_maxabs=zero,response_basis_count=3,mixed_response_basis_count=2,symmetric_product_count=6,product_scalars_per_context=6*1152,previous_product_scalars=10*1152,contexts=4,strength_pairs=121,preparation_seconds_six=prep6,preparation_seconds_ten=prep10,contraction_grid_seconds_six=eval6,contraction_grid_seconds_ten=contraction10,ten_grid_including_coefficients_seconds=eval10,seconds=time.perf_counter()-start,scope='Exact context-dependent response-family simplification. Actual weights and synthetic signed edits; normalized response coefficients exact, attention and MLP10 joint denominator/background/suffix external. Six-vector bank is per context, not global factors or six identified circuits.')
    (P/'RESPONSE_PRODUCT_BASIS_V2_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
