"""Exact finite-difference operator for RMS followed by an ungated bilinear MLP."""
import torch


def rms_secant(x,y,vector,eps):
    """Apply a symmetric diagonal-minus-rank-one secant map; no dense D-by-D matrix."""
    sx=(x.square().mean(-1,keepdim=True)+eps).sqrt()
    sy=(y.square().mean(-1,keepdim=True)+eps).sqrt()
    middle=(x+y)/2
    scale=(1/sx+1/sy)/2
    rank_one=2/(x.shape[-1]*sx*sy*(sx+sy))
    return scale*vector-rank_one*middle*(middle*vector).sum(-1,keepdim=True)


def mlp_secant(x,y,vector,left,right,down,eps):
    normalize=lambda z:z/(z.square().mean(-1,keepdim=True)+eps).sqrt()
    middle=(normalize(x)+normalize(y))/2
    dn=rms_secant(x,y,vector,eps)
    return ((middle@left.T)*(dn@right.T)+(dn@left.T)*(middle@right.T))@down.T


def controls():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(60927)
    rand=lambda *s:torch.randn(*s,generator=g,dtype=torch.float64)
    x=rand(4,6);y=rand(4,6);y[0]=x[0];y[1]=-x[1];y[2]=0
    eps=.02;left=rand(9,6);right=rand(9,6);down=rand(8,9);bias=rand(8)
    normalize=lambda z:z/(z.square().mean(-1,keepdim=True)+eps).sqrt()
    mlp=lambda z:((normalize(z)@left.T)*(normalize(z)@right.T))@down.T+bias
    norm_error=float((rms_secant(x,y,y-x,eps)-(normalize(y)-normalize(x))).abs().max())
    mlp_error=float((mlp_secant(x,y,y-x,left,right,down,eps)-(mlp(y)-mlp(x))).abs().max())
    # A generic finite step differs from the tangent at just the receiving state.
    tangent=mlp_secant(x,x,y-x,left,right,down,eps)
    finite_vs_tangent=float((tangent-(mlp(y)-mlp(x))).norm())
    a=rand(4,6);b=rand(4,6)
    symmetry=float(((a*rms_secant(x,y,b,eps)).sum()-(b*rms_secant(x,y,a,eps)).sum()).abs())
    checks={'rms_finite_difference':norm_error<1e-11,'bilinear_finite_difference':mlp_error<1e-10,'secant_symmetric':symmetry<1e-11,'finite_step_is_not_receiving_tangent':finite_vs_tangent>.1,'identical_endpoint_zero':bool((mlp_secant(x,x,torch.zeros_like(x),left,right,down,eps)==0).all())}
    return {'passed':all(checks.values()),'checks':checks,'rms_error':norm_error,'mlp_error':mlp_error,'symmetry_error':symmetry,'finite_vs_tangent_error':finite_vs_tangent,'model_forwards':0,'scope':'Exact-real pair-conditioned operator with both native endpoints supplied; useful for accounting finite intervention propagation. Not a fixed shared task operator, independent prediction, or structural reduction.'}


if __name__=='__main__':
    import json
    from pathlib import Path
    result=controls();assert result['passed']
    with Path(__file__).with_name('NORMALIZED_BILINEAR_SECANT_V1_CONTROLS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
