"""Exact symmetric bilinear dictionary fit with free input readers."""
import torch

def gram(left,right,other_left,other_right):
    return .5*((left@other_left.T)*(right@other_right.T)+
               (left@other_right.T)*(right@other_left.T))

def normalized(x):
    return x/x.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)

def solve(left,right,target_left,target_right,target_writers,target_energy):
    """Writers use [products,outputs]; loss gradient uses the envelope theorem.

    No ridge: dependent dictionaries fail Cholesky instead of silently changing
    the objective. Unit rows remove trivial scale freedom; permutation and
    left/right swaps remain representation symmetries.
    """
    l=normalized(left);r=normalized(right)
    g=gram(l,r,l,r)
    rhs=gram(l,r,target_left,target_right)@target_writers
    with torch.no_grad():
        writers=torch.cholesky_solve(rhs.detach(),torch.linalg.cholesky(g.detach()))
    loss=(target_energy+(writers*(g@writers)).sum()-2*(writers*rhs).sum())/target_energy
    return loss,writers,l,r

def control():
    torch.manual_seed(7131224);dtype=torch.float64
    tl=torch.randn(12,7,dtype=dtype);tr=torch.randn_like(tl);tw=torch.randn(12,4,dtype=dtype)
    energy=(tw*(gram(tl,tr,tl,tr)@tw)).sum()
    l=torch.randn(8,7,dtype=dtype,requires_grad=True);r=torch.randn_like(l,requires_grad=True)
    loss,w,ln,rn=solve(l,r,tl,tr,tw,energy)
    def dense(a,b,c):return torch.einsum('ko,kij->oij',c,.5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:]))
    target=dense(tl,tr,tw);fit=dense(ln,rn,w)
    oracle=float(abs(loss.detach()-(target-fit).square().sum()/target.square().sum()))
    dl=torch.randn_like(l);dr=torch.randn_like(r)
    gl,gr=torch.autograd.grad(loss,(l,r));analytic=(gl*dl).sum()+(gr*dr).sum()
    eps=1e-6
    plus=solve(l+eps*dl,r+eps*dr,tl,tr,tw,energy)[0]
    minus=solve(l-eps*dl,r-eps*dr,tl,tr,tw,energy)[0]
    fd=(plus-minus)/(2*eps);gradient_error=float(abs(fd-analytic)/analytic.abs().clamp_min(1e-12))
    exact=solve(tl,tr,tl,tr,tw,energy)[0]
    return dict(dense_loss_absolute_error=oracle,gradient_relative_error=gradient_error,
                full_dictionary_relative_squared_error=float(exact),pred_a=oracle<1e-10,
                pred_b=gradient_error<1e-6,pred_c=abs(float(exact))<1e-10)

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);result=control()
    Path(__file__).with_name('SYMMETRIC_PRODUCT_VARPRO_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
