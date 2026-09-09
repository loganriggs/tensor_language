"""Small exact-form reference: shared bilinear readers, norm scalar, legal edits.

This is a module-level quotient with externally supplied additive writer edits.
It does not replace the native prefix, identify semantic features, or prove a
small quotient exists in the trained model.
"""
import torch


def contract(left,right,down,readers):
    raw=torch.einsum('oh,hi,hj->oij',readers@down,left,right)
    return (raw+raw.transpose(-1,-2))/2


def compile_core(forms,basis,writers):
    core=torch.einsum('ia,oij,jb->oab',basis,forms,basis)
    folded=basis.T@writers
    return {'basis':basis,'core':core,'writers':folded,'writer_gram':writers.T@writers,
            'form_closure':float((torch.einsum('ia,oab,jb->oij',basis,core,basis)-forms).abs().max()),
            'writer_closure':float((basis@folded-writers).abs().max())}


def encode(x,program):return x@program['basis'],x.square().sum(-1)


def evaluate(state,program,width,epsilon,bias):
    c,rho=state
    return torch.einsum('...i,oij,...j->...o',c,program['core'],c)/(rho[...,None]/width+epsilon)+bias


def edit(state,amplitudes,program):
    c,rho=state;delta=amplitudes@program['writers'].T
    energy=torch.einsum('...i,ij,...j->...',amplitudes,program['writer_gram'],amplitudes)
    return c+delta,rho+2*(c*delta).sum(-1)+energy


def controls():
    dtype=torch.float64;width=64;eps=torch.finfo(torch.float32).eps
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60912)
        rotation,_=torch.linalg.qr(torch.randn(width,width,dtype=dtype))
        # Outputs: task A=(x0*x1,x3^2), task B=(x0*x2,2*x3^2).
        left=torch.zeros(3,width,dtype=dtype);right=left.clone()
        left[0,0]=left[1,0]=left[2,3]=1;right[0,1]=right[1,2]=right[2,3]=1
        down=torch.tensor([[1,0,0],[0,0,1],[0,1,0],[0,0,2]],dtype=dtype)
        left=left@rotation.T;right=right@rotation.T
        forms=contract(left,right,down,torch.eye(4,dtype=dtype))
        writers=rotation[:,[1,2]]+rotation[:,[4,5]]
        basis=rotation[:,:6];program=compile_core(forms,basis,writers)
        x=torch.randn(17,width,dtype=dtype);amps=torch.randn(17,2,dtype=dtype)
    bias=torch.tensor([.2,-.3,.1,.4],dtype=dtype)
    def direct(x):
        n=x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
        return ((n@left.T)*(n@right.T))@down.T+bias
    state=encode(x,program);errors=[]
    for a in (torch.zeros_like(amps),amps,amps*torch.tensor([1.,0.]),amps*torch.tensor([0.,1.])):
        errors.append(float((evaluate(edit(state,a,program),program,width,eps,bias)-direct(x+a@writers.T)).abs().max()))
    first=edit(state,amps*torch.tensor([1.,0.]),program)
    joint=edit(first,amps*torch.tensor([0.,1.]),program)
    direct_joint=edit(state,amps,program)
    extra=x+3*rotation[:,6]
    c_extra,rho_extra=encode(extra,program)
    frozen_norm=evaluate((c_extra,state[1]),program,width,eps,bias)
    # Equal numerator coordinates and energy need extra writer projections for edits.
    base=rotation[:,0]+rotation[:,1]+rotation[:,3]
    xp=base+rotation[:,4];xm=base-rotation[:,4]
    too_small=rotation[:,:4]
    equal_old=torch.allclose(xp@too_small,xm@too_small,atol=1e-12,rtol=1e-12) and abs(float(xp.square().sum()-xm.square().sum()))<1e-12
    checks={'full_direct_and_four_edits':max(errors)<1e-9,
            'weight_and_writer_closure':program['form_closure']<1e-12 and program['writer_closure']<1e-12,
            'joint_update_composes':all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(joint,direct_joint)),
            'discarded_direction_preserves_numerator':torch.allclose(c_extra,state[0],atol=1e-12,rtol=1e-12),
            'frozen_norm_negative':float((frozen_norm-direct(extra)).abs().max())>1e-3,
            'missing_edit_projection_negative':equal_old and float((direct(xp+writers[:,0])-direct(xm+writers[:,0])).abs().max())>1e-3,
            'shared_quadratic':torch.allclose(forms[3],2*forms[1],atol=1e-12,rtol=1e-12),
            'different_products_shared_linear_input':torch.allclose(left[0],left[1]) and not torch.allclose(forms[0],forms[2])}
    return {'passed':all(checks.values()),'checks':checks,'max_full_output_error':max(errors),
            'native_state_width':width,'retained_linear_coordinates':6,'retained_norm_scalars':1,
            'scope':'Planted rotated module with two reader families and independent additive edits; no trained reduction claim.'}
