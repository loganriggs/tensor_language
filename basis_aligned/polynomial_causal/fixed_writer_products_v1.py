"""Share the fixed response writer across pristine contexts."""
import torch

def prepare_global(writer,left,right,down,compile_mixed=False):
    lw=left@writer;rw=right@writer
    result=dict(writer=writer,left_writer=lw,right_writer=rw,self_product=2*(lw*rw)@down.T)
    if compile_mixed:result['mixed_map']=down@(rw[:,None]*left+lw[:,None]*right)
    return result

def products(basis,left,right,down,global_state,contiguous_private=True):
    assert torch.allclose(basis[...,0,:],global_state['writer'].expand_as(basis[...,0,:]),atol=0,rtol=0)
    private=basis[...,1:,:]
    if contiguous_private:private=private.contiguous()
    l=private@left.T;r=private@right.T
    l1,l2=l.unbind(-2);r1,r2=r.unbind(-2)
    self0=global_state['self_product'].expand_as(basis[...,0,:])
    if 'mixed_map' in global_state:
        cross=private@global_state['mixed_map'].T
        p01,p02=cross.unbind(-2)
        p11,p12,p22=(torch.stack([2*l1*r1,l1*r2+l2*r1,2*l2*r2],-2)@down.T).unbind(-2)
    else:
        lw=global_state['left_writer'];rw=global_state['right_writer']
        p01,p02,p11,p12,p22=(torch.stack([lw*r1+l1*rw,lw*r2+l2*rw,2*l1*r1,l1*r2+l2*r1,2*l2*r2],-2)@down.T).unbind(-2)
    return torch.stack([self0,p01,p02,p11,p12,p22],-2)
