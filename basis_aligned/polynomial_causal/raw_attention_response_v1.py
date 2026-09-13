"""Full squared attention from raw residual projections, preserving both QK factors."""
import torch

EPS = torch.finfo(torch.float32).eps


def execute(projections, rho, first_values, mixture, output_matrix, heads=9):
    """Five [B,T,d] raw projections; rho [B,T,1]; first_values [B,T,d].

    rho includes EPS. No softmax; native BF16 rotary tables. The output is
    double precision algebra, not a bitwise emulation of FP32 native execution.
    """
    B, T, d = projections[0].shape
    width = d//heads
    q, k, q2, k2, value = [p.reshape(B,T,heads,width) for p in projections]
    inv = 1/(10000**(torch.arange(0,width,2,device=q.device).float()/width))
    angles = torch.arange(T,device=q.device).float()[:,None]*inv[None,:]
    cos = angles.cos().bfloat16().to(q)[None,:,None,:]
    sin = angles.sin().bfloat16().to(q)[None,:,None,:]

    def normalized_rotated(x):
        x = x/(x.square().mean(-1,keepdim=True)+EPS*rho[...,None]).sqrt()
        x1, x2 = x.chunk(2,dim=-1)
        return torch.cat([x1*cos+x2*sin,-x1*sin+x2*cos],dim=-1)

    q,k,q2,k2 = [normalized_rotated(x) for x in (q,k,q2,k2)]
    a = torch.einsum('bthd,bshd->bhts',q,k)/width
    b = torch.einsum('bthd,bshd->bhts',q2,k2)/width
    pattern = (a*b).tril()
    values = (1-mixture)*value/rho[...,None].sqrt()+mixture*first_values.reshape_as(value)
    head_writes = torch.einsum('bhts,bshd->bthd',pattern,values)
    return head_writes.reshape(B,T,d)@output_matrix.T


def prepare(raw, matrices):
    raw = raw.double()
    return tuple(raw@m.double().T for m in matrices), raw.square().mean(-1,keepdim=True)+EPS


def changed(raw, delta, baseline_projections, matrices):
    raw,delta = raw.double(),delta.double()
    projections = tuple(p+delta@m.double().T for p,m in zip(baseline_projections,matrices))
    # Direct norm avoids cancellation when delta nearly cancels the background.
    rho = (raw+delta).square().mean(-1,keepdim=True)+EPS
    return projections,rho
