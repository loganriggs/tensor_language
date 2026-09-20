"""Joint-tensor contraction for conditional normalized bilinear responses.

P maps input coordinates to residual deltas; Q supplies output readers.
Neither basis is inferred here. Baseline h and projected baseline write are charged.
"""
import torch


def compile_response(left, right, down, P, Q):
    """Contract output and input slots jointly, preserving mixed background terms."""
    C = Q.T @ down
    A, B = left @ P, right @ P
    mixed = (torch.einsum('ak,kp,kj->apj', C, A, right)
             + torch.einsum('ak,kp,kj->apj', C, B, left))
    core = torch.einsum('ak,kp,kq->apq', C, A, B)
    core = (core + core.transpose(-1, -2))/2
    return dict(mixed=mixed, core=core, carry=Q.T @ P,
                geometry=P.T @ P, input_basis=P, width=P.shape[0])


def evaluate(program, z, background, baseline_write, eps, include_quadratic=True):
    """Return Q^T[(h+Pz)+M(N(h+Pz)) - h-M(N(h))].

    baseline_write = Q^T D[(Lh)*(Rh)] / (mean(h^2)+eps), excluding output bias.
    Bias cancels in the response; the background generator still includes it.
    Works for nonorthogonal P: its Gram geometry is explicit.
    """
    s0 = background.square().mean(-1, keepdim=True) + eps
    overlap = background @ program['input_basis']
    shift = 2*(overlap*z).sum(-1, keepdim=True)
    shift = shift + torch.einsum('...p,pq,...q->...', z, program['geometry'], z)[...,None]
    s1 = s0 + shift/program['width']
    numerator = torch.einsum('apj,...p,...j->...a', program['mixed'], z, background)
    if include_quadratic:
        numerator = numerator + torch.einsum('apq,...p,...q->...a', program['core'], z, z)
    return z @ program['carry'].T + numerator/s1 + baseline_write*(s0/s1-1)


def prepare_context(program, background, baseline_write, eps):
    """Close full-vector background ports to sufficient scalar coefficients.

    Preparation still requires the background generator and fixed mixed tensor.
    The returned coefficients can be reused across response coordinates z.
    """
    return dict(linear=torch.einsum('apj,...j->...ap', program['mixed'], background),
                overlap=background@program['input_basis'],
                s0=background.square().mean(-1,keepdim=True)+eps,
                baseline_write=baseline_write)


def evaluate_prepared(program, z, context):
    shift=2*(context['overlap']*z).sum(-1,keepdim=True)
    shift=shift+torch.einsum('...p,pq,...q->...',z,program['geometry'],z)[...,None]
    s1=context['s0']+shift/program['width']
    numerator=torch.einsum('...ap,...p->...a',context['linear'],z)
    numerator=numerator+torch.einsum('apq,...p,...q->...a',program['core'],z,z)
    return z@program['carry'].T+numerator/s1+context['baseline_write']*(context['s0']/s1-1)


def prepare_readout(background, basis, readout, eps):
    """Final RMS/softcap needs dot products and norm geometry, not the full state."""
    fixed=dict(reader=readout@basis,geometry=basis.T@basis,width=basis.shape[0])
    context=dict(raw_background=background@readout.T,overlap=background@basis,
                 s0=background.square().mean(-1,keepdim=True)+eps)
    return fixed,context


def readout_prepared(fixed,z,context):
    shift=2*(context['overlap']*z).sum(-1,keepdim=True)
    shift=shift+torch.einsum('...p,pq,...q->...',z,fixed['geometry'],z)[...,None]
    scale=(context['s0']+shift/fixed['width']).sqrt()
    raw=context['raw_background']+z@fixed['reader'].T
    return 30*torch.tanh(raw/scale/30)
