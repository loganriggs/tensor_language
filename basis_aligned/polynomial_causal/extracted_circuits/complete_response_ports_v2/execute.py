"""Standalone conditional head8.2 -> MLP8 -> head9.8 response (FP64 CPU).

Input generation, native suffix and the rank-64 approximation are external.
The only numerical dependency is PyTorch. No repository imports or callbacks.
"""
from pathlib import Path
import torch

EPS = torch.finfo(torch.float32).eps


def load(path=None):
    p = torch.load(path or Path(__file__).with_name('program.pt'),
                   map_location='cpu', weights_only=True)
    r, left, d = p['readers'], p['left'], p['direction']
    p.update(read_left=r @ left, read_direction=r @ d, gram=left.T @ left,
             left_direction=left.T @ d, direction_norm2=d @ d,
             right_direction=p['right'] @ d)
    keys = torch.cat([r[128:256], r[384:512]])
    p['inside_adapters'] = ((keys @ keys.T) @ p['key_coordinates']).reshape(2, 128, 64)
    return p


def project(z, h, u, p):
    """Project supplied pristine MLP input, attention input and bias-free MLP output."""
    z, h, u = z.double(), h.double(), u.double()
    return dict(bz=z @ p['right'].T, zd=z @ p['direction'], z2=z.square().sum(-1),
                ch=h @ p['readers'].T, ath=h @ p['left'], hd=h @ p['direction'],
                h2=h.square().sum(-1), cu=u @ p['readers'].T, atu=u @ p['left'],
                ud=u @ p['direction'], u2=u.square().sum(-1), hu=(h*u).sum(-1))


def execute(ports, amplitude, p):
    """Return head scalar and attention-input squared RMS, including compensation."""
    a, gain = amplitude.double()[..., None], p['gain']
    r = ports['z2'][..., None]/1152 + EPS
    rm = (ports['z2'][..., None] - 2*a*ports['zd'][..., None]
          + a.square()*p['direction_norm2'])/1152 + EPS
    beta = r/rm - 1
    c = -a/rm*(ports['bz'] - a/2*p['right_direction'])
    reads = ports['ch'] + gain*(-a*p['read_direction'] + beta*ports['cu']
                               + c @ p['read_left'].T)
    hw = -a*ports['hd'][..., None] + (ports['ath']*c).sum(-1, keepdim=True)
    uw = -a*ports['ud'][..., None] + (ports['atu']*c).sum(-1, keepdim=True)
    ww = (a.square()*p['direction_norm2']
          - 2*a*(c*p['left_direction']).sum(-1, keepdim=True)
          + (c*(c @ p['gram'])).sum(-1, keepdim=True))
    norm = (ports['h2'][..., None] + 2*gain*beta*ports['hu'][..., None]
            + gain.square()*beta.square()*ports['u2'][..., None]
            + 2*gain*(hw + gain*beta*uw) + gain.square()*ww)
    rho2 = norm/1152 + EPS
    shared = torch.cat([reads[..., 128:256], reads[..., 384:512]], -1) @ p['key_coordinates']
    n = reads.shape[-2]
    inv = 1/(10000**(torch.arange(0, 128, 2, dtype=torch.float32)/128))
    angle = torch.outer(torch.arange(n, dtype=torch.float32), inv)
    co, si = angle.cos().bfloat16(), angle.sin().bfloat16()

    def rotate(x):
        a, b = x.chunk(2, -1)
        return torch.cat((a*co + b*si, -a*si + b*co), -1)

    full, reflected = [], []
    for j, (qi, ki) in enumerate([(0, 128), (256, 384)]):
        q, k = reads[..., qi:qi+128], reads[..., ki:ki+128]
        inside = shared @ p['inside_adapters'][j].T
        q = rotate(q/(q.square().mean(-1, keepdim=True) + EPS*rho2).sqrt())
        den = (k.square().mean(-1, keepdim=True) + EPS*rho2).sqrt()
        full.append(q @ rotate(k/den).transpose(-1, -2)/128)
        reflected.append(q @ rotate((k-2*inside)/den).transpose(-1, -2)/128)
    gamma = (full[0]*full[1] + reflected[0]*reflected[1])/2
    gamma = gamma.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), 0)
    return (gamma @ (reads[..., -1:]/rho2.sqrt()))[..., 0], rho2
