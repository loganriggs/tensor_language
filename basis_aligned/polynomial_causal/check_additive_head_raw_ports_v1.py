"""Actual-weight algebra control and normalization-information falsifier."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from additive_head_raw_ports_v1 import EPS, project, from_projections, additive
from folded_normalized_router_v1 import rotary
from head17_source_interface_v1 import CHECKPOINT

P = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    torch.manual_seed(310)
    sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    matrices = [sd[f'transformer.h.17.attn.{k}.weight'].reshape(9,128,1152)[2].double()
                for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    mu = float(sd['transformer.h.17.attn.lamb'])
    n = torch.randn(3, 13, 1152, dtype=torch.float64)
    c = n+0.2*torch.randn_like(n)
    r = n+0.3*torch.randn_like(n)
    raw = c+r-n
    first = torch.randn(3, 13, 128, dtype=torch.float64)
    rho = raw.square().mean(-1)+EPS
    candidate = from_projections(additive(project(n, matrices), project(c, matrices), project(r, matrices)), rho, first, mu)
    normalized = F.rms_norm(raw, (1152,), eps=EPS)
    q1,k1,q2,k2,v = project(normalized, matrices)
    rotations = torch.stack([rotary(t,128) for t in range(13)])
    def score(q,k):
        q = F.rms_norm(q,(128,),eps=EPS)
        k = F.rms_norm(k,(128,),eps=EPS)
        return torch.stack([((q[:,-1]@rotations[-1].T)*(k[:,t]@rotations[t].T)).sum(-1)/128 for t in range(13)],-1)
    reference = (score(q1,k1), score(q2,k2), (1-mu)*v+mu*first)
    errors = [float((x-y).norm()/y.norm()) for x,y in zip(candidate,reference)]
    # Same five projections but different residual norm: omitted norm is not identifiable.
    bank = torch.cat(matrices)
    q,_ = torch.linalg.qr(bank.T, mode='reduced')
    u = torch.randn(1152,dtype=torch.float64)
    u = u-q@(q.T@u)
    u = u/u.norm()*raw[0,0].norm()
    base = raw[:1,:1]
    changed = base+3*u
    pn,pc = project(base,matrices),project(changed,matrices)
    projection_error = float((torch.cat(pn,-1)-torch.cat(pc,-1)).norm()/torch.cat(pn,-1).norm())
    a = from_projections(pn,base.square().mean(-1)+EPS,first[:1,:1],mu)
    b = from_projections(pc,changed.square().mean(-1)+EPS,first[:1,:1],mu)
    value_change = float((a[2]-b[2]).norm()/a[2].norm())
    result = dict(port_relative_errors=errors, nullspace_projection_error=projection_error,
                  nullspace_value_relative_change=value_change,
                  scope='FP64 actual-weight algebra on synthetic states. Reuses known nested-RMS identity. Additive norm remains required; no native FP32 replay or reduced fullmodel forward count yet.')
    assert max(errors)<1e-10 and projection_error<1e-10 and value_change>0.1
    (P/'ADDITIVE_HEAD_RAW_PORTS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
