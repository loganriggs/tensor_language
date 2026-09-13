"""Projection-interface identity versus direct native-form head arithmetic."""
from pathlib import Path
import json
import torch
from additive_head_raw_ports_v1 import EPS, project, from_projections
from three_corner_head_interface_v1 import execute
from joint_attention_mixed_ports_v1 import decompose
from joint_attention_three_group_v1 import execute as three_group
from head17_source_interface_v1 import CHECKPOINT

P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(314)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    matrices=[sd[f'transformer.h.17.attn.{k}.weight'].reshape(9,128,1152)[2].double()
              for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    mu=float(sd['transformer.h.17.attn.lamb'])
    n=torch.randn(4,17,1152,dtype=torch.float64)
    c=n+0.15*torch.randn_like(n);r=n+0.25*torch.randn_like(n);a=c+r-n
    raw=[n,c,r];projections=[project(x,matrices) for x in raw]
    norms=[x.square().mean(-1)+EPS for x in raw]
    cross=((c-n)*(r-n)).mean(-1)
    rhoa=a.square().mean(-1)+EPS
    normerror=float((norms[1]+norms[2]-norms[0]+2*cross-rhoa).norm()/rhoa.norm())
    first=torch.randn(4,17,128,dtype=torch.float64)
    ports=[from_projections(project(x,matrices),x.square().mean(-1)+EPS,first,mu) for x in [n,c,r,a]]
    parts=decompose(*ports);reference=parts['cross']+parts['defect'];small=three_group(*ports)
    full=execute(projections,norms,cross,first,mu)
    compact=execute(projections,norms,cross,first,mu,compact=True)
    result=dict(norm_closure_error=normerror,full_mixed_error=float((full-reference).norm()/reference.norm()),compact_mixed_error=float((compact-small).norm()/small.norm()),scope='Actual-weight synthetic algebra. Cross-norm scalar is supplied, not inferred from projection vectors. No native behavioral or autonomous input-generation claim.')
    assert max(result[k] for k in ('norm_closure_error','full_mixed_error','compact_mixed_error'))<1e-10
    (P/'THREE_CORNER_HEAD_INTERFACE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
