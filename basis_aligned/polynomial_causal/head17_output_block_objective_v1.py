"""Exact variable projection for output-rotated low-rank interaction blocks."""
from pathlib import Path
import json
import torch
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent


def build(device='cpu'):
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    pairs=sorted(set((r['uk_id'],r['us_id']) for r in rows))
    ids=[i for pair in pairs for i in pair];assert len(set(ids))==12
    O=sd['lm_head.weight'][ids].to(device=device,dtype=torch.float64)
    W=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].to(device=device,dtype=torch.float64)
    L,R,D=[sd['transformer.h.17.mlp.'+k+'.weight'].to(device=device,dtype=torch.float64) for k in ('Left','Right','Down')]
    C=O@D;LW,RW=L@W,R@W
    T=torch.stack([L.T@(c[:,None]*RW)+R.T@(c[:,None]*LW) for c in C])
    return T,ids


class Objective:
    def __init__(self,tensor,rank=32):
        self.scale=float(tensor.norm());self.tensor=tensor/self.scale
        self.rank=rank;self.n=tensor.shape[0]
        self.indices=torch.triu_indices(self.n,self.n,offset=1,device=tensor.device)
        self.flat=self.tensor.flatten(1)

    def rotate(self,theta,base):
        skew=theta.new_zeros(self.n,self.n)
        skew[self.indices[0],self.indices[1]]=theta
        return base@torch.matrix_exp(skew-skew.T)

    @torch.no_grad()
    def evaluate(self,Q):
        mixed=(Q.T@self.flat).reshape_as(self.tensor)
        u,s,vh=torch.linalg.svd(mixed,full_matrices=False)
        fitted=(u[:,:,:self.rank]*s[:,:self.rank,None].transpose(1,2))@vh[:,:self.rank,:]
        residual=mixed-fitted
        loss=float(residual.square().sum())
        gradient=2*self.flat@residual.flatten(1).T
        tangent=(Q.T@gradient-gradient.T@Q)/2
        return loss,gradient,float(tangent.norm()),fitted

    def value_gradient(self,theta,base):
        x=torch.as_tensor(theta,dtype=self.tensor.dtype,device=self.tensor.device).requires_grad_(True)
        Q=self.rotate(x,base)
        value,gq,intrinsic,_=self.evaluate(Q.detach())
        gradient=torch.autograd.grad(Q,x,grad_outputs=gq)[0]
        return value,gradient.detach().cpu().numpy(),intrinsic
