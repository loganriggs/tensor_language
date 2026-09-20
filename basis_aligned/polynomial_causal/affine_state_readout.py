"""Three affine state vectors ->54numerator+6Gram fields for nine contrasts."""
import torch
from final_readout_field_program import evaluate_fields

def compile(basis,rows):
    # basis[B,3,D], token rows[B,O,2,D]
    numerators=torch.einsum('bkd,bopd->bkop',basis.double(),rows.double()).flatten(2)
    gram=basis.double()@basis.double().transpose(1,2)/basis.shape[-1]
    ids=torch.triu_indices(3,3,device=basis.device)
    return numerators,gram[:,ids[0],ids[1]]

def evaluate(program,s,a):
    nums,upper=program;coords=torch.tensor([1.,s,a],device=nums.device,dtype=nums.dtype);idx=torch.triu_indices(3,3,device=nums.device)
    weights=coords[idx[0]]*coords[idx[1]]*torch.where(idx[0]==idx[1],1.,2.)
    q=(upper*weights).sum(-1,keepdim=True)+torch.finfo(torch.float32).eps
    return evaluate_fields(torch.cat([torch.einsum('k,bko->bo',coords,nums),q],-1))
