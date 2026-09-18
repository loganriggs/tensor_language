"""Conditional city interchange from causally sufficient recipient/donor prefixes."""
import torch
from city_residual6_fields_v1 import interchange

def execute(program,recipient_residual6,recipient_tokens,donor_residual6,donor_tokens,city,destination):
    if recipient_residual6.shape[0]!=1 or donor_residual6.shape[0]!=1:raise ValueError('Only batch1 is certified')
    positions=torch.where(destination)[0]
    if not len(positions):raise ValueError('At least one destination required')
    stop=int(positions.max())+1
    if stop<=city or recipient_residual6.shape[1]!=stop or donor_residual6.shape[1]!=city+1:raise ValueError('Supply exactly the required causal prefixes')
    delta=interchange(program,recipient_residual6,recipient_tokens,donor_residual6,donor_tokens,city,destination[:stop])
    result=torch.zeros(1,len(destination),1152,dtype=delta.dtype,device=delta.device);result[:,:stop]=delta
    return result
