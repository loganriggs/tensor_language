"""Exact query-row correction after the fixed-query source-column update."""
import torch

def query_row(q10,q20,q11,q21,keys1,keys2,values,output,positions):
    """Queries[B,H,E]; updated full K/V[B,T,H,E]; result[B,D].

    The column update must already use the updated source K/V. Using baseline
    source K/V here would drop the query-source mixed interaction.
    """
    b,t,h,e=keys1.shape
    def route(q1,q2):
        return torch.einsum('bhe,bthe->bth',q1,keys1)*torch.einsum('bhe,bthe->bth',q2,keys2)/e**2
    delta=route(q11,q21)-route(q10,q20)
    delta*= (torch.arange(t,device=values.device)[None]<=positions[:,None])[:,:,None]
    heads=torch.einsum('bth,bthe->bhe',delta,values)
    return heads.flatten(1)@output.T
