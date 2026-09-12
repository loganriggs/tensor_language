"""Four-port source-block execution and anchored finite intervention algebra."""
import torch

def write(gate,parent,children,writers):
    # [N,H], [N], [N,2], [N,H,2,O] -> [N,O]
    return torch.einsum('nh,n,nj,nhjo->no',gate,parent,children,writers)

def vertices(base,donor,fn=write):
    return torch.stack([fn(*[donor[j] if mask&(1<<j) else base[j] for j in range(len(base))]) for mask in range(1<<len(base))])

def mobius(values):
    result=values.clone();count=len(values);assert count>0 and count&(count-1)==0
    for bit in range(count.bit_length()-1):
        for mask in range(count):
            if mask&(1<<bit):result[mask]=result[mask]-result[mask^(1<<bit)]
    return result

def reconstruct(coefficients):
    result=coefficients.clone();count=len(result)
    for bit in range(count.bit_length()-1):
        for mask in range(count):
            if mask&(1<<bit):result[mask]=result[mask]+result[mask^(1<<bit)]
    return result
