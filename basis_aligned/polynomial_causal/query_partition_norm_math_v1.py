"""Separate local numerator cross terms from RMS-dependent interaction."""
import json
from pathlib import Path
import torch
import source_gain_attention as M


def numerator(bank,z):
    scores=torch.einsum('bs,bijst->bijt',z,bank['factors'])
    return torch.einsum('bjt,btjh->bjh',scores[:,0]*scores[:,1]/bank['width']**2,bank['values'])


def partition(bank,left,right):
    full=left+right;zero=torch.zeros_like(full)
    interaction=M.evaluate(bank,full)-M.evaluate(bank,left)-M.evaluate(bank,right)+M.evaluate(bank,zero)
    norms=torch.einsum('bs,bijst,bt->bij',full,bank['norm'],full)+bank['eps2']
    denominator=(norms[:,0]*norms[:,1]).sqrt()[:,:,None]
    cross=(numerator(bank,full)-numerator(bank,left)-numerator(bank,right))/denominator
    return {'total':interaction,'numerator_cross_at_full_norm':cross,'normalization_remainder':interaction-cross}


if __name__=='__main__':
    torch.set_num_threads(2);d=torch.float64
    sources=torch.eye(2,dtype=d)[None]
    readers=torch.eye(2,dtype=d).expand(2,1,2,2)
    keys=torch.eye(2,dtype=d)[None,None,:,None,:].expand(1,2,2,1,2)
    values=torch.tensor([[[[1.,0.]],[[2.,0.]]]],dtype=d)
    bank=M.prepare(sources,readers,keys,values,torch.ones(1,1,dtype=d),torch.zeros(1,1,dtype=d),[1],.01)
    left=torch.tensor([[1.,0.]],dtype=d);right=1-left
    parts=partition(bank,left,right)
    assert float(parts['numerator_cross_at_full_norm'].abs().max())==0.
    assert float(parts['total'].norm())>.1
    assert torch.equal(parts['total'],parts['numerator_cross_at_full_norm']+parts['normalization_remainder'])
    result={'passed':True,'parts':{k:v.tolist() for k,v in parts.items()},'model_forwards':0,'scope':'Planted source-gain bank: zero numerator cross coefficient yet nonzero source/rest interaction from RMS. This partitions the local head-read frame using the full-gain norm as an explicit reference; it does not identify the source of trained-model final-logit interaction.'}
    with Path(__file__).with_name('QUERY_PARTITION_NORM_MATH_V1_CONTROLS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
