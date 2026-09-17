"""Specified local cross-term edit; all normalization/background terms retained."""
import torch

def retained_delta(complete,delta8,attention8,attention8_head2,g,mlp8):
    delta=delta8.double();omitted=(attention8-attention8_head2).double()
    L=mlp8['left'].double();R=mlp8['right'].double();D=mlp8['down'].double()
    s1=((g.double()+delta).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    removed=((delta@L.T)*(omitted@R.T)+(omitted@L.T)*(delta@R.T))@D.T/s1
    return complete.double()-removed
