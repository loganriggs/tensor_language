"""Exact (query degree2, source degree3) attention numerator contraction.

q: [batch,2,dq], s: [batch,3,ds]. Read maps: [heads,width,input].
output: [dout,heads,width]. gates: [batch,heads], optional external factors.
Rotary positions may be compiled into query/key maps. Normalizers, masking and
score scaling must be supplied explicitly through gates; no softmax is assumed.
"""
import itertools
import torch

def contract(q,s,query1,key1,query2,key2,values,output,gates=None):
    a=torch.einsum('nai,hki->nhak',q,query1)
    b=torch.einsum('nai,hki->nhak',q,query2)
    ka=torch.einsum('nbi,hki->nhbk',s,key1)
    kb=torch.einsum('nbi,hki->nhbk',s,key2)
    v=torch.einsum('nbi,hki->nhbk',s,values)
    score1=torch.einsum('nhak,nhbk->nhab',a,ka)
    score2=torch.einsum('nhak,nhbk->nhab',b,kb)
    result=torch.zeros_like(v[:,:,0])
    for i,j,k in itertools.permutations(range(3)):
        scalar=(score1[:,:,0,i]*score2[:,:,1,j]+score1[:,:,1,i]*score2[:,:,0,j])/12
        result=result+scalar[...,None]*v[:,:,k]
    if gates is not None:result=result*gates[...,None]
    return torch.einsum('nhk,ohk->no',result,output)
