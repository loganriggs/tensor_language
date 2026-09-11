"""Shared quadratic nodes across output-sharing LL1 groups.

Reader bank -> squares -> sparse signed group mixtures -> output writes.
Canonicalize each group before proposing cross-group square merges.
"""
import torch
from symmetric_ll1_projected_v1 import evaluate as projected_evaluate


def canonicalize(a,s,c):
    q,r=torch.linalg.qr(a.transpose(1,2),mode='reduced')
    core=(r*s[:,None,:])@r.transpose(1,2)
    values,vectors=torch.linalg.eigh(core)
    return (q@vectors).transpose(1,2),values,c


def merge_readers(a,s,c,threshold):
    m,r,d=a.shape;flat=a.reshape(-1,d)
    similarity=(flat@flat.T).abs()
    energy=(s.square()*c.square().sum(1)[:,None]).flatten()
    order=torch.argsort(energy,descending=True,stable=True).tolist()
    unused=set(order);clusters=[]
    for first in order:
        if first not in unused:continue
        cluster=[first];unused.remove(first)
        for candidate in order:
            if candidate in unused and bool((similarity[candidate,cluster]>=threshold).all()):
                cluster.append(candidate);unused.remove(candidate)
        clusters.append(cluster)
    readers=[];indices=torch.empty(len(flat),dtype=torch.long,device=a.device)
    for i,ids in enumerate(clusters):
        if len(ids)==1:reader=flat[ids[0]]
        else:
            weights=energy[ids]/energy[ids].sum()
            reader=torch.linalg.svd(flat[ids]*weights.sqrt()[:,None],full_matrices=False).Vh[0]
        readers.append(reader);indices[ids]=i
    readers=torch.stack(readers)
    projection=(flat*readers[indices]).sum(1).square()
    return readers,indices.reshape(m,r),s*projection.reshape(m,r),clusters


def execute(readers,indices,s,c,x):
    squares=(x@readers.T).square()
    groups=(squares[:,indices]*s[None]).sum(-1)
    return groups@c


class Objective:
    def __init__(self,target,readers,indices,s,total,penalty=.01):
        self.target,self.total,self.penalty=target,total,penalty
        self.indices=indices;self.device=readers.device
        norms=readers.norm(dim=-1,keepdim=True)
        readers=readers/norms;s=s*norms[:,0][indices].square()
        s=s/s.norm(dim=-1,keepdim=True)
        self.shapes=[readers.shape,s.shape];self.sizes=[readers.numel(),s.numel()]
        self.scales=[float(readers.norm()),float(s.norm())]
        self.initial=torch.cat([(v/scale).flatten().cpu() for v,scale in zip((readers,s),self.scales)]).numpy()

    def unpack(self,point):
        values=torch.as_tensor(point,device=self.device,dtype=torch.float64).split(self.sizes)
        return tuple(v.reshape(shape)*scale for v,shape,scale in zip(values,self.shapes,self.scales))

    def physical(self,point):
        readers,s=self.unpack(point)
        return readers/readers.norm(dim=-1,keepdim=True),s/s.norm(dim=-1,keepdim=True)

    def evaluate(self,point):
        readers,s=self.unpack(point);rn=readers.norm(dim=-1,keepdim=True);sn=s.norm(dim=-1,keepdim=True)
        rr,ss=readers/rn,s/sn
        loss,(slot_gradient,gs),writer,details=projected_evaluate(self.target,rr[self.indices],ss,self.total,self.penalty)
        gr=torch.zeros_like(rr)
        gr.index_add_(0,self.indices.flatten(),slot_gradient.flatten(0,1))
        gr=(gr-rr*(rr*gr).sum(-1,keepdim=True))/rn
        gs=(gs-ss*(ss*gs).sum(-1,keepdim=True))/sn
        self.last=details;self.writer=writer
        gradient=torch.cat([(v*scale).flatten().cpu() for v,scale in zip((gr,gs),self.scales)]).numpy()
        return float(loss),gradient
