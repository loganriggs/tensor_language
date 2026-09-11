"""Exact shared-node interventions and pair corrections in an emitted DAG.

Intervention zeros computed z_i, keeping private readers and background fixed.
It is not deletion of an input direction everywhere in the original model.
"""
import torch
from ll1_joint_parent_graph_v2 import execute


def disable(graph,x,parents):
    readers=graph['readers'].clone();readers[list(parents)]=0
    return execute(dict(graph,readers=readers),x)


def concatenate(*banks):
    return torch.cat([b[0] for b in banks]),torch.cat([b[1] for b in banks]),torch.cat([b[2] for b in banks],1)


def subset(bank,indices):
    return bank[0][indices],bank[1][indices],bank[2][:,indices]


def value(bank,x):
    return ((x@bank[0].T)*(x@bank[1].T))@bank[2].T


def banks(graph):
    readers=graph['readers'];pairs=graph['pairs'];output=graph['groups'][0]['writer'].numel()
    pair_w=readers.new_zeros(output,len(pairs));a=[];b=[];w=[];ids=[]
    for g in graph['groups']:
        pair_w.index_add_(1,g['pair_ids'],g['writer'][:,None]*g['shared_coeff'][None])
        a.append(readers[g['parent_ids']]);b.append(g['cross']@g['private'])
        w.append(g['writer'][:,None].expand(-1,len(g['parent_ids'])))
        ids.append(g['parent_ids'])
    mixed=(torch.cat(a),torch.cat(b),torch.cat(w,1));mixed_ids=torch.cat(ids)
    pair=(readers[pairs[:,0]],readers[pairs[:,1]],pair_w)
    nodes=[]
    for i in range(len(readers)):
        nodes.append(concatenate(subset(mixed,mixed_ids==i),subset(pair,(pairs==i).any(1))))
    return mixed,pair,nodes


def group_parent(graph,index,parent):
    g=graph['groups'][index];position=(g['parent_ids']==parent).nonzero()[0,0]
    mixed=(graph['readers'][parent:parent+1],(g['cross'][position]@g['private'])[None],g['writer'][:,None])
    pairs=graph['pairs'][g['pair_ids']];selected=(pairs==parent).any(1)
    pairs=pairs[selected];coeff=g['shared_coeff'][selected]
    pair=(graph['readers'][pairs[:,0]],graph['readers'][pairs[:,1]],g['writer'][:,None]*coeff[None])
    return concatenate(mixed,pair)
