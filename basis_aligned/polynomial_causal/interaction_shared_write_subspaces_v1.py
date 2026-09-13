"""Shared low-dimensional output subspaces per coordinate-product group."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def assign(x,bases):
    k,width,r=bases.shape
    coordinates=(x@bases.permute(1,0,2).reshape(width,k*r)).reshape(x.shape[0],k,r)
    labels=coordinates.square().sum(-1).argmax(-1)
    codes=coordinates[torch.arange(x.shape[0]),labels]
    return labels,codes


def update(x,labels,bases):
    k,width,r=bases.shape;gram=x.new_zeros(k,width,width)
    for i in range(width):
        for j in range(i,width):
            value=torch.bincount(labels,weights=x[:,i]*x[:,j],minlength=k)
            gram[:,i,j]=value;gram[:,j,i]=value
    _,q=torch.linalg.eigh(gram);occupied=torch.bincount(labels,minlength=k)>0
    return torch.where(occupied[:,None,None],q[:,:,-r:],bases)


def reconstruction(labels,codes,bases):
    return torch.einsum('ndr,nr->nd',bases[labels],codes)


def main():
    torch.set_num_threads(2);torch.manual_seed(61451);start=time.perf_counter()
    toy=torch.zeros(200,4,dtype=torch.float64);toy[:100,:2]=torch.randn(100,2);toy[100:,2:]=torch.randn(100,2)
    q=torch.stack([torch.eye(4)[:,:2],torch.eye(4)[:,2:]]).double()
    labels,codes=assign(toy,q);q=update(toy,labels,q);labels,codes=assign(toy,q)
    planted=float((toy-reconstruction(labels,codes,q)).norm()/toy.norm());assert planted<1e-10
    t,ids=build();x=t.permute(1,2,0).reshape(-1,12).contiguous();total=float(x.square().sum());rows=[]
    for rank in (2,4,8):
        q=torch.linalg.qr(torch.randn(32,12,rank,dtype=x.dtype),mode='reduced')[0]
        labels,codes=assign(x,q);initial=float((x-reconstruction(labels,codes,q)).square().sum())/total
        q=update(x,labels,q);labels,codes=assign(x,q);fitted=reconstruction(labels,codes,q)
        loss=float((x-fitted).square().sum())/total;assert loss<=initial+1e-10
        z=torch.randn(3,1152,dtype=x.dtype);h=torch.randn(3,128,dtype=x.dtype)
        products=(z[:,:,None]*h[:,None,:]).flatten(1);features=x.new_zeros(3,32,rank)
        for component in range(rank):features[:,:,component].scatter_add_(1,labels[None].expand(3,-1),products*codes[:,component][None])
        output=torch.einsum('bkr,kdr->bd',features,q);reference=products@fitted
        replay=float((output-reference).norm()/reference.norm());assert replay<1e-10
        rows.append(dict(rank=rank,initial_error=initial**.5,one_update_error=loss**.5,execution_error=replay,
                         bytes=4*(x.shape[0]*rank+32*12*rank)+x.shape[0]))
    result=dict(planted_error=planted,rows=rows,seconds=time.perf_counter()-start,
                scope='Operator feasibility and one actual-weight update only, K32,r2/4/8. Not converged fitting or native behavior. Per input product r signed coefficients plus uint8group, shared12xr orthonormal writers; each group feature is a sparse bilinear form. Full input rank allowed.')
    (P/'INTERACTION_SHARED_WRITE_SUBSPACES_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
