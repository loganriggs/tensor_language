"""Fixed-group shared writers plus sparse residual, exact matched-byte screen."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from interaction_shared_write_subspaces_v1 import update,reconstruction
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);start=time.perf_counter()
    t,ids=build();x=t.permute(1,2,0).reshape(-1,12).contiguous()
    prior=torch.load(P/'INTERACTION_SHARED_WRITE_POLISH_V1_PROGRAM.pt',weights_only=True)
    assert ids==prior['token_ids']
    labels=prior['groups'].long();total=float(x.square().sum());budget=4896936;bitmap=x.numel()//8
    torch.manual_seed(61521);products=torch.randn(3,x.shape[0],dtype=x.dtype)
    rows=[];best=None
    for rank in range(8):
        if rank:
            q=update(x,labels,prior['output_bases'].double()[:,:,:rank])
            codes=torch.einsum('ndr,nd->nr',q[labels],x)
            common=reconstruction(labels,codes,q)
            shared_bytes=4*(x.shape[0]*rank+32*12*rank)+x.shape[0]
        else:
            q=x.new_zeros(32,12,0);codes=x.new_zeros(x.shape[0],0);common=torch.zeros_like(x);shared_bytes=0
        residual=x-common;count=min(x.numel(),(budget-shared_bytes-bitmap)//4)
        values,order=residual.flatten().square().sort(descending=True)
        sparse=torch.zeros_like(residual.flatten());sparse[order[:count]]=residual.flatten()[order[:count]]
        fitted=common+sparse.reshape_as(x);error=float((x-fitted).square().sum()/total)
        expected=float(values[count:].sum()/total);identity=abs(error-expected);assert identity<1e-12
        features=x.new_zeros(3,32,rank)
        for component in range(rank):features[:,:,component].scatter_add_(1,labels[None].expand(3,-1),products*codes[:,component][None])
        execution=torch.einsum('bkr,kdr->bd',features,q)+products@sparse.reshape_as(x)
        direct=products@fitted;replay=float((execution-direct).norm()/direct.norm());assert replay<1e-10
        row=dict(rank=rank,shared_bytes=shared_bytes,private_entries=count,bitmap_bytes=bitmap,
            total_bytes=shared_bytes+bitmap+4*count,relative_error=error**.5,
            residual_energy_identity_error=identity,readout_replay_error=replay)
        rows.append(row)
        if best is None or error<best['loss']:best=dict(loss=error,row=row)
    result=dict(pred_a=True,pred_b=best['loss']<=.01,pred_c=best['loss']<=.009,rows=rows,best=best['row'],
        seconds=time.perf_counter()-start,
        scope='Fixed ordinary K32 assignments; exact conditional output PCA then globally largest residual entries at matched nominal byte budget. No joint optimization or serialized artifact; bitmap metadata/runtime costs not discounted. Compared with existing ~10% output/head sparse baseline; no native fidelity or behavioral reuse claim.')
    (P/'INTERACTION_SHARED_PRIVATE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
