"""One signed output ray per coordinate input product; exact alternating steps."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def assign(x,dictionary):
    score=x@dictionary.T
    assignment=score.square().argmax(dim=1)
    coefficients=score.gather(1,assignment[:,None]).squeeze(1)
    return assignment,coefficients


def update(x,assignment,dictionary):
    k=dictionary.shape[0];width=x.shape[1]
    gram=x.new_zeros(k,width,width)
    for i in range(width):
        for j in range(i,width):
            value=torch.bincount(assignment,weights=x[:,i]*x[:,j],minlength=k)
            gram[:,i,j]=value;gram[:,j,i]=value
    _,vectors=torch.linalg.eigh(gram)
    new=vectors[:,:,-1]
    occupied=torch.bincount(assignment,minlength=k)>0
    return torch.where(occupied[:,None],new,dictionary)


def fit(x,k,seed,steps):
    gen=torch.Generator().manual_seed(seed)
    dictionary=x[torch.randperm(x.shape[0],generator=gen)[:k]].clone()
    dictionary/=dictionary.norm(dim=1,keepdim=True)
    total=float(x.square().sum());history=[];previous=None
    for step in range(steps+1):
        labels,coefficients=assign(x,dictionary)
        loss=float((x-coefficients[:,None]*dictionary[labels]).square().sum())/total
        changes=None if previous is None else int((labels!=previous).sum())
        history.append(dict(step=step,loss=loss,changed_assignments=changes))
        if len(history)>1:assert loss<=history[-2]['loss']+1e-12
        if changes==0:break
        previous=labels.clone()
        if step<steps:dictionary=update(x,labels,dictionary)
    return dictionary,labels,coefficients,history


def main():
    torch.set_num_threads(2);torch.manual_seed(61431);start=time.perf_counter()
    # A planted two-ray control with deterministic axis-aligned geometry.
    planted=torch.zeros(80,3,dtype=torch.float64);planted[:40,0]=torch.linspace(-2,2,40);planted[40:,1]=torch.linspace(-2,2,40)
    initial=torch.tensor([[1.,.2,0],[.2,1.,0]],dtype=torch.float64);initial/=initial.norm(dim=1,keepdim=True)
    labels,_=assign(planted,initial);dictionary=update(planted,labels,initial);labels,coeff=assign(planted,dictionary)
    planted_error=float((planted-coeff[:,None]*dictionary[labels]).norm()/planted.norm());assert planted_error<1e-12
    t,ids=build();x=t.permute(1,2,0).reshape(-1,12).contiguous();results=[]
    for seed in (61431,61432,61433):
        dictionary,labels,coeff,history=fit(x,32,seed,20)
        # Explicit shared bilinear-feature program versus reconstructed tensor.
        z=torch.randn(3,1152,dtype=x.dtype);h=torch.randn(3,128,dtype=x.dtype)
        products=(z[:,:,None]*h[:,None,:]).flatten(1)
        features=x.new_zeros(3,32)
        features.scatter_add_(1,labels[None].expand(3,-1),products*coeff[None])
        output=features@dictionary
        reconstructed=coeff[:,None]*dictionary[labels]
        reference=products@reconstructed
        replay=float((output-reference).norm()/reference.norm());assert replay<1e-10
        results.append(dict(seed=seed,relative_error=history[-1]['loss']**.5,history=history,
                            assignment_converged=history[-1]['changed_assignments']==0,
                            execution_error=replay,occupied_rays=int(labels.unique().numel())))
    result=dict(planted_error=planted_error,k=32,results=results,token_ids=ids,
                scalar_input_products=x.shape[0],dictionary_scalars=32*12,
                packed_fp32_bytes=4*x.shape[0]+x.shape[0]+4*32*12,
                dense_tensor_bytes=4*t.numel(),seconds=time.perf_counter()-start,
                scope='Actual-weight20sweep feasibility screen, three starts; unfinished assignments are not convergence. One signed coefficient and uint8 ray index per input product. Each shared feature is a sparse bilinear form, not a simple rank-one product. No native fidelity/runtime/adoption claim.')
    (P/'INTERACTION_SHARED_WRITE_RAYS_V1_SCREEN.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    for r in results:print(r['seed'],r['relative_error'],r['assignment_converged'],r['history'][-1]['changed_assignments'])


if __name__=='__main__':main()
