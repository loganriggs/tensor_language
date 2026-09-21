"""Output-aware greedy projection in an implicit product dictionary Gram metric."""
import torch

def select(gram,cross,count):
    diagonal=gram.diag().clone();residual=cross.clone();chol=torch.zeros(len(gram),count,dtype=gram.dtype);chosen=[]
    for step in range(count):
        scores=residual.square().sum(0)/diagonal.clamp_min(1e-30)
        scores[diagonal<1e-12]=-float('inf')
        if chosen:scores[chosen]=-float('inf')
        idx=int(scores.argmax());assert torch.isfinite(scores[idx])
        root=diagonal[idx].sqrt();column=(gram[:,idx]-chol[:,:step]@chol[idx,:step])/root
        gain=residual[:,idx]/root;residual-=gain[:,None]*column[None,:];diagonal=(diagonal-column.square()).clamp_min(0);chol[:,step]=column;chosen.append(idx)
    return torch.tensor(chosen)

def refit(gram,teacher,indices):
    block=gram[indices][:,indices];cross=teacher@gram[:,indices];v,Q=torch.linalg.eigh(block);keep=v>v[-1]*1e-10;writer=(cross@Q[:,keep]/v[keep])@Q[:,keep].T
    return writer,float((writer@block-cross).norm()/cross.norm()),int(keep.sum())

def toy_check():
    gen=torch.Generator().manual_seed(261231);f=torch.randn(7,20,generator=gen,dtype=torch.float64);f=f/f.norm(dim=1)[:,None];c=torch.randn(3,7,generator=gen,dtype=torch.float64);K=f@f.T;target=c@f;ids=select(K,c@K,3);chosen=[]
    for step in range(3):
        errors={}
        for j in range(7):
            if j in chosen:continue
            s=torch.tensor(chosen+[j]);w=torch.linalg.lstsq(f[s].T,target.T).solution.T;errors[j]=float((w@f[s]-target).square().sum())
        best=min(errors,key=errors.get);assert int(ids[step])==best;chosen.append(best)
    w,normal,_=refit(K,c,ids);dense=torch.linalg.lstsq(f[ids].T,target.T).solution.T;error=float((w-dense).norm()/dense.norm());assert error<1e-12
    return dict(dense_writer_replay=error,normal_equation_error=normal,greedy_steps_match_exhaustive=True)
