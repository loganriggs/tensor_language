"""Optimal variable-product allocation for orthogonal outputs and fixed slice SVDs."""
import itertools,json
from pathlib import Path
import torch

def allocate(energies,budget):
    assert energies.ndim==2 and 0<=budget<=energies.numel()
    assert bool((energies>=0).all())
    assert bool((energies[:,:-1]>=energies[:,1:]).all())
    order=energies.flatten().argsort(descending=True,stable=True)
    mask=torch.zeros(energies.numel(),dtype=torch.bool);mask[order[:budget]]=True
    mask=mask.reshape_as(energies)
    assert not bool((mask[:,1:] & ~mask[:,:-1]).any())
    return mask.sum(1),float(energies[mask].sum())

def toy_check():
    energy=torch.tensor([[9.,1.],[8.,7.],[2.,1.],[1.,.5]],dtype=torch.float64)
    ranks,kept=allocate(energy,4)
    brute=max(sum(float(energy[j,:r].sum()) for j,r in enumerate(rr)) for rr in itertools.product(range(3),repeat=4) if sum(rr)==4)
    assert kept==brute==26
    assert ranks.tolist()==[1,2,1,0]
    ties=torch.ones(4,3,dtype=torch.float64)
    for budget in range(13):allocate(ties,budget)
    return dict(ranks=ranks.tolist(),adaptive_retained_energy=kept,uniform_four_outputs_rank1=float(energy[:,0].sum()),first_two_outputs_rank2=float(energy[:2].sum()),exhaustive_optimum=brute,tie_prefix_checks=13)

if __name__=='__main__':
    p=Path(__file__).resolve().parent
    result=dict(toy=toy_check(),scope='Exact only for orthogonal output modes, fixed per-output matrix SVDs, same separable metric, and fixed number of variable products. Does not optimize output basis, factor reuse, coefficient storage, native intervention error, or semantic identity. Active-output count changes writer cost.')
    out=p/'ADAPTIVE_OUTPUT_RANK_ORACLE_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
