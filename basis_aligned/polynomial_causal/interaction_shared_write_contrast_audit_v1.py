"""Exact paired output sum/difference error audit, without refitting."""
import json
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);t,ids=build();program=torch.load(P/'INTERACTION_SHARED_WRITE_POLISH_V1_PROGRAM.pt',weights_only=True)
    assert ids==program['token_ids']
    f=torch.einsum('ndr,nr->nd',program['output_bases'].double()[program['groups'].long()],program['codes'].double()).reshape(1152,128,12).permute(2,0,1)
    bases=[]
    for axis in (0,2):
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1);bases.append(torch.linalg.eigh(flat@flat.T)[1])
    core=torch.einsum('op,oia,ab->pib',bases[0],t,bases[1]);values,order=core.flatten().square().sort(descending=True)
    k=int(torch.searchsorted(values.cumsum(0),.99*t.square().sum()))+1
    flat=torch.zeros_like(core.flatten());flat[order[:k]]=core.flatten()[order[:k]]
    sparse=torch.einsum('op,pib,ab->oia',bases[0],flat.reshape_as(core),bases[1])
    def parts(x):
        pairs=x.reshape(6,2,1152,128)
        return ((pairs[:,0]+pairs[:,1])/2**.5,(pairs[:,0]-pairs[:,1])/2**.5)
    reference=parts(t);total=float(t.square().sum());rows=[]
    for name,fit in [('shared_output_subspaces',f),('sparse_entry_10percent',sparse)]:
        residual=parts(fit-t);partition=sum(float(x.square().sum()) for x in residual)
        assert abs(partition-float((fit-t).square().sum()))/total<1e-10
        rows.append(dict(method=name,total_relative_error=float((fit-t).norm()/t.norm()),
                         sum_relative_error=float(residual[0].norm()/reference[0].norm()),
                         difference_relative_error=float(residual[1].norm()/reference[1].norm()),
                         difference_fraction_of_squared_residual=float(residual[1].square().sum())/partition))
    result=dict(target_sum_energy_fraction=float(reference[0].square().sum())/total,
                target_difference_energy_fraction=float(reference[1].square().sum())/total,methods=rows,
                scope='Post-fit coefficient audit using existing six token pairs. Exact orthogonal sum/difference partition; no textfit or change to original total-Frobenius objective. Does not by itself explain conditional behavioral error.')
    (P/'INTERACTION_SHARED_WRITE_CONTRAST_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
