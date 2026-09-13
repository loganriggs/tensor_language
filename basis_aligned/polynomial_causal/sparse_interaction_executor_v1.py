"""Bitmap-packed mixed bilinear operator with an explicitly priced CSR runtime."""
import numpy as np
import torch


def compile_program(tensor, tolerance=.1):
    bases=[]
    for axis in [0,2]:
        f=tensor.movedim(axis,0).reshape(tensor.shape[axis],-1)
        bases.append(torch.linalg.eigh(f@f.T)[1])
    output,head=bases
    core=torch.einsum('op,oia,ab->pib',output,tensor,head)
    energy,order=core.flatten().square().sort(descending=True)
    k=int(torch.searchsorted(energy.cumsum(0),(1-tolerance**2)*energy.sum()))+1
    mask=torch.zeros(core.numel(),dtype=torch.bool);mask[order[:k]]=True
    program=dict(shape=list(core.shape),mask=torch.from_numpy(np.packbits(mask.numpy(),bitorder='little')),
                 values=core.flatten()[mask].float(),output=output.float(),head=head.float())
    fitted=torch.zeros_like(core.flatten());fitted[mask]=core.flatten()[mask]
    dense_fit=torch.einsum('op,pib,ab->oia',output,fitted.reshape_as(core),head)
    return program,dense_fit


class Executor:
    def __init__(self,program):
        shape=program['shape'];n=int(np.prod(shape))
        mask=torch.from_numpy(np.unpackbits(program['mask'].numpy(),bitorder='little',count=n).copy()).bool().reshape(shape)
        self.output=program['output'];self.head=program['head'];self.blocks=[];offset=0
        for block in mask:
            rows,cols=block.nonzero(as_tuple=True);counts=block.sum(1)
            rowptr=torch.cat([torch.zeros(1,dtype=torch.int64),counts.cumsum(0)]).int()
            size=len(cols);values=program['values'][offset:offset+size];offset+=size
            self.blocks.append(torch.sparse_csr_tensor(rowptr,cols.int(),values,size=block.shape))
        assert offset==program['values'].numel()
    def __call__(self,z,a):
        head=a@self.head
        terms=torch.stack([(torch.sparse.mm(block,head.T).T*z).sum(-1) for block in self.blocks],-1)
        return terms@self.output.T
    def resident_bytes(self):
        arrays=[self.output,self.head]
        for b in self.blocks:arrays.extend([b.crow_indices(),b.col_indices(),b.values()])
        return sum(t.numel()*t.element_size() for t in arrays)
