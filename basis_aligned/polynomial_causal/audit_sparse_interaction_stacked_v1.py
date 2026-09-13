"""Post-result call-overhead countercheck: one stacked CSR contraction."""
import json,time
from pathlib import Path
import numpy as np
import torch
from sparse_interaction_executor_v1 import Executor
from audit_sparse_interaction_executor_v1 import timed,relative
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    p=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    execute=Executor(p);o,i,a=p['shape'];n=o*i*a
    mask=torch.from_numpy(np.unpackbits(p['mask'].numpy(),bitorder='little',count=n).copy()).bool().reshape(o*i,a)
    rows,cols=mask.nonzero(as_tuple=True)
    ptr=torch.cat([torch.zeros(1,dtype=torch.int64),mask.sum(1).cumsum(0)]).int()
    csr=torch.sparse_csr_tensor(ptr,cols.int(),p['values'],size=(o*i,a),check_invariants=True)
    def stacked(z,h):
        values=torch.sparse.mm(csr,(h@p['head']).T).T.reshape(z.shape[0],o,i)
        return (values*z[:,None,:]).sum(-1)@p['output'].T
    core=torch.zeros(o*i,a);core[mask]=p['values'];core=core.reshape(o,i,a)
    def dense_rotated(z,h):
        return torch.einsum('oia,ni,na->no',core,z,h@p['head'])@p['output'].T
    gen=torch.Generator().manual_seed(61325);z=torch.randn(120,i,generator=gen);h=torch.randn(120,a,generator=gen)
    checks=dict(stacked_vs_original=relative(stacked(z,h),execute(z,h)),stacked_vs_dense=relative(stacked(z,h),dense_rotated(z,h)))
    assert max(checks.values())<=1e-5
    cells=[]
    for batch in [1,120]:
        timings={name:timed(fn,z[:batch],h[:batch]) for name,fn in [('original',execute),('stacked',stacked),('dense_rotated',dense_rotated)]}
        cells.append(dict(batch=batch,methods=timings))
    out=dict(checks=checks,cells=cells,stacked_intermediate_bytes_batch120=120*o*i*4,
        scope='Same packed program; post-result CPU overhead discriminator. Dense rotated baseline includes identical adapters. No new fitting.')
    (P/'SPARSE_INTERACTION_STACKED_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(checks)
    for c in cells:print(c['batch'],{k:v['median_seconds'] for k,v in c['methods'].items()})

if __name__=='__main__':main()
