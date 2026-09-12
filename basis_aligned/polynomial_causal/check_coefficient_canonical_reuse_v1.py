"""Shared scalar computations survive different head coordinate maps."""
from pathlib import Path
import json
import torch
from coefficient_canonical_reuse_v1 import fit_pair,cosines
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73290);dt=torch.float64
    u=torch.linalg.qr(torch.randn(12,12,dtype=dt))[0];v=torch.linalg.qr(torch.randn(12,12,dtype=dt))[0]
    def sample(n):
        shared=torch.randn(n,3,dtype=dt)
        return torch.cat([shared,torch.randn(n,9,dtype=dt)],1)@u.T,torch.cat([shared,torch.randn(n,9,dtype=dt)],1)@v.T
    a,b=sample(512);r,s,sv=fit_pair(a.T@a/512,b.T@b/512,a.T@b/512,3)
    x,y=sample(512);c=cosines(x@r,y@s);error=float(((x@r-y@s).norm()/(x@r).norm()))
    assert float(c.min())>1-1e-10 and error<1e-10
    out=P/'COEFFICIENT_CANONICAL_REUSE_V1_CONTROL.json';assert not out.exists();result=dict(train_canonical_correlations=sv.tolist(),heldout_correlations=c.tolist(),heldout_relative_error=error);out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
