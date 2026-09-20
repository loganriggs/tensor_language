"""CPU snapshot baseline; no native causal or independent extraction claim."""
from pathlib import Path
import json
import torch

def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2]
    source=root/'basis_aligned/bilinear_quotient/circuits/followups/residual_reader_transfer_v1_tensors.pt'
    groups=torch.load(source,map_location='cpu',weights_only=False)['groups']
    fit=[g for g in groups if g['panel']=='opposite']
    X=torch.cat([g['sources'].reshape(-1,1152) for g in fit]).T
    Y=torch.cat([g['reader'].reshape(-1,1152) for g in fit]).T
    H=Y.T@X
    U,s,Vh=torch.linalg.svd(H,full_matrices=False)
    Q,_,_=torch.linalg.svd(X,full_matrices=False)
    records=[]
    for rank in (8,16,32):
        V=(X@Vh[:rank].T)/s[:rank].sqrt()
        W=(Y@U[:,:rank])/s[:rank].sqrt()
        # Exact truncated cross-snapshot reconstruction and biorthogonality.
        replay=((Y.T@V)@(W.T@X)-(U[:,:rank]*s[:rank])@Vh[:rank]).abs().max().item()
        bio=(W.T@V-torch.eye(rank,dtype=X.dtype)).abs().max().item()
        for name,A,B in [('balanced',V,W),('pod_same_rank',Q[:,:rank],Q[:,:rank]),('pod_same_storage',Q[:,:2*rank],Q[:,:2*rank])]:
            errors=[]
            for g in groups:
                if g['panel']!='congruent':continue
                R,D=g['reader'],g['sources']
                exact=R@D.transpose(1,2)
                approx=(R@A)@(D@B).transpose(1,2)
                rel=((approx-exact).square().sum((0,2))/exact.square().sum((0,2)).clamp_min(1e-24)).sqrt()
                errors.append({'template':g['template'],'role':g['role'],'relative_errors':rel.tolist()})
            records.append({'rank':rank,'method':name,'stored_values':A.numel()+(B.numel() if name=='balanced' else 0),'biorthogonal_error':bio if name=='balanced' else None,'cross_snapshot_replay':replay if name=='balanced' else None,'held':errors})
    # Negative tripwire: energy POD misses a low-energy but highly observed axis.
    x=torch.diag(torch.tensor([100.,1.],dtype=torch.float64)); y=torch.diag(torch.tensor([.001,100.],dtype=torch.float64))
    u,ss,vh=torch.linalg.svd(y.T@x); v=x@vh[:1].T/ss[:1].sqrt();w=y@u[:,:1]/ss[:1].sqrt()
    planted=(v@w.T-torch.diag(torch.tensor([0.,1.],dtype=x.dtype))).abs().max().item()
    assert planted<1e-12 and max(r['biorthogonal_error'] or 0 for r in records)<1e-8
    out={'scope':'opened held-panel gradient contraction only; no native projection evaluation','fit_panel':'opposite','X_shape':list(X.shape),'Y_shape':list(Y.shape),'planted_observed_axis_error':planted,'records':records}
    path=Path(__file__).with_name('BALANCED_SOURCE_READERS_V1.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(out,indent=2)+'\n')
    for r in records:print(r['rank'],r['method'],max(max(h['relative_errors']) for h in r['held']))
if __name__=='__main__':main()
