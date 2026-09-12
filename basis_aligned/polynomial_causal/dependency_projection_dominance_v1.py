"""Dense coefficient control: linear input dependency versus oblique output map."""
from pathlib import Path
import itertools,json
import torch
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73260);dtype=torch.float64;d=4;k=2
    a=torch.randn(3,d,d,dtype=dtype);a=(a+a.transpose(-1,-2))/2
    raw=torch.einsum('ab,jcd->jabcd',a[0],a[1:]);target=sum(raw.permute(0,*[i+1 for i in perm]) for perm in itertools.permutations(range(4)))/24
    writer=torch.randn(5,2,dtype=dtype);target=torch.einsum('oj,jabcd->oabcd',writer,target)
    basis=torch.linalg.qr(torch.randn(d,k,dtype=dtype),mode='reduced')[0];q=basis@basis.T
    def pull(t,m):return torch.einsum('oabcd,ai,bj,ck,dl->oijkl',t,m,m,m,m)
    projected=pull(target,q);reports=[]
    for seed in range(8):
        torch.manual_seed(73261+seed);write=torch.randn(d,k,dtype=dtype);m=write@basis.T;approx=pull(target,m)
        error=(target-approx).square().sum();floor=(target-projected).square().sum();excess=(projected-approx).square().sum()
        identity=float(abs(error-floor-excess)/error)
        reports.append(dict(seed=seed,pythagorean_relative_error=identity,projected_error=float(floor),arbitrary_map_error=float(error),excess=float(excess),dependency_replay_error=float((pull(approx,q)-approx).norm()/approx.norm())))
    assert max(max(z['pythagorean_relative_error'],z['dependency_replay_error']) for z in reports)<1e-10
    assert all(z['arbitrary_map_error']>=z['projected_error']-1e-10 for z in reports)
    result=dict(reports=reports,scope='Formal independent input slots, coefficient Frobenius norm with output writers already folded. All approximations depend on the fixed rowspace of the map. Not a theorem about repeated native polynomial inputs, arbitrary nonlinear features, data-weighted metrics, or globally optimal subspace choice.')
    out=P/'DEPENDENCY_PROJECTION_DOMINANCE_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
