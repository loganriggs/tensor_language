"""Known connected mixed-product graph; distinguish stationarity and recovery."""
import json
from pathlib import Path
import torch
from amortized_sparse_frame_v1 import fit
from streamed_sparse_frame_v2 import select


def run():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    g=torch.Generator().manual_seed(120503)
    qtrue=torch.linalg.qr(torch.randn(8,8,generator=g)).Q
    edges=torch.tensor([(0,1),(0,2),(0,3),(1,2),(1,4),(2,5),(3,6),(4,7),(5,7),(6,7),(0,0),(7,7)]).T
    weights=torch.randn(10,12,generator=g)
    i,j=edges;l=qtrue[:,i].T;r=qtrue[:,j].T
    w=weights*torch.where(i==j,1.,2.**.5)
    total=float(weights.square().sum());_,exact,_=select(qtrue,l,r,w,total,12)
    reports=[]
    for seed,near in [(120509,True),(120521,False),(120523,False),(120527,False),(120529,False)]:
        raw=torch.randn(8,8,generator=torch.Generator().manual_seed(seed))
        q=torch.linalg.qr(qtrue+.02*raw if near else raw).Q
        out,e,receipt=fit(q,l,r,w,total,12,max_steps=2000,seconds=30)
        row=dict(seed=seed,near=near,**receipt)
        row['recovered']=receipt['history'][-1]['capture']>=1-1e-8
        row['orthogonality']=float((out.T@out-torch.eye(8)).abs().max())
        reports.append(row)
    return dict(planted_capture=float(exact),reports=reports,recoveries=sum(v['recovered'] for v in reports),
                converged=sum(v['converged'] for v in reports),
                instrument_pass=abs(float(exact)-1)<1e-12 and all(v['orthogonality']<1e-12 and v['maximum_epoch_decrease']<1e-12 and v['maximum_fixed_decrease']<1e-12 for v in reports),
                scope='Five local starts on one known mixed-edge graph; not native or generic recovery theorem.')


if __name__=='__main__':
    result=run();out=Path(__file__).with_name('AMORTIZED_MIXED_GRAPH_V1_CONTROL.json');assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))
    for r in result['reports']:print(json.dumps({k:v for k,v in r.items() if k!='history'}|{'initial':r['history'][0],'final':r['history'][-1]}))
