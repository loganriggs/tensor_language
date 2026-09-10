"""Training-only moment spectrum; second-moment geometry, not fourth-moment fit."""
import json,time
from pathlib import Path
import torch
from prepare_million_token_panel_v1 import digest
P=Path(__file__).resolve().parent
def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    source=P/'MILLION_TOKEN_PANEL_V1_INPUTS.pt'
    d=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    s=d['training_second_moment'].double();mu=d['training_mean'].double();cov=s-mu[:,None]*mu[None,:]
    assert d['training_moment_tokens']==819200
    result={}
    for name,m in [('second_moment',s),('centered_covariance',cov)]:
        e=torch.linalg.eigvalsh((m+m.T)/2).flip(0);trace=float(e.sum())
        result[name]=dict(trace=trace,minimum_eigenvalue=float(e[-1]),maximum_eigenvalue=float(e[0]),condition=float(e[0]/e[-1]) if e[-1]>0 else None,top_energy_fraction={str(k):float(e[:k].sum()/e.sum()) for k in [1,8,32,64,128,256,512,1024]},relative_asymmetry=float((m-m.T).norm()/m.norm()))
        assert float(e[-1])>=-1e-6*float(e[0]), 'Covariance not numerically PSD'
    out=dict(schema='million.token.training.metric.audit.v1',input_sha256=digest(source),training_moment_tokens=819200,mean_squared_norm=float(mu.square().sum()),mean_energy_fraction=float(mu.square().sum()/s.trace()),metrics=result,wall_seconds=time.perf_counter()-tic,scope='Only training mean and second moment inspected. Top covariance directions are geometry, not circuits or full quadratic loss. FP32 matrix products accumulatedFP64; tiny negative eigenvalues within relative1e-6 accepted as roundoff. No validation/test or fitting.')
    with (P/'MILLION_TOKEN_TRAINING_METRIC_V1_AUDIT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
