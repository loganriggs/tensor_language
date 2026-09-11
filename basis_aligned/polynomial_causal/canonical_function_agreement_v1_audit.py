"""Principal angles of fitted quadratic-function row spaces, separate readouts.

A full-rank whitening/orthogonality/cross replay <=1e-8; B >=16/32cos>=.95;
C >=8of those also have <=.1native relative error for both readouts.
Reuses cached exact covariance matrices. No new fitting or behavioral claim.
"""
import hashlib,json,time
from pathlib import Path
import torch


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    p=Path(__file__).resolve().parent;out=p/'CANONICAL_FUNCTION_AGREEMENT_V1_AUDIT.json';assert not out.exists()
    source=json.loads((p/'OUTPUT_FUNCTION_AGREEMENT_V1_AUDIT.json').read_text());cache=source['cache']
    assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    saved=torch.load(cache['path'],weights_only=True,map_location='cpu');cov=saved['covariances'];start=time.perf_counter()
    n,a,b,c,u,v=[cov[k] for k in ('native','first','second','cross','native_first','native_second')]
    def whiten(matrix):
        e,q=torch.linalg.eigh((matrix+matrix.T)/2);keep=e>e.max()*1e-12
        return q[:,keep]/e[keep].sqrt()[None,:],int(keep.sum()),float(e.max()/e.min())
    x,ra,ca=whiten(a);y,rb,cb=whiten(b)
    left,values,right=torch.linalg.svd(x.T@c@y,full_matrices=False)
    q0=x@left[:,:32];q1=y@right[:32].T
    identity=torch.eye(32);replay=max(float((q0.T@a@q0-identity).abs().max()),
        float((q1.T@b@q1-identity).abs().max()),float((q0.T@c@q1-torch.diag(values[:32])).abs().max()))
    e0=n+a-u-u.T;e1=n+b-v-v.T;rows=[]
    for i in range(32):
        f=q0[:,i];g=q1[:,i];nf=f@n@f;ng=g@n@g
        errors=[float(f@e0@f/nf),float(g@e1@g/ng)]
        rows.append(dict(mode=i,function_cosine=float(values[i]),native_relative_errors=errors,
            native_function_cosine=float((f@n@g)/(nf*ng).sqrt()),
            ordinary_output_readout_cosine=float((f@g)/(f.norm()*g.norm())),
            qualifies=float(values[i])>=.95 and max(errors)<=.1))
    stable=sum(r['function_cosine']>=.95 for r in rows);qualified=sum(r['qualifies'] for r in rows)
    pred=dict(pred_a_instrument=replay<=1e-8 and ra==rb==1152 and float(values.max())<=1+1e-8,
        pred_b_shared_functions=stable>=16,pred_c_native_accuracy=qualified>=8)
    artifact=Path('/dev/shm/bilin18_canonical_function_agreement_v1.pt');assert not artifact.exists()
    torch.save(dict(first_readouts=q0,second_readouts=q1,correlations=values,sources=saved['sources'],parent_cache=cache),artifact)
    result=dict(predictions=pred,rows=rows,stable_count=stable,qualified_count=qualified,
        full_correlations_above_point95=int((values>=.95).sum()),
        correlation_quantiles={str(q):float(torch.quantile(values,q)) for q in (0.,.5,.9,.99,1.)},
        covariance_replay=replay,ranks=[ra,rb],conditions=[ca,cb],seconds=time.perf_counter()-start,
        source=cache,cache=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
        scope='Different output readouts, shared polynomial-function test on incomplete fits; no native simplicity or semantic circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
