"""Same-output readouts stable across two fits, with separate native accuracy.

Generalized eigenvectors of disagreement covariance versus native covariance.
A numerical replay/full output rank; B >=16/32 modes have disagreement<=.05
and both native errors<=.1; C qualified span contains>=.01 native energy.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from cancellation_group_v1_audit import load
from joint_quadratic_fit_v1 import product_cross
from chunked_bilinear_coefficient_v1 import dense


def covariance(first,second):
    a,b,w=first;c,d,v=second
    return (w@product_cross(a,b,c,d))@v.T


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'OUTPUT_FUNCTION_AGREEMENT_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    torch.manual_seed(681)
    f=(torch.randn(5,4),torch.randn(5,4),torch.randn(3,5))
    g=(torch.randn(6,4),torch.randn(6,4),torch.randn(3,6))
    exact=dense(*f).flatten(1)@dense(*g).flatten(1).T
    toy=float((covariance(f,g)-exact).norm()/exact.norm())
    source=json.loads((P/'INTERMEDIATE_FUNCTION_STABILITY_V1_AUDIT.json').read_text())
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;first,second=[load(s,wh) for s in source['sources']]
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    native=(l,r,wh@d);total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    cov={}
    for name,a,b in [('native',native,native),('first',first,first),('second',second,second),
                     ('cross',first,second),('native_first',native,first),('native_second',native,second)]:
        cov[name]=covariance(a,b)/total;print(json.dumps(dict(covariance=name,seconds=time.perf_counter()-start)),flush=True)
    n,a,b,c,u,v=[cov[k] for k in ('native','first','second','cross','native_first','native_second')]
    n=(n+n.T)/2;diff=a+b-c-c.T;diff=(diff+diff.T)/2
    errors=[n+a-u-u.T,n+b-v-v.T]
    eigen,basis=torch.linalg.eigh(n);keep=eigen>eigen.max()*1e-12
    transform=basis[:,keep]/eigen[keep].sqrt()[None,:]
    reduced=transform.T@diff@transform;reduced=(reduced+reduced.T)/2
    values,modes=torch.linalg.eigh(reduced);directions=transform@modes
    eigen_replay=float((diff@directions-(n@directions)*values).norm()/(diff@directions).norm().clamp_min(1e-30))
    expected=[1.,*source['fitted_energies'],source['function_cross'],*source['native_crosses']]
    replay=max(abs(float(cov[k].trace())-e) for k,e in zip(cov,expected))
    trace_vector=native[2]@(l*r).sum(1)
    rows=[]
    for i in range(32):
        q=directions[:,i];den=q@n@q
        errs=[float(q@er@q/den) for er in errors]
        rows.append(dict(mode=i,disagreement=float(q@diff@q/den),native_errors=errs,
            native_energy_for_unit_output=float(den/q.square().sum()),
            native_radial_fraction=float((q@trace_vector).square()/(1152*total*den)),
            qualifies=float(values[i])<=.05 and max(errs)<=.1))
    selected=[i for i,row in enumerate(rows) if row['qualifies']]
    if selected:
        orthogonal=torch.linalg.qr(directions[:,selected],mode='reduced').Q
        span_energy=float(torch.trace(orthogonal.T@n@orthogonal))
    else:span_energy=0.
    pred=dict(pred_a_instrument=toy<=1e-10 and replay<=1e-8 and eigen_replay<=1e-7 and int(keep.sum())==1152 and float(values.min())>=-1e-8,
              pred_b_stable_accurate_modes=len(selected)>=16,pred_c_nontrivial_span=span_energy>=.01)
    cache=Path('/dev/shm/bilin18_output_function_agreement_v1.pt');assert not cache.exists()
    torch.save(dict(covariances=cov,directions=directions[:,:32],generalized_eigenvalues=values,
        native_eigenvalues=eigen,sources=source['sources'],native_total=total),cache)
    result=dict(predictions=pred,rows=rows,qualifying_modes=selected,qualified_span_native_energy=span_energy,
        toy_replay=toy,covariance_trace_replay=replay,generalized_eigen_replay=eigen_replay,
        native_retained_rank=int(keep.sum()),native_condition=float(eigen.max()/eigen.min()),
        disagreement_spectrum_quantiles={str(q):float(torch.quantile(values,q)) for q in (0.,.01,.1,.5,1.)},
        sources=source['sources'],cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
        seconds=time.perf_counter()-start,
        scope='Weight-selected common output readouts of two incomplete fits. No independent validation, simple input arithmetic or circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
