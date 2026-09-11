"""Inspect native scalar polynomial selected by cross-fit canonical agreement.

A exact form/eigen/CP/executor replay <=1e-8; B one-product capture>=.8 or
four-product capture>=.95; C native readout functions cosine>=.99.
Fixed first canonical mode; signed spectral pairing reuses established formula.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    previous=json.loads((P/'CANONICAL_FUNCTION_AGREEMENT_V1_AUDIT.json').read_text());source=previous['cache']
    assert previous['rows'][0]['qualifies']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    canonical=torch.load(source['path'],weights_only=True,map_location='cpu')
    cov=torch.load(canonical['parent_cache']['path'],weights_only=True,map_location='cpu')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;writer=wh@d;total=cov['native_total']
    q=canonical['first_readouts'][:,0];q=q/q.norm()
    other=canonical['second_readouts'][:,0];other=other/other.norm()
    coefficients=q@writer;coeff2=other@writer
    form=(l.T*coefficients)@r;form=(form+form.T)/2
    second=(l.T*coeff2)@r;second=(second+second.T)/2
    energy=form.square().sum();nativecos=float((form*second).sum()/(energy*second.square().sum()).sqrt())
    ev,basis=torch.linalg.eigh(form)
    replay=float((form-(basis*ev)@basis.T).norm()/form.norm())
    covariance_replay=abs(float(energy/total-q@cov['covariances']['native']@q))/float(energy/total)
    x=torch.randn(8,1152,generator=torch.Generator().manual_seed(923))
    direct=((x@l.T)*(x@r.T))@coefficients
    densevalue=torch.einsum('ni,ij,nj->n',x,form,x)
    execution=float((direct-densevalue).norm()/direct.norm())
    positive=ev.argsort(descending=True);positive=positive[ev[positive]>0]
    negative=ev.argsort();negative=negative[ev[negative]<0]
    rows=[];programs={}
    for k in (1,2,4,8,16):
        pos=positive[:k];neg=negative[:k]
        pp=torch.zeros(1152,k);nn=torch.zeros(1152,k)
        pp[:,:len(pos)]=basis[:,pos]*ev[pos].sqrt();nn[:,:len(neg)]=basis[:,neg]*(-ev[neg]).sqrt()
        a=(pp+nn).T;b=(pp-nn).T
        approximate=(a.T@b+b.T@a)/2
        capture=1-float((form-approximate).square().sum()/energy)
        expected=float((ev[pos].square().sum()+ev[neg].square().sum())/energy)
        row=dict(products=k,capture=capture,spectral_replay=abs(capture-expected),
            parameter_floats=2*k*1152+1152)
        rows.append(row);programs[str(k)]=dict(left=a,right=b,output_readout=q)
    # Native loading magnitudes describe this readout, not orthogonal attribution.
    loading=coefficients.square();loading/=loading.sum()
    result=dict(predictions=dict(pred_a_instrument=max(replay,covariance_replay,execution,max(r['spectral_replay'] for r in rows))<=1e-8,
        pred_b_simple_arithmetic=rows[0]['capture']>=.8 or rows[2]['capture']>=.95,
        pred_c_native_agreement=nativecos>=.99),rows=rows,native_function_cosine=nativecos,
        native_energy_fraction=float(energy/total),native_radial_fraction=float(form.trace().square()/(1152*energy)),
        native_loading_participation=float(1/loading.square().sum()),top_native_products=loading.topk(16).indices.tolist(),
        top16_loading_squared_share=float(loading.topk(16).values.sum()),eigen_replay=replay,
        covariance_replay=covariance_replay,executor_replay=execution,source=source,
        scope='One post-selected native scalar output function. Exact low-product optimum for this fixed readout, not complete-module or behavioral circuit identification.')
    artifact=Path('/dev/shm/bilin18_shared_native_function_products_v1.pt');assert not artifact.exists()
    torch.save(dict(programs=programs,native_form=form,native_output_readout=q,
        native_eigenvalues=ev,source=source,unembedding_whitener=wh),artifact)
    result['cache']=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest());result['seconds']=time.perf_counter()-start
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
