"""Exact fixed-support output-code refit; no new learned atoms or support links."""
import json,time,hashlib
from pathlib import Path
import torch
from quadratic_token_dictionary_v1 import embed
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    receipt=json.loads((P/'TOKEN_FUNCTION_DICTIONARY_V1_RESULT.json').read_text());cache=Path(receipt['cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu');a=saved['codes'].double();b=saved['dictionary'].double()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double();l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    x,root,basis,scale,mean=embed(u,l,r,d)
    g=b@b.T;c=x@b.T;support=a!=0;counts=support.sum(1)
    debiased=torch.zeros_like(a);max_residual=0.;max_condition=0.;rank_deficient=0
    for count in counts.unique().tolist():
        if count==0:continue
        rows=torch.where(counts==count)[0]
        for chunk in rows.split(512):
            indices=support[chunk].nonzero()[:,1].reshape(len(chunk),count)
            small=g[indices[:,:,None],indices[:,None,:]]
            rhs=c[chunk[:,None],indices]
            ev,vec=torch.linalg.eigh((small+small.transpose(1,2))/2)
            keep=ev>ev[:,-1:]*1e-10
            rank_deficient+=int((~keep.all(1)).sum())
            inv=torch.where(keep,ev.clamp_min(1e-30).reciprocal(),0.)
            solution=(vec@(inv[:,:,None]*(vec.transpose(1,2)@rhs[:,:,None]))).squeeze(-1)
            residual=(small@solution[:,:,None]).squeeze(-1)-rhs
            max_residual=max(max_residual,float((residual.norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-30)).max()))
            max_condition=max(max_condition,float((ev[:,-1]/ev[:,0]).max()))
            debiased[chunk[:,None],indices]=solution
    error=float((x-debiased@b).square().sum()/x.square().sum())
    original_error=float((x-a@b).square().sum()/x.square().sum())
    support_ok=bool((debiased[~support]==0).all())
    target=Path('/dev/shm/bilin18_token_dictionary_debiased_v1.pt')
    assert not target.exists()
    torch.save(dict(codes=debiased,dictionary=b,source_cache_sha256=receipt['cache']['sha256']),target)
    result=dict(instrument_passed=support_ok and max_residual<=1e-8 and error<=original_error+1e-10,
                original_capture=1-original_error,debiased_capture=1-error,
                capture_gain=original_error-error,maximum_relative_solve_residual=max_residual,
                maximum_support_gram_condition=max_condition,rank_deficient_rows=rank_deficient,
                original_support_links=int(support.sum()),debiased_nonzero_codes=int((debiased!=0).sum()),
                maximum_active=int(counts.max()),median_active=float(counts.double().median()),
                coefficient_norm_ratio=float(debiased.norm()/a.norm()),
                coefficient_sign_changes=int(((debiased*a)<0).sum()),
                cache=dict(path=str(target),bytes=target.stat().st_size,
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest(),ephemeral=True),
                wall_seconds=time.perf_counter()-start,
                scope='Unpenalized refit on frozen learned functions and exact same token supports. Changed coefficients, not convergence of the original penalty-free joint problem.')
    (P/'TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
