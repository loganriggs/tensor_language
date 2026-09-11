"""CPU native product-energy heterogeneity versus equal reader votes.
Energy weighting is gauge invariant under native CP scalings, but is not the
full tensor objective because cross-product terms remain. No discovery fit.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import torch

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def energy(l,r,w2):
    return w2*(l.square().sum(1)*r.square().sum(1)+(l*r).sum(1).square())/2


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',mmap=True,weights_only=True)
    u=sd['lm_head.weight'].double()
    gram=u.T@u
    del u
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    w2=(d*(gram@d)).sum(0)
    assert float(w2.min())>0
    e=energy(l,r,w2)
    torch.manual_seed(716)
    a=torch.exp(torch.randn(len(l))*2)*torch.where(torch.rand(len(l))>.5,1.,-1.)
    b=torch.exp(torch.randn(len(l))*2)*torch.where(torch.rand(len(l))>.5,1.,-1.)
    changed=energy(l*a[:,None],r*b[:,None],w2/(a*b).square())
    gauge_error=float((changed-e).norm()/e.norm())
    fractions=e/e.sum()
    top,order=torch.sort(fractions,descending=True)
    cumulative=top.cumsum(0)
    support={str(q):int(torch.searchsorted(cumulative,torch.tensor(q)))+1 for q in (.5,.9,.99)}
    n=len(e)
    top10=float(top[:math.ceil(n/10)].sum())
    effective=float(1/fractions.square().sum())
    split=torch.randperm(n,generator=torch.Generator().manual_seed(700))
    train_fraction=float(fractions[split[:3072]].sum())
    total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    cache=Path('/dev/shm/bilin18_native_product_energy_v1.pt')
    assert not cache.exists()
    torch.save(dict(product_energy=e,normalized_energy=fractions,writer_norm_squared=w2,
        left_norm=l.norm(dim=1),right_norm=r.norm(dim=1),unembedding_gram=gram),cache)
    result=dict(predictions={
        'pred_a_scaling_invariant':gauge_error<=1e-10,
        'pred_b_top_decile_majority':top10>=.5,
        'pred_c_effective_count_halved':effective<=2304,
        'pred_d_weight_split_balanced':abs(train_fraction-2/3)<=.05},
        gauge_relative_error=gauge_error,top_decile_energy_fraction=top10,
        effective_product_count=effective,support_for_component_energy=support,
        training_product_energy_fraction=train_fraction,
        sum_component_energy=float(e.sum()),native_total=total,
        component_sum_over_full_tensor_energy=float(e.sum()/total),
        energy_quantiles={str(q):float(torch.quantile(e,q)) for q in (0.,.1,.5,.9,.99,1.)},
        largest_products=order[:20].tolist(),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size),
        source_sha256=digest(__file__),seconds=time.perf_counter()-started,
        gpu_access=False,corpus_access=False,
        scope='Native-factor metric mismatch diagnostic; not component selection, joint tensor factorization or behavioral importance.')
    with (P/'NATIVE_READER_METRIC_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
