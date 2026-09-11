"""Exact output-only least-squares frontier under a component-energy budget.

A energy budget relative<=1e-9, normal/CP<=1e-8; B capture loss<=.001 at4x reduction.
The previous coarse-grid miss remains recorded. Fixed input features/readers.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from folded_sparse_dictionary_v1 import decode
from joint_quadratic_fit_v1 import product_cross
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'OUTPUT_COMPONENT_BUDGET_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    parent=json.loads((P/'PROJECTED_SPARSE_DICTIONARY_FIT_V1_SEED_0.json').read_text());source=parent['cache']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu');ids=saved['code_indices'].long();values=saved['code_values'].double()
    readers,basis,_,_=decode(saved['analysis_basis'].double(),ids,values,torch.ones(len(ids)));a,b=readers.chunk(2)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;native=(l,r,wh@down)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    gram=product_cross(a,b,a,b);norms=gram.diag().sqrt();normalized=gram/norms[:,None]/norms[None,:]
    eig,vec=torch.linalg.eigh((normalized+normalized.T)/2);assert eig[0]>0
    rhs=native[2]@product_cross(l,r,a,b);h=(rhs/norms[None,:])@vec;q=h.square().sum(0)/total
    energy=lambda lam:float((q/(eig+lam).square()).sum())
    base=energy(0.);budget=base/4;low,high=0.,.01
    while energy(high)>budget:high*=2
    for _ in range(60):
        middle=(low+high)/2
        if energy(middle)>budget:low=middle
        else:high=middle
    penalty=high;coefficients=h/(eig[None,:]+penalty);w=(coefficients@vec.T)/norms[None,:]
    capture=float((q*(eig+2*penalty)/(eig+penalty).square()).sum())
    component=energy(penalty);budget_error=abs(component-budget)/budget
    normal=float((w@gram+penalty*w*norms.square()[None,:]-rhs).norm()/rhs.norm())
    proposal=(a,b,w);cp_capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
    replay=abs(capture-cp_capture);loss=parent['final_capture']-capture
    cache=Path('/dev/shm/bilin18_output_component_budget_v1.pt');assert not cache.exists()
    torch.save(dict(analysis_basis=basis,code_indices=ids.to(torch.int16),code_values=values,
        down=torch.linalg.solve_triangular(wh,w,upper=True),retained_bias_key=saved['retained_bias_key'],
        penalty=penalty,source=source,normalized_gram_eigenvalues=eig,rhs_spectral_energy=q),cache)
    result=dict(predictions=dict(pred_a_instrument=budget_error<=1e-9 and max(normal,replay)<=1e-8,
        pred_b_low_cost=loss<=.001),penalty=penalty,component_energy=component,original_component_energy=base,
        capture=capture,capture_loss=loss,budget_relative_error=budget_error,normal_residual=normal,cp_replay=replay,
        source=source,cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
        seconds=time.perf_counter()-start,
        scope='Global convex optimum for output weights at this fixed reader configuration and component budget. '
              'Not a joint feature optimum, convergence repair, same objective or circuit adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
