"""Fixed-support feasibility and maximal concentration of residual-space writes.

Numerical bounds for specified sets only; no generic-cospark claim about trained U.
"""
from pathlib import Path
import json, time, hashlib
import torch
import tiktoken
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def support_bound(u, indices, chol):
    us=u[indices]
    z=torch.linalg.solve_triangular(chol,us.T,upper=False)
    eigen, vectors=torch.linalg.eigh(z.T@z)
    upper=float(eigen[-1])
    # A maximizing witness independently attains the generalized Rayleigh bound.
    y=z@vectors[:,-1]; w=torch.linalg.solve_triangular(chol.T,y[:,None],upper=True)[:,0]
    code=u@w; achieved=float(code[indices].square().sum()/code.square().sum())
    outside=chol@chol.T-us.T@us
    spectrum=torch.linalg.eigvalsh(outside)
    return dict(max_fraction_squared_loading_in_group=upper,
        min_fraction_squared_loading_outside_group=1-upper,
        witness_fraction=achieved, witness_absolute_error=abs(achieved-upper),
        outside_gram_min_eigenvalue=float(spectrum[0]),
        outside_gram_min_to_max_eigenvalue=float(spectrum[0]/spectrum[-1]),
        outside_numerical_rank=int((spectrum>spectrum[-1]*1e-10).sum()))

def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    toy=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    control=support_bound(toy,[0],torch.linalg.cholesky(toy.T@toy))
    assert abs(control['max_fraction_squared_loading_in_group']-2/3)<1e-12
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'][:50257].double();M=u.T@u;chol=torch.linalg.cholesky(M)
    compact=torch.load(P/'STABLE_JOINT32_REFLECTION_V1_PROGRAM.pt',map_location='cpu',weights_only=True)
    codes=u@compact['programs']['native']['w'];enc=tiktoken.get_encoding('gpt2')
    groups={'six_pronouns':compact['pronoun_ids']}
    for j,name in enumerate(['stable_native_0_top12_absolute','stable_native_9_top12_absolute']):
        groups[name]=codes[:,j].abs().topk(12).indices.tolist()
    reports={}
    for name,ids in groups.items():
        reports[name]=support_bound(u,ids,chol)
        reports[name].update(token_ids=ids,tokens=[enc.decode([i]) for i in ids],
            recovered_pair_loading_fractions=(codes[ids].square().sum(0)/codes.square().sum(0)).tolist())
        assert reports[name]['witness_absolute_error']<1e-10
    result=json.loads((P/'STABLE_JOINT32_REFLECTION_V1_RESULT.json').read_text())
    rows=result['rows'];flatten=lambda k:np.asarray([v for r in rows for v in r[k]],dtype=float)
    nc,nn=flatten('class_count'),flatten('nonclass_count')
    bs=PairedPanelBootstrap(96,9114901)
    paired={}
    for name in ['native','random']:
        ps=lambda k:np.asarray([v for r in rows for v in r['pairs'][name][k]],dtype=float)
        def ratio(values,counts):
            den=counts[bs.indices].sum(1);assert (den>0).all()
            return np.quantile(values[bs.indices].sum(1)/den,[.025,.975]).tolist()
        paired[name]=dict(response_error_interval=bs.relative_l2(ps('response_error_squared'),flatten('full_response_squared')),
            target_ce_response_error_interval=bs.relative_l2(ps('ce_error_squared'),flatten('full_ce_squared')),
            removal_class_absolute_ce_interval=ratio(ps('removal_class_abs'),nc),
            removal_nonclass_absolute_ce_interval=ratio(ps('removal_nonclass_abs'),nn))
    out=dict(scope='Numerical fixed-group bounds on linear numerator writes Uw; not exact cospark of trained U, not final probability or behavior bounds.',
        groups=reports,analytic_control=control,reflection_paired=paired,
        interpretation='Every specified complement has full numerical column rank. No nonzero residual writer can be supported exactly on these groups, subject to numerical rank qualification. The extremal concentration bound applies to any residual writer, not only fitted products.',
        complexity='One O(V d^2) Gram and O(d^3) factorization; each size-s group O(s d^2+s^3), plus O(d^3) complement spectral diagnostic; O(Vd+d^2) storage.',
        cpu_seconds=time.perf_counter()-tic,native_forwards=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=P/'UNEMBEDDING_SUPPORT_FEASIBILITY_V1_AUDIT.json'
    with target.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
