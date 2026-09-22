"""Canonical feature-space and aggregate correction stability across two hybrid fits."""
import json
import time
import torch
from audit_conditional_residual_accounting import P, load
from mixed_gaussian_cp import gram_dynamic


def compare(G, H, X, c, d):
    L, R = torch.linalg.cholesky(G), torch.linalg.cholesky(H)
    left = torch.linalg.solve_triangular(L, X, upper=False)
    white = torch.linalg.solve_triangular(R, left.T, upper=False).T
    corr = torch.linalg.svdvals(white)
    assert torch.isfinite(corr).all() and corr.max() <= 1+1e-7
    cosine = (c @ X @ d) / ((c @ G @ c)*(d @ H @ d)).sqrt()
    assert torch.isfinite(cosine) and abs(cosine) <= 1+1e-7
    return dict(canonical_correlations=corr.tolist(), min_correlation=float(corr.min()), correction_cosine=float(cosine))


def main():
    start=time.monotonic(); torch.set_num_threads(2); torch.set_grad_enabled(False)
    torch.manual_seed(59001)
    a=torch.randn(32,8,dtype=torch.float64); g=a.T@a; c=torch.randn(8,dtype=torch.float64)
    control=compare(g,g,g,c,c)
    assert max(abs(v-1) for v in control['canonical_correlations']) < 1e-9
    first,sha1=load('HYBRID_LOCAL_QUARTIC_ADAM_SEED25001_V1.pt')
    second,sha2=load('HYBRID_LOCAL_QUARTIC_ADAM_SEED25002_V1.pt')
    receipt=json.loads((P/'HYBRID_LOCAL_QUARTIC_NATIVE_V1.json').read_text())
    assert {r['sha256'] for r in receipt['rows']} == {sha1,sha2}
    assert first['parent_sha256']==second['parent_sha256']
    cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True)
    S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double()
    data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double()
    rows=[]
    for output in range(12):
        sl=slice(output*8,(output+1)*8)
        f=[a[sl] for a in first['factors']];h=[a[sl] for a in second['factors']]
        fs,hs=[a@S for a in f],[a@S for a in h];fb,hb=[a@mu for a in f],[a@mu for a in h]
        G=gram_dynamic(fs,fb,fs,fb);H=gram_dynamic(hs,hb,hs,hb);X=gram_dynamic(fs,fb,hs,hb)
        c,d=first['coefficients'][output],second['coefficients'][output]
        gaussian=compare(G,H,X,c,d)
        phi=torch.stack([x@a.T for a in f]).prod(0);psi=torch.stack([x@a.T for a in h]).prod(0)
        empirical=compare(phi.T@phi,psi.T@psi,phi.T@psi,c,d)
        rows.append(dict(output=output+4,gaussian=gaussian,empirical=empirical))
    preds=dict(pred_a_integrity=True,
               pred_b_subspace=all(r[m]['min_correlation']>=.9 for r in rows for m in ['gaussian','empirical']),
               pred_c_correction=all(r[m]['correction_cosine']>=.9 for r in rows for m in ['gaussian','empirical']))
    result=dict(rows=rows,predictions=preds,sha256=[sha1,sha2],control=control,seconds=time.monotonic()-start,
                scope='Frozen dictionaries, exact Gaussian and opened text panel; no refitting, causal/OOD or circuit adoption claim.')
    (P/'HYBRID_FEATURE_STABILITY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(preds)
    for metric in ['gaussian','empirical']:
        print(metric,'minimum canonical',min(r[metric]['min_correlation'] for r in rows),'minimum correction cosine',min(r[metric]['correction_cosine'] for r in rows))


if __name__=='__main__': main()
