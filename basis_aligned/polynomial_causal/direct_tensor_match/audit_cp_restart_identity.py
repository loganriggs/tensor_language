"""Gauge-aware atom alignment and whole-polynomial restart agreement.
Measures candidate agreement, not truth, causality or identified semantic units.
"""
import argparse,json,time
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from quartic_cp import cp_gram
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import affine_moment
P=Path(__file__).resolve().parent
SCALE=19054614563.464127


def comparison(a,b,gram,ga,gb):
    fa,ca=a;fb,cb=b
    aa=ca.T@ca;bb=cb.T@cb;ab=ca.T@cb
    na=float((aa*ga).sum());nb=float((bb*gb).sum());cross=float((ab*gram).sum())
    atom_a=aa.diag()*ga.diag();atom_b=bb.diag()*gb.diag()
    similarity=(ab*gram)/(atom_a[:,None]*atom_b[None,:]).clamp_min(1e-300).sqrt()
    i,j=linear_sum_assignment(-similarity.numpy())
    matched=similarity[i,j]
    # Allow arbitrary linear combinations of atoms, distinguishing a stable span
    # from stable individual contribution vectors. Normalize atoms before cutoff.
    da=ga.diag().clamp_min(1e-300).sqrt();db=gb.diag().clamp_min(1e-300).sqrt()
    ga_unit=ga/(da[:,None]*da[None,:]);gb_unit=gb/(db[:,None]*db[None,:])
    def whitening(g):
        e,v=torch.linalg.eigh((g+g.T)/2);keep=e>e[-1]*1e-10
        return v[:,keep]/e[keep].sqrt(),int(keep.sum())
    wa,ra=whitening(ga_unit);wb,rb=whitening(gb_unit)
    canonical=torch.linalg.svdvals(wa.T@(gram/(da[:,None]*db[None,:]))@wb)
    target_cross=cb@gram.T/da[None,:]
    captured=float((target_cross@wa).square().sum())
    assert captured<=nb+1e-7*max(nb,1e-30),(captured,nb)
    return dict(span_rank_first=ra,span_rank_second=rb,
                second_function_projection_error_in_first_span=(max(0,nb-captured)/nb)**.5,
                subspace_canonical_median=float(canonical.median()),
                subspace_directions_above_09=int((canonical>.9).sum()),
                subspace_directions_above_099=int((canonical>.99).sum()),
                function_cosine=cross/(na*nb)**.5,relative_function_difference=(max(0,na+nb-2*cross)/na)**.5,
                assignment_mean_contribution_cosine=float(matched.mean()),assignment_median_contribution_cosine=float(matched.median()),
                matched_above_09=int((matched>.9).sum()),matched_above_099=int((matched>.99).sum()),
                maximum_contribution_cosine=float(similarity.max()),terms=len(i))


def controls():
    torch.manual_seed(11500);f=[torch.randn(7,4,dtype=torch.float64) for _ in range(4)];c=torch.randn(3,7,dtype=torch.float64)
    perm=torch.randperm(7);g=[f[i][perm].clone() for i in [2,0,3,1]];d=c[:,perm].clone();g[0]*=-2.;d/=-2.
    result=comparison((f,c),(g,d),cp_gram(f,g),cp_gram(f,f),cp_gram(g,g))
    assert result['matched_above_099']==7 and abs(result['function_cosine']-1)<1e-12
    assert result['relative_function_difference']<1e-7
    assert result['second_function_projection_error_in_first_span']<1e-7
    assert result['subspace_directions_above_099']==7
    A=torch.randn(7,7,dtype=torch.float64)+3*torch.eye(7,dtype=torch.float64)
    G=cp_gram(f,f);changed=torch.linalg.solve(A.T,c.T).T
    # An arbitrary basis mixture need not remain one CP atom per coordinate;
    # it is nevertheless a valid polynomial-dictionary gauge control.
    basis=comparison((f,c),(f,changed),G@A.T,G,A@G@A.T)
    assert basis['relative_function_difference']<1e-7 and basis['second_function_projection_error_in_first_span']<1e-7
    assert basis['subspace_directions_above_099']==7
    return dict(cp_gauge=result,arbitrary_dictionary_basis=basis)


def main(include_mixed=False):
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();check=controls()
    saved=torch.load(P/'EXPANDED_INPUT_GEOMETRY_V1.pt',weights_only=True)
    S=torch.linalg.cholesky(saved['covariance'].double());mu=saved['mean'].double();rows=[]
    names=['coefficient']+(['mixed'] if include_mixed else [])
    for name in names:
        models=[]
        for seed in [1001,1002]:
            file=f'QUARTIC_CP512_SEED{seed}_V2.pt' if name=='coefficient' else f'MIXED_CP_FEATURES_SEED{seed}_V1.pt'
            p=torch.load(P/file,weights_only=True);models.append(([a.double() for a in p['factors']],p['coefficients'].double()/SCALE))
        a,b=models;ga,gb,gab=cp_gram(a[0],a[0]),cp_gram(b[0],b[0]),cp_gram(a[0],b[0])
        coeff=comparison(a,b,gab,ga,gb)
        fw=[[f@S for f in model[0]] for model in models];bias=[[f@mu for f in model[0]] for model in models]
        ga=gram_dynamic(fw[0],bias[0],fw[0],bias[0]);gb=gram_dynamic(fw[1],bias[1],fw[1],bias[1]);gab=gram_dynamic(fw[0],bias[0],fw[1],bias[1])
        raw=comparison(a,b,gab,ga,gb)
        ma,mb=[affine_moment(f,v) for f,v in zip(fw,bias)]
        centered=comparison(a,b,gab-ma[:,None]*mb[None,:],ga-ma[:,None]*ma[None,:],gb-mb[:,None]*mb[None,:])
        rows.append(dict(program=name,coefficient=coeff,shifted_gaussian=raw,centered_shifted_gaussian=centered))
    result=dict(rows=rows,gauge_control=check,seconds=time.monotonic()-start,scope='Candidate-candidate comparisons only in fixed16-output coordinate metric. Hungarian assignment maximizes signed contribution cosine, counting actual output effect; opposite effects are not identity. Invariant to CP term/slot permutation and compensating factor/readout scaling. Raw Gaussian agreement may be dominated by means; centered comparison removes means but not other low-degree chaos. High aggregate agreement is not stable individual units or causal identification.')
    suffix='MIXED' if include_mixed else 'BASELINE'
    (P/f'CP_RESTART_IDENTITY_{suffix}_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-mixed',action='store_true');args=parser.parse_args();main(args.include_mixed)
