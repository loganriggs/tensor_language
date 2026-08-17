"""The registered follow-on to §13: do the causal directions and the nameable atoms
coincide once interference is handled?

§78 (repo) found dictionary atoms are nameable and not causal -- but "not causal" was
measured by solo ablation, which §13 showed misranks by up to 4.6x and assigns negative
effects to genuinely load-bearing directions. So the recorded orthogonality of the
nameable axis and the causal axis might be an artefact of the causal instrument.

Now both sides are on the table: §13's Shapley values give a solo-free causal ranking of
the 32 SVD directions, and the 4096-atom dictionary from §12 gives the nameable atoms.
No new model evaluations needed -- the question is geometric:

    does the span of the TOP-k Shapley directions hold more of the dictionary's
    best atoms than the span of the BOTTOM-k, and than random k-subsets?

For each of the top 32 atoms (by usage, §78's selection rule), compute the fraction of
its norm lying in span(top-10 Shapley dirs) vs span(bottom-10) within the same top-32
SVD subspace, plus a matched random-10-subset null (200 draws). If the dictionary's
nameable structure lives where the causal mass lives, top >> null > bottom. If §78's
orthogonality is real and not an instrument artefact, top ~ null ~ bottom.
"""
import json, sys, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import collect_out, fit_sae, orth, TRAIN, LAYER, DEV

def main():
    R=json.load(open('bilin18_shapley_results.json'))
    phi=torch.tensor(R['shapley'])
    Ytr=collect_out(TRAIN, LAYER); Ybar=Ytr.mean(0); Yc=Ytr-Ybar
    _,Sv,Vh=torch.linalg.svd(Yc, full_matrices=False)
    Q=orth(Vh[:32].T)                                   # (d,32)
    a4k,u4k,l0,fvu,l1=fit_sae(Ytr,4096,target_l0=40)
    atoms=a4k[u4k.argsort(descending=True)[:32]]        # (32,d) unit rows
    order=phi.argsort(descending=True)
    top=Q[:,order[:10]]; bot=Q[:,order[-10:]]
    A=atoms.to(DEV)
    # fraction of each atom's energy inside each 10-dim span (atoms are unit norm)
    def frac(P): return (A@P).pow(2).sum(1)
    f_top=frac(top); f_bot=frac(bot)
    g=torch.Generator().manual_seed(0)
    nulls=[]
    for _ in range(200):
        idx=torch.randperm(32,generator=g)[:10]
        nulls.append(frac(Q[:,idx]).mean())
    nulls=torch.stack(nulls)
    # usage-weighted versions (the atoms that matter most, matter most)
    w=u4k.sort(descending=True).values[:32].to(DEV); w=w/w.sum()
    out={'l0':l0,'fvu':fvu,
         'mean_frac_top10':float(f_top.mean()),'mean_frac_bottom10':float(f_bot.mean()),
         'weighted_frac_top10':float((f_top*w).sum()),
         'weighted_frac_bottom10':float((f_bot*w).sum()),
         'null_mean':float(nulls.mean()),'null_p95':float(nulls.quantile(0.95)),
         'null_p05':float(nulls.quantile(0.05))}
    print(f'dictionary: L0 {l0:.1f}, FVU {fvu:.3f}')
    print(f'\natom energy inside 10-direction spans of the same 32-dim subspace:')
    print(f"  top-10 Shapley directions:    {out['mean_frac_top10']:.3f} "
          f"(usage-weighted {out['weighted_frac_top10']:.3f})")
    print(f"  bottom-10 Shapley directions: {out['mean_frac_bottom10']:.3f} "
          f"(usage-weighted {out['weighted_frac_bottom10']:.3f})")
    print(f"  random 10-subsets:            {out['null_mean']:.3f} "
          f"[5th-95th pct {out['null_p05']:.3f}-{out['null_p95']:.3f}]")
    ratio=out['mean_frac_top10']/max(out['mean_frac_bottom10'],1e-9)
    above=out['mean_frac_top10']>out['null_p95']
    out['top_over_bottom']=ratio
    out['top_above_null_p95']=bool(above)
    if above and ratio>1.5:
        v='ALIGNED: the nameable atoms concentrate where the causal mass is -- 78\'s orthogonality was at least partly an instrument artefact of solo ablation'
    elif out['mean_frac_top10']<out['null_p05']:
        v='ANTI-ALIGNED: nameable structure actively avoids the causal directions -- 78\'s orthogonality is real and stronger than reported'
    else:
        v='INDIFFERENT: nameable atoms are spread across the causal spectrum -- 78\'s naming/causation split survives the better causal instrument'
    out['verdict']=v
    print(f'\ntop/bottom ratio {ratio:.2f}x, top above the 95th-pct null: {above}')
    print(f'VERDICT: {v}')
    json.dump(out,open('bilin18_shapley_dict_results.json','w'),indent=1)
    print('\nwrote bilin18_shapley_dict_results.json')

if __name__=='__main__': main()
