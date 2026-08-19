"""CIRCUIT PIPELINE Stages B+C: discovery + mechanical structural
certification. From circuit_atlas_big.pt: well-predicted tokens (base CE
<= median), z-score profiles using DISCOVERY-half statistics only, k-means
k=256 on the discovery half, then certify each cluster on the REPLICATION
half (tokens assigned to nearest centroid).

A cluster is a CERTIFIED OWNERSHIP CIRCUIT when, on held-out data:
  (i) profile replication: cosine(disc mean |z| profile, rep mean |z|
      profile) >= 0.7;
 (ii) ownership replication: >= 2 of top-3 components shared between
      halves;
(iii) cohesion above floor: rep-token mean cosine to centroid exceeds the
      measured random-centroid floor by >= 0.10;
 (iv) power: >= 40 discovery tokens and >= 20 replication tokens.

REGISTERED PREDICTIONS: (a) >= 60 certified ownership circuits; (b) >= 40%
of powered clusters certify; (c) certified slices cover >= 30% of
well-predicted tokens; (d) the random-centroid floor is reported (rule).
Outputs circuits_registry.pt + CIRCUITS_SCOREBOARD.md."""
import json, sys, time, torch
DEV='cuda'
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_certify_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    d=torch.load(PT+'circuit_atlas_big.pt',weights_only=False)
    base=d['base'].float().to(DEV)
    keys=sorted(d['fingerprints'])
    M=torch.stack([d['fingerprints'][k].float().to(DEV) for k in keys])
    even=d['even_mask'].to(DEV)
    wp=base<=base.median()
    disc=wp&even; rep=wp&(~even)
    mu=M[:,disc].mean(1,keepdim=True)
    sd=M[:,disc].std(1,keepdim=True).clamp_min(1e-8)
    Z=(M-mu)/sd
    Xd=Z[:,disc].T; Xd=Xd/Xd.norm(dim=1,keepdim=True).clamp_min(1e-8)
    Xr=Z[:,rep].T;  Xr=Xr/Xr.norm(dim=1,keepdim=True).clamp_min(1e-8)
    K=256
    g=torch.Generator(device=DEV).manual_seed(0)
    C=Xd[torch.randperm(len(Xd),generator=g,device=DEV)[:K]].clone()
    for _ in range(100):
        a=(Xd@C.T).argmax(1)
        for j in range(K):
            m_=a==j
            if m_.any():
                C[j]=Xd[m_].mean(0); C[j]=C[j]/C[j].norm().clamp_min(1e-8)
    ar=(Xr@C.T).argmax(1)
    gr=torch.Generator(device=DEV).manual_seed(1)
    floor=float((Xr*(C[torch.randint(0,K,(len(Xr),),generator=gr,
                                     device=DEV)])).sum(1).mean())
    certified=[]; powered=0
    for j in range(K):
        md=a==j; mr=ar==j
        nd,nr=int(md.sum()),int(mr.sum())
        if nd<40 or nr<20: continue
        powered+=1
        pd=Xd[md].abs().mean(0); pr=Xr[mr].abs().mean(0)
        pcos=float((pd/pd.norm())@(pr/pr.norm()))
        td=set(pd.argsort(descending=True)[:3].tolist())
        tr=set(pr.argsort(descending=True)[:3].tolist())
        ov=len(td&tr)
        coh=float((Xr[mr]*C[j]).sum(1).mean())
        ok=pcos>=0.7 and ov>=2 and (coh-floor)>=0.10
        if ok:
            sh=pd/pd.sum()
            top=sh.argsort(descending=True)[:5]
            certified.append({'id':j,'n_disc':nd,'n_rep':nr,
                'top':[(keys[i],round(float(sh[i]),3)) for i in top.tolist()],
                'profile_cos':round(pcos,3),'overlap':ov,
                'cohesion':round(coh,3)})
    cov=sum(c['n_disc']+c['n_rep'] for c in certified)/float(
        int(disc.sum())+int(rep.sum()))
    pa=len(certified)>=60
    pb=powered>0 and len(certified)/powered>=0.40
    pc=cov>=0.30
    torch.save({'centroids':C.cpu(),'assign_disc':a.cpu(),
                'assign_rep':ar.cpu(),'disc_mask':disc.cpu(),
                'rep_mask':rep.cpu(),'keys':keys,
                'certified':certified},PT+'circuits_registry.pt')
    with open(PT+'CIRCUITS_SCOREBOARD.md','w') as f:
        f.write('# Circuit scoreboard\n\nCertified OWNERSHIP circuits '
                '(structural claims, held-out replicated). Semantic '
                'stories pending per circuit (Stage D).\n\n'
                f'- certified: {len(certified)} of {powered} powered '
                f'clusters\n- coverage of well-predicted tokens: '
                f'{cov:.0%}\n- floor (random-centroid cohesion): '
                f'{floor:.3f}\n\n|id|n(disc/rep)|top components '
                '(damage share)|profile cos|\n|--|--|--|--|\n')
        for c in certified:
            tops=' '.join(f'{k}:{s}' for k,s in c['top'][:3])
            f.write(f"|{c['id']}|{c['n_disc']}/{c['n_rep']}|{tops}|"
                    f"{c['profile_cos']}|\n")
    out={'certified':len(certified),'powered':powered,'coverage':cov,
         'floor':floor,'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'certified {len(certified)}/{powered} powered | coverage '
          f'{cov:.0%} | floor {floor:.3f}')
    print(f"(a) >=60 certified: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=40% certify: {'HELD' if pb else 'FAILED'}")
    print(f"(c) coverage >=30%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
