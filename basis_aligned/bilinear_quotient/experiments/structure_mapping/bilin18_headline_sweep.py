"""Replication sweep of three headline numbers on fresh rows.
(1) Score-rank (section 127): median on-distribution eff-rank of head score
    factors, stats from rows 36-96 (orig 0-24): registered 2.6 <= median <= 6.6.
(2) Watch-list sharing gap (section 130): within-layer filter alignment minus
    covariance-matched null at L9 (orig +0.36 overall): registered gap >= +0.25.
(3) L1 linearization cost (section 107, +0.282): fresh eval rows 384-448:
    registered within +-30% (0.20-0.37)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_pipe_refit as PR
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_headline_sweep_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    out={}
    # (1) score-rank on fresh stats
    caps={li:[] for li in (2,9,16)}
    hs=[]
    for li in caps:
        def mk(li=li):
            return lambda mod,inp: caps[li].append(
                F.rms_norm(inp[0].detach().reshape(-1,D).float(),(D,))) or None
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    for i in range(36,96,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    ranks=[]
    for li in caps:
        X=torch.cat(caps[li]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        a=m.transformer.h[li].attn
        for h_ in range(NH):
            for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
                Wq=wq.weight.detach().float().view(NH,HD,D)[h_]
                Wk=wk.weight.detach().float().view(NH,HD,D)[h_]
                sv=torch.linalg.svdvals(Ch@Wq.T@Wk@Ch); e=sv**2
                ranks.append(float(e.sum()**2/(e**2).sum()))
    mr=sorted(ranks)[len(ranks)//2]
    out['score_rank_median']=mr
    p1=2.6<=mr<=6.6
    print(f'(1) score-rank median (fresh): {mr:.1f} -> {"HELD" if p1 else "FAILED"}',
          flush=True)
    # (2) watch-list gap at L9
    X=torch.cat(caps[9]); Xc=X-X.mean(0)
    C=Xc.T@Xc/Xc.shape[0]
    ev,U=torch.linalg.eigh(C.double())
    Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
    a=m.transformer.h[9].attn
    subs=[]
    for h_ in range(NH):
        for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
            Wq=wq.weight.detach().float().view(NH,HD,D)[h_]
            Wk=wk.weight.detach().float().view(NH,HD,D)[h_]
            Uk,S,Vk=torch.linalg.svd(Ch@Wq.T@Wk@Ch)
            subs.append(orth(torch.cat([Uk[:,:4],Vk[:4].T],dim=1)))
    import itertools
    cs=[]
    for A,B in itertools.combinations(subs,2):
        s_=torch.linalg.svdvals(A.T@B)
        cs+=[float(x) for x in s_[:4]]
    w=sorted(cs)[len(cs)//2]
    g=torch.Generator(device=DEV).manual_seed(1)
    cn=[]
    for _ in range(80):
        R1=orth(Ch@torch.randn(D,8,device=DEV,generator=g))
        R2=orth(Ch@torch.randn(D,8,device=DEV,generator=g))
        s_=torch.linalg.svdvals(R1.T@R2)
        cn+=[float(x) for x in s_[:4]]
    null=sorted(cn)[len(cn)//2]
    gap=w-null
    out['watchlist_within']=w; out['watchlist_null']=null; out['gap']=gap
    p2=gap>=0.25
    print(f'(2) watch-list gap at L9: {w:.2f}-{null:.2f}={gap:+.2f} -> '
          f'{"HELD" if p2 else "FAILED"}',flush=True)
    # (3) L1 linearization on fresh eval rows
    PR.LINS={}
    def ce(lo,hi):
        tot,n=0.0,0
        for i in range(lo,hi,4):
            b=FW[i:i+4,:257].to(DEV)
            lg,_=PR.fwd_lin(b[:,:-1].contiguous())
            c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    base=ce(384,448)
    PR.LINS={1:PR.fit_layer(1)}
    cost=ce(384,448)-base
    PR.LINS={}
    out['l1_linearization']=cost
    p3=0.20<=cost<=0.37
    print(f'(3) L1 linearization (fresh rows): +{cost:.3f} -> '
          f'{"HELD" if p3 else "FAILED"}',flush=True)
    out['p1']=bool(p1); out['p2']=bool(p2); out['p3']=bool(p3)
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
