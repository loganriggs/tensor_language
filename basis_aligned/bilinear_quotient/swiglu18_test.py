"""FOURTH MODEL: swiglu18 -- same depth as bilin18 (18 layers), swiglu-gated
MLPs. At equal depth absolute=fractional, so this tests the laws and the
correspondence without the depth-warp ambiguity. REGISTERED: (a) identity CE <=
4.5; (b) dilution: tail ratios decline (<= 2 inversions, layers 5-16); (c)
front-loaded linearization: peak in L0-L2 and >= 2x the median of L5/L9
(bilinear-family law -- swiglu is still a gated product, prediction extends);
(d) correspondence: attention fingerprints at L1/L2/L6/L9 best-match bilin18's
SAME layers (within +-2) at >= 3x the 0.05 floor, for >= 3/4.

Prior context (bilin12 v2): do bilin18's structural laws hold in its sibling checkpoint
bilin12 (same bilinear-sqrd-attn family, 12 layers, 6 heads, 768 dims)? Three
load-bearing structures, cheapest instruments:

REGISTERED PREDICTIONS (grounded in bilin18's values):
(a) DILUTION: MLP write-to-stream ratios decline monotonically through the tail
    (layers 3-10, <= 1 inversion), as in bilin18 (0 inversions).
(b) FRONT-LOADED NONLINEARITY: linearization cost (consistent in-forward
    protocol) peaks in L0-L2 and the peak is >= 2x the median of mid layers
    (bilin18: L1 +0.28 vs mid ~0.03).
(c) MATCHED FILTERS: median on-distribution score eff-rank <= 12 of 128, and
    row-shuffled weights >= 2x the trained median (bilin18: 4.3 vs 18.7).
Also logged, no bar: the lambda table (bilin18 re-injects embeddings at 8.0).
Any failure = a bilin18-specific result, itself a finding."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs, rope_tables, apply_rot
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'swiglu18_test_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('swiglu18', device=DEV)
    NL=len(m2.transformer.h); D=m2.transformer.wte.weight.shape[1]
    NH=cfg.get('n_head',6); HD=D//NH
    print(f'bilin12 loaded: {NL} layers, {NH} heads, d={D}, hd={HD}',flush=True)
    lam_tab=[(float(b.lambdas[0]),float(b.lambdas[1])) for b in m2.transformer.h]
    print('lambdas (l0,l1):',' '.join(f'{a:.2f}/{b:.2f}' for a,b in lam_tab),flush=True)
    def fwd_all(idx, lins=None):
        # module-level blocks with manual lambda-mix so post-mix input is exact
        x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
        ins={};mos={};hcs={}
        for li in range(NL):
            blk=m2.transformer.h[li]
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            ins[li]=x.detach().reshape(-1,D).float()
            hcur=F.rms_norm(x,(D,))
            hcs[li]=hcur.detach().reshape(-1,D).float()
            x1,v1=blk.attn(hcur,v1)
            x=x+x1
            if lins is not None and li in lins:
                mp=lins[li]
                xi=ins[li]
                mo=((xi-mp['bx'])@mp['W']+mp['by']).to(x.dtype).view_as(x)
            else:
                mo=blk.mlp(F.rms_norm(x,(D,)))
            mos[li]=mo.detach().reshape(-1,D).float()
            x=x+mo
        lg=m2.lm_head(F.rms_norm(x,(D,)))
        return ins,mos,hcs,(30*torch.tanh(lg/30)).float()
    # collect stats
    tri=[]
    for i in range(0,36,6): tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    # (a) dilution ratios
    ratios={}
    for li in range(5,NL-1):
        Y=torch.cat([t[1][li] for t in tri]); X=torch.cat([t[0][li+1] for t in tri])
        ratios[li]=float((Y-Y.mean(0)).pow(2).sum(1).mean())/ \
                   float((X-X.mean(0)).pow(2).sum(1).mean())
    seq=[ratios[li] for li in sorted(ratios)]
    inv=sum(1 for i in range(len(seq)-1) if seq[i+1]>seq[i])
    pa=inv<=2
    print('dilution ratios: '+' '.join(f'{r:.3f}' for r in seq)
          +f' | inversions {inv} -> (a) {"HELD" if pa else "FAILED"}',flush=True)
    # (b) linearization costs L0,1,2,5,8 (in-forward stand-in)
    def fit(li):
        X=torch.cat([t[0][li] for t in tri]); Y=torch.cat([t[1][li] for t in tri])
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        return {'W':W,'bx':bx,'by':by}
    LINS={}
    def fwd_ce(idx,tg):
        _,_,_,lg=fwd_all(idx, lins=LINS if LINS else None)
        return float(F.cross_entropy(lg.view(-1,lg.size(-1)),tg))
    def ce():
        tot,n=0.0,0
        for i in range(300,348,4):
            b=FW[i:i+4,:257].to(DEV)
            c=fwd_ce(b[:,:-1].contiguous(), b[:,1:].reshape(-1))
            tot+=c*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    base=ce()
    costs={}
    for li in (0,1,2,5,9):
        LINS={li:fit(li)}
        costs[li]=ce()-base
        LINS={}
        print(f'L{li}: linearization cost +{costs[li]:.4f}',flush=True)
    peak=max(costs[li] for li in (0,1,2))
    midmed=(costs[5]+costs[9])/2
    pb=peak==max(costs.values()) and peak>=2*midmed
    print(f'(b) front peak {peak:.3f} vs mid {midmed:.3f} -> '
          f'{"HELD" if pb else "FAILED"}',flush=True)
    # (d) correspondence fingerprints vs bilin18 same layers
    d18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprints.pt')
    f18={k:v.float() for k,v in d18['fingerprints'].items()}
    def spearman(a,b):
        ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
        ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
        rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
        return float((ra*rb).mean())
    def pertok(ablate=None):
        ces=[]
        for i in range(384,448,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            hs=[]
            if ablate is not None:
                ali,mu=ablate
                def hook(mod,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu[None,None,:].to(y.dtype).expand_as(y), v1)
                hs.append(m2.transformer.h[ali].attn.register_forward_hook(hook))
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m2.transformer.h:
                x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        return torch.cat(ces)
    amu={}
    hs=[]
    for li in (1,2,6,9):
        def mka(li=li):
            def hook(mod,i_,o_):
                y,v1=o_
                amu.setdefault(li,[]).append(y.detach().reshape(-1,D).float())
            return hook
        hs.append(m2.transformer.h[li].attn.register_forward_hook(mka()))
    for i in range(0,12,6):
        b=FW[i:i+6,:257].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    ce0=pertok()
    L18avail=[int(k[4:]) for k in f18 if k.startswith('attn')]
    hits=0; matches={}
    for li in (1,2,6,9):
        mu=torch.cat(amu[li]).mean(0)
        fp=(pertok(ablate=(li,mu))-ce0).cpu().float()
        cur={lj:abs(spearman(fp,f18[f'attn{lj}'])) for lj in L18avail}
        best=max(cur,key=cur.get)
        matches[li]=(best,cur[best])
        ok=abs(best-li)<=2 and cur[best]>=0.15
        if ok: hits+=1
        print(f'swiglu18 attn{li}: best bilin18 attn{best} ({cur[best]:.2f}) '
              f'-> {"hit" if ok else "miss"}',flush=True)
    pd=hits>=3
    print(f'(d) same-layer correspondence >= 3/4: '
          f'{"HELD" if pd else "FAILED"} ({hits}/4)',flush=True)
    out={'lambdas':lam_tab,'dilution':ratios,'inversions':inv,'costs':costs,
         'matches':{str(k):v for k,v in matches.items()},
         'pa':bool(pa),'pb':bool(pb),'pd':bool(pd),'base_ce':base}
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
