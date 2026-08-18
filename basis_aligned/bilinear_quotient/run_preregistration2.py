"""Scorecard v2 -- implements the four hardened rules the v1 validation
bought (sections 228-230): POOLED aggregation only; THREE reader ensembles
with the notch bar RELATIVE (<= 0.5x the in-model control median, every
ensemble); P2 span fits REPLICATED on disjoint stats rows; tail bar in
ABSOLUTE form (>= 0.35 every ensemble). Usage: python run_preregistration2.py
<model>. P6-P8 remain MANUAL.

VALIDATION REGISTRATION (model=bilin12, known ground truth): (a) P2 now
HELD -- the v1 L7 boundary flip does not replicate on disjoint fits;
(b) P3: L4 is the unique profile minimum in ALL 3 ensembles (rank
robustness), and the relative <= 0.5x bar holds in >= 2/3 ensembles
(section 228 showed set-B pooled 0.26 vs control ~0.45 sits near the
boundary -- if it fails in one ensemble the scorecard must SAY so, which is
the honest verdict for bilin12; on an 18L-class target the bar is expected
to hold in all); fraction + width + tail sub-bars all HELD; (c) P1/P4/P5
verdicts unchanged from v1 (FAILED-by-size / HELD / HELD); (d) < 15 min."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
import torch.nn.functional as F
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=48; KT=40; NF=40
MODEL=sys.argv[1] if len(sys.argv)>1 else 'bilin12'
OUT=(f'/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     f'prereg2_scorecard_{MODEL}_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs(MODEL, device=DEV)
    D=m2.transformer.wte.weight.shape[1]; NL=len(m2.transformer.h)
    def grab(li, r0, r1):
        outs=[]
        h=m2.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(r0,r1,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    def ce(hook_li=None, mode=None, Q=None, cbar=None, const=None):
        hs=[]
        if hook_li is not None:
            def hook(mod,i_,o_):
                if mode=='const': return const.to(o_.dtype).expand_as(o_)
                c=o_.float().reshape(-1,D)@Q
                return o_-((c-cbar)@Q.T).to(o_.dtype).view_as(o_)
            hs.append(m2.transformer.h[hook_li].mlp
                      .register_forward_hook(hook))
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            loss=m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
            ntok=(b.shape[1]-1)*b.shape[0]
            tot+=float(loss)*ntok; n+=ntok
        for h in hs: h.remove()
        return tot/n
    base=ce()
    tail=list(range(NL//3, NL))
    licensed=[]; improves=[]
    for li in tail:
        mu=grab(li,0,60).mean(0)
        c=ce(li,'const',const=mu)-base
        if c<=0.05: licensed.append(li)
        flags=[]
        for r0,r1 in ((0,30),(30,60)):
            Y=grab(li,r0,r1); Yb=Y.mean(0)
            _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
            Q=orth(Vh[:8].T)
            cs=ce(li,'span',Q=Q,cbar=Yb.float()@Q)-base
            flags.append(cs<=-0.01)
        if all(flags): improves.append(li)
        print(f'P1/P2 L{li:2d}: const {c:+.3f} span-flags {flags}',flush=True)
    p1=len(licensed)>=4
    p2=set(improves)<=set(licensed)
    # ---- P3-P5: three ensembles, pooled aggregation
    def mkens(shift):
        return tuple(sorted(set(min(NL-1,max(1,round(f*NL))+shift)
                     for f in (0.11,0.17,0.28,0.5,0.72,0.94))))
    ENS=[mkens(0),mkens(1),mkens(-1)]
    cacheP={}
    def getP(j):
        if j not in cacheP:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            cacheP[j]=orth(Vhj[:NF].T)
        return cacheP[j]
    cacheW={}
    def wassets(Wl):
        if Wl not in cacheW:
            Yw=grab(Wl,0,300); mu=Yw.mean(0)
            _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
            cacheW[Wl]=(Vh,(grab(Wl,384,448)-mu).float())
        return cacheW[Wl]
    def loro(Wl, readers, comp0=0, Kc=K):
        readers=tuple(r for r in readers if r!=Wl)
        Vh,Yf=wassets(Wl)
        V=orth(Vh[comp0:comp0+Kc].T)
        yproj=(Yf@V)[:20000]
        fams={}
        for j in readers:
            P=getP(j)
            mlp=m2.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
        pooled=[]; folds={}
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            B=W[:80]
            r2s=[]
            for Mm in fams[jout][:12]:
                ct=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=ct.var().clamp_min(1e-12)
                Mre=((B@Mm.flatten())@B).view(Kc,Kc)
                ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((ch-ct)**2).mean()/vt))
            pooled+=r2s; folds[jout]=sorted(r2s)[len(r2s)//2]
        return sorted(pooled)[len(pooled)//2], folds
    writers=list(range(min(10,NL-2)))
    profiles=[]; foldtabs=[]
    for ei,ens in enumerate(ENS):
        prof={}; ft={}
        for Wl in writers:
            med,folds=loro(Wl,ens)
            prof[Wl]=med; ft[Wl]=folds
        profiles.append(prof); foldtabs.append(ft)
        row=' '.join(f'{v:+.2f}' for v in prof.values())
        print(f'P3 ens{ei} {ens}: {row}',flush=True)
    mins=[min(p,key=p.get) for p in profiles]
    unique_min=len(set(mins))==1
    notch=mins[0]
    rel=[]
    for p in profiles:
        ctrl=sorted(v for k,v in p.items() if k!=notch)[len(p)//2]
        rel.append(p[notch]<=0.5*ctrl)
    frac_ok=0.25<=notch/NL<=0.41
    width_ok=all(all(p.get(nb,1)>p[notch]+0.15
                     for nb in (notch-1,notch+1) if nb in p)
                 for p in profiles)
    tails=[]
    for ens in ENS:
        tm,_=loro(notch,ens,comp0=8,Kc=KT)
        tails.append(tm)
    tail_ok=all(t>=0.35 for t in tails)
    p3_parts={'unique_min':unique_min,'relative_all':all(rel),
              'relative_2of3':sum(rel)>=2,'fraction':frac_ok,
              'width':width_ok,'tail':tail_ok}
    p3=unique_min and sum(rel)>=2 and frac_ok and width_ok and tail_ok
    import statistics
    def spear(d):
        xs=list(d); ys=[d[x] for x in xs]
        rx=sorted(range(len(xs)),key=lambda i:xs[i])
        ry=sorted(range(len(ys)),key=lambda i:ys[i])
        ra=[0]*len(xs); rb=[0]*len(ys)
        for r,i in enumerate(rx): ra[i]=r
        for r,i in enumerate(ry): rb[i]=r
        ma=statistics.mean(ra); mb=statistics.mean(rb)
        return (sum((a-ma)*(b-mb) for a,b in zip(ra,rb))/
                ((sum((a-ma)**2 for a in ra)*
                  sum((b-mb)**2 for b in rb))**0.5))
    sps=[spear(p) for p in profiles]
    p4=all(sp<=-0.3 for sp in sps)
    p5v=[]
    for ens,ft in zip(ENS,foldtabs):
        deepest=max(ens)
        worst=sum(1 for f in ft.values()
                  if deepest in f and min(f,key=f.get)==deepest)
        have=sum(1 for f in ft.values() if deepest in f)
        dm=sorted(f[deepest] for f in ft.values()
                  if deepest in f)[have//2]
        p5v.append((worst,have,dm))
    p5=all(w>h//2 for w,h,_ in p5v) and all(
        (dm<=0.25) if NL>=18 else (dm>0.25) for _,_,dm in p5v)
    out={'model':MODEL,'NL':NL,
         'P1':{'licensed':licensed,'held':bool(p1)},
         'P2':{'improves':improves,'held':bool(p2)},
         'P3':{'notch':notch,'mins':mins,'relative':rel,'tails':tails,
               'parts':{k:bool(v) for k,v in p3_parts.items()},
               'held':bool(p3)},
         'P4':{'spearmans':sps,'held':bool(p4)},
         'P5':{'per_ens':p5v,'held':bool(p5)}}
    print(f"\n=== SCORECARD v2 {MODEL} (NL={NL}) ===")
    print(f"P1: {'HELD' if p1 else 'FAILED'} ({len(licensed)} licensed)")
    print(f"P2: {'HELD' if p2 else 'FAILED'} (replicated improves {improves})")
    print(f"P3: {'HELD' if p3 else 'FAILED'} (notch L{notch}, mins {mins}, "
          f"rel {rel}, tails {[f'{t:+.2f}' for t in tails]}, parts {p3_parts})")
    print(f"P4: {'HELD' if p4 else 'FAILED'} (spearmans "
          f"{[f'{x:+.2f}' for x in sps]})")
    print(f"P5: {'HELD' if p5 else 'FAILED'} ({p5v})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
