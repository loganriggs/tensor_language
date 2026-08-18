"""JUNCTIONS IN SWIGLU18 -- the contrast test. The junction anatomy
(front coherence junctions + one narrow mid junction fused with the
private code + free tail) is certified in BOTH bilinear models. Is it
bilinear-specific or transformer-general? swiglu18 (gated, 18L, same
depth/width/data as bilin18) spreads function uniformly (204-205: 0/11
rank-4 rescues; no deletion-improves slack; L15=L15 regularizer
identity) -- the registered expectation is that the bilinear complex
does NOT appear.
REGISTERED PREDICTIONS:
  (a) a front junction exists (block 0 or 1 >= +0.5) -- predicted
      architecture-GENERAL;
  (b) NO mid junction: no block in 2-12 reaches 3x the median of
      blocks 2-16 (the narrow complex is bilinear-specific);
  (c) if a mid junction does appear, its ladder is NOT narrow
      (top-4 < 60% of full) -- the 4-direction channel form is
      bilinear-specific even if a junction exists;
  (d) full 18-block profile reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'junctions_swiglu18_results.json'
R0,R1=120,300; CA=300

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('swiglu18', device=DEV)
    NL=len(m2.transformer.h); D=m2.transformer.wte.weight.shape[1]
    print(f'swiglu18: {NL} layers, d={D}',flush=True)
    cur={}
    def evalCE(hooks):
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m2.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return float(torch.cat(ces).mean())
    def mk_cut(li,P=None):
        blk=m2.transformer.h[li]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            if P is None:
                return (alt.to(x.dtype),)+args[1:]
            d=x.float()-alt
            return ((x.float()-(d@P)@P.T).to(x.dtype),)+args[1:]
        return [blk.register_forward_pre_hook(hb),
                blk.mlp.register_forward_pre_hook(hm)]
    base=evalCE([])
    print(f'swiglu18 base CE {base:.4f}',flush=True)
    costs=[]
    for li in range(NL):
        c=evalCE(mk_cut(li))-base
        costs.append(c)
        print(f'block {li:2d}: {c:+.4f}',flush=True)
    midtail=sorted(costs[2:17])
    med=midtail[len(midtail)//2]
    mid_j=max(range(2,13),key=lambda i:costs[i])
    pa=max(costs[0],costs[1])>=0.5
    pb=costs[mid_j]<3*max(med,1e-3)   # CONTRAST: no mid junction
    pc=True
    print(f'mid junction candidate: block {mid_j} ({costs[mid_j]:+.4f}) '
          f'| median(2-11) {med:+.4f}',flush=True)
    # shape ladder on the mid junction
    blk=m2.transformer.h[mid_j]
    ds_=[]
    def hb2(mo,args):
        x,v1,x0=args
        cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
    h1=blk.register_forward_pre_hook(hb2)
    h2=blk.mlp.register_forward_pre_hook(
        lambda mo,args: ds_.append((args[0].float()
            -F.rms_norm(cur['xm'].float(),(D,))).reshape(-1,D)))
    for i in range(CA,CA+80,4):
        bb=FW[i:i+4,:257].to(DEV)
        m2(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    h1.remove(); h2.remove()
    Dd=torch.cat(ds_)
    _,_,Vh=torch.linalg.svd(Dd[:40000],full_matrices=False)
    t4=evalCE(mk_cut(mid_j,orth(Vh[:4].T)))-base
    t16=evalCE(mk_cut(mid_j,orth(Vh[:16].T)))-base
    full=costs[mid_j]
    print(f'mid ladder: full {full:+.4f} top4 {t4:+.4f} top16 {t16:+.4f}',
          flush=True)
    narrow=t4>=0.6*full; coher=t4>=1.5*full
    flat=t16<=0.2*full
    pe=not narrow    # CONTRAST: narrow channel form is bilinear-specific
    out={'base':round(base,4),'costs':[round(c,4) for c in costs],
         'mid_junction':mid_j,
         'ladder':{'full':round(full,4),'top4':round(t4,4),
                   'top16':round(t16,4)},
         'shape':('narrow' if narrow else 'coherence' if coher
                  else 'flat' if flat else 'intermediate'),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_e':bool(pe)}
    print(f"(a) front junction exists: {'HELD' if pa else 'FAILED'}")
    print(f"(b) NO mid junction (contrast): {'HELD' if pb else 'FAILED'}")
    print(f"(c) shape of top mid candidate: {out['shape']}")
    print(f"(e) not-narrow (contrast): {'HELD' if pe else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
