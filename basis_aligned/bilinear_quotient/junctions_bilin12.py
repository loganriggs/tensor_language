"""JUNCTIONS IN BILIN12 -- cross-model test of the junction anatomy
(295-299). bilin18 has exactly three critical attn->own-mlp handoffs:
front coherence junctions (blocks 0,1) and a mid-depth narrow regulator
(block 5 = fraction 0.28) that governs the private span -- and the
private writer itself was already shown UNIVERSAL at fraction ~0.33
(bilin12 L4, section 215). If the regulator mechanism is a family
trait, bilin12 should show: a front junction, a mid junction near the
private-writer depth, and a free tail. Method: mean-handoff cut
(mlp sees rms(x_mixed) instead of rms(x_mixed + attn_out)) per block,
all 12; then the rank ladder (top-4/top-16 vs full) on the largest
MID-DEPTH junction (blocks 2-7).
REGISTERED PREDICTIONS:
  (a) FRONT: block 0 or 1 costs >= +0.5;
  (b) MID: some block in 2-7 costs >= 3x the median of blocks 2-11;
  (c) TAIL: blocks 8-11 all <= 0.1;
  (d) informational: is the mid junction at L4 (the private writer)?
  (e) SHAPE: on the top mid junction, either the narrow signature
      (top-4 >= 60% of full) or the coherence signature (top-4 >= 1.5x
      full) appears -- the b18 typology transports; a flat partial
      ladder (top-16 <= 20%) would refute transport."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'junctions_bilin12_results.json'
R0,R1=120,300; CA=300

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('bilin12', device=DEV)
    NL=len(m2.transformer.h); D=m2.transformer.wte.weight.shape[1]
    print(f'bilin12: {NL} layers, d={D}',flush=True)
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
    print(f'bilin12 base CE {base:.4f}',flush=True)
    costs=[]
    for li in range(NL):
        c=evalCE(mk_cut(li))-base
        costs.append(c)
        print(f'block {li:2d}: {c:+.4f}',flush=True)
    midtail=sorted(costs[2:])
    med=midtail[len(midtail)//2]
    mid_j=max(range(2,8),key=lambda i:costs[i])
    pa=max(costs[0],costs[1])>=0.5
    pb=costs[mid_j]>=3*max(med,1e-3)
    pc=all(costs[i]<=0.1 for i in range(8,NL))
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
    pe=narrow or coher
    out={'base':round(base,4),'costs':[round(c,4) for c in costs],
         'mid_junction':mid_j,'mid_at_private_writer':bool(mid_j==4),
         'ladder':{'full':round(full,4),'top4':round(t4,4),
                   'top16':round(t16,4)},
         'shape':('narrow' if narrow else 'coherence' if coher
                  else 'flat' if flat else 'intermediate'),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_e':bool(pe)}
    print(f"(a) front junction >= +0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) mid junction >= 3x median: {'HELD' if pb else 'FAILED'}")
    print(f"(c) tail free: {'HELD' if pc else 'FAILED'}")
    print(f"(d) mid at private writer L4: {mid_j==4}")
    print(f"(e) typology transports ({out['shape']}): "
          f"{'HELD' if pe else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
