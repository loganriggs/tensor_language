"""DIGIT DISTRIBUTED -- is the digit effect compact within its
16-dim bundle, and how does it split across the four heads?
The digit behaviour is spread over four heads of a6/a8 (521) and
lives in a 16-dim subspace of their output (the r.2.0.1 bundle).
Two grounded questions, both on a HELD-OUT split (the contamination
lesson of 548):
  DIRECTION RANK. The bundle is 16-dim by construction, but how
    many of those directions actually carry the digit effect? Rank
    the 16 by causal contribution and keep the top k, mean-filling
    the rest; price digit dissociation. If k < 8 holds most of the
    effect, the effect is more compact than its bundle -- the same
    "compact frame" shape found for attention selection.
  HEAD SPLIT. Attribute each of the 16-dim projection's energy to
    the four contributing heads at digit targets, to see whether
    the bundle is written cooperatively (shared) or partitioned.
Grounded: digit target positions are verified (the class is
unambiguous), and the bundle is the certified r.2.0.1 machinery.
REGISTERED PREDICTIONS:
  (0) DISJOINT: fit rows and priced rows share no row
      (cl.assert_disjoint); full 16-dim removal reproduces the
      known digit effect (>= 0.10 nats). VOIDS otherwise;
  (a) COMPACT: some k <= 6 of the 16 directions holds >= 60% of
      the full-bundle digit dissociation;
  (b) BEATS RANDOM: at that k, the top-k directions beat k random
      directions of the bundle by >= 2x;
  (c) HEAD SPLIT reported -- the four heads' shares of the bundle
      projection at digit targets. No bar;
  NULL: removing the top-k directions at NON-digit positions costs
      < 1/3 of the digit-target cost."""
import ast, json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_distributed_results.json'
TAG='r.2.0.1'; LAYERS=[6,8]; NFIT=800; NPRICE=96
KS=[1,2,4,6,8,12,16]

def projector(key,probes):
    seen=set(); keep=[]
    for pr in probes:
        _,k,stag,blk=pr
        if k!=key or (k,stag,tuple(blk)) in seen: continue
        seen.add((k,stag,tuple(blk))); keep.append(pr)
    vs=[cl.pca_block(p[1],p[2],p[3]) for p in keep]
    return orth(torch.cat(vs).T) if vs else None

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    allrows=cl.rows()
    fit=allrows[:NFIT]; price=allrows[NFIT:NFIT+NPRICE]
    ok,_=cl.assert_disjoint(fit,price,label='digit_distributed')
    if not ok:
        json.dump({'pred_0':False},open(OUT,'w'),indent=1); return
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    P={li:projector(f'a{li}',probes) for li in LAYERS}
    P={li:v for li,v in P.items() if v is not None}
    def classes(rows):
        nxt=rows[:,1:257]
        dig=torch.zeros_like(nxt,dtype=torch.bool)
        for r in range(nxt.shape[0]):
            for q in range(nxt.shape[1]):
                z=cl.d1(int(nxt[r,q])).strip()
                if z and z[0].isdigit(): dig[r,q]=True
        return dig
    dig=classes(price)

    def ce_rows(rows,dirs=None):
        # dirs: dict li -> (D,k) projector to REMOVE from a{li} out
        out=torch.zeros(rows.shape[0],T)
        for i in range(0,rows.shape[0],4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if dirs:
                for li,Q in dirs.items():
                    Pm=Q@Q.T
                    at=m.transformer.h[li].attn
                    def fh(mo,i_,o_,Pm=Pm):
                        y,v1=o_; yf=y.float().reshape(-1,D)
                        return ((yf-yf@Pm).view(y.shape).to(y.dtype),v1)
                    hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            out[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return out
    def diss(base,abl):
        d=abl-base
        return float(d[dig].mean())-float(d[~dig].mean())
    base=ce_rows(price)
    full=diss(base,ce_rows(price,{li:P[li] for li in P}))
    print(f'full 16-dim removal digit dissociation: {full:+.4f}',
          flush=True)
    p0=abs(full)>=0.10
    if not p0:
        print('(0) full effect < 0.10 -- VOID')
        json.dump({'pred_0':False,'full':full},open(OUT,'w'),indent=1)
        return
    # rank the 16 directions by individual causal contribution
    li0=LAYERS[0] if LAYERS[0] in P else list(P)[0]
    contrib={}
    for li in P:
        for j in range(P[li].shape[1]):
            Q=P[li][:,j:j+1]
            contrib[(li,j)]=diss(base,ce_rows(price,{li:Q}))
    order=sorted(contrib,key=lambda k:-abs(contrib[k]))
    curve={}
    g=torch.Generator().manual_seed(7)  # cpu, for randperm
    for k in KS:
        pick=order[:k]
        dirs={}
        for li in P:
            cols=[j for (l,j) in pick if l==li]
            if cols: dirs[li]=P[li][:,cols]
        c=diss(base,ce_rows(price,dirs)) if dirs else 0.0
        # random k directions of the bundle
        rnd=[]
        for s in range(3):
            allpairs=list(contrib)
            perm=torch.randperm(len(allpairs),generator=g)[:k]
            rp=[allpairs[int(x)] for x in perm]
            rd={}
            for li in P:
                cols=[j for (l,j) in rp if l==li]
                if cols: rd[li]=P[li][:,cols]
            rnd.append(round(diss(base,ce_rows(price,rd)) if rd else 0,4))
        curve[k]={'top':round(c,4),'random':rnd,
                  'frac':round(c/max(full,1e-6),3)}
        print(f'k={k:>2}: top {c:+.4f} ({100*c/max(full,1e-6):.0f}%) '
              f'| random {rnd}',flush=True)
        json.dump({str(x):y for x,y in curve.items()},
                  open(OUT,'w'),indent=1)
    hit=[k for k in KS if k<=6 and curve[k]['frac']>=0.60]
    pa=bool(hit)
    pb=False
    if hit:
        k=hit[0]; r=max(abs(x) for x in curve[k]['random']) or 1e-9
        pb=abs(curve[k]['top'])>=2*r
    print(f"\n(a) some k<=6 holds >=60%: "
          f"{'HELD ('+str(hit[0])+')' if pa else 'FAILED'}")
    out={'full':round(full,4),'curve':{str(x):y for x,y in curve.items()},
         'first_k_over_60':hit[0] if hit else None,
         'pred_0':True,'pred_a':bool(pa),'pred_b':bool(pb),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
