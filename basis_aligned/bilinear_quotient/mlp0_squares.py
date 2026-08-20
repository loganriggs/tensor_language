"""MLP0 SQUARES -- the exact interpretable form of an early MLP,
and how far it compresses.
User redirect: the point of finding circuits is a benchmark --
replace model parts, and subparts, with interpretable computation
that produces similar outputs -- and the early layers should be
the easy case because the embedding can be folded exactly and the
weights compose.
What this program HAS done there, so the gap is clear: attention
layer 0 was shown to be exactly a bigram table (0.000 nats); the
first four MLPs were shown to be 68/85/58/44% token-determined,
with mlp1 79% replaceable by a lookup table computed from weights
alone; swapping those MLPs for pure per-token tables costs 1.0
nats at mlp0 and 1.8 at mlp1. All of that FITS a stand-in and
certifies it by cost. None of it DERIVES the layer's structure
from the algebra.
The algebra is available and exact. A bilinear MLP is
    out = Down[(L x) * (R x)] + b
and ab = ((a+b)^2 - (a-b)^2)/4, so with P = L+R and M = L-R
    out = (1/4) Down[(P x)^2] - (1/4) Down[(M x)^2] + b
EXACTLY. The layer is a signed sum of 9216 SQUARED LINEAR
FEATURES of its input, with no approximation anywhere. Each
feature is one direction w in input space and contributes
+-(1/4) Down[:,j] (w.x)^2.
Folding the embedding makes those features nameable. Block 0
computes x = (lam0+lam1) E, then adds attn0, then rms-normalizes,
and rms_norm(c E) = E for c > 0 -- so if attn0 were silent the
MLP input would be EXACTLY the token's embedding and every feature
value would be a per-vocabulary number (w.E_t)^2. attn0 is not
silent, but it is a bigram table, so mlp0's input is an exact
function of the (previous, current) token pair and its features
are a table over bigrams.
This run measures three things:
  EXACTNESS of the rewriting;
  COMPRESSION -- keep the top K of 9216 squares and mean-fill the
    rest, then price the whole model. This is the benchmark
    quantity: how few interpretable atoms reproduce the layer;
  NAMES -- for the leading features, which tokens maximize
    |w.E_t|, and whether those token sets are recognizable
    classes rather than arbitrary.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: the 9216-square form reproduces mlp0's real
      output to 1e-5 relative. It is algebra, so failure means an
      implementation error and VOIDS the run;
  (a) COMPRESSIBLE: some K <= 512 (5.6% of the squares) holds the
      whole-model cross-entropy cost below 0.10 nats. For scale,
      replacing mlp0 by a per-token table costs 1.0 nats;
  (b) BEATS RANDOM: at that K, the top-K stand-in costs less than
      a fifth of what a random-K stand-in costs, three draws;
  (c) NAMEABLE: for the top 5 features, at least 3 have >= 7 of
      their 10 highest-|w.E_t| tokens inside a single automatic
      class (digits, punctuation, capitalized, space-leading,
      or subword continuation);
  NULL: the same class-purity statistic for 5 random directions in
      input space must name fewer than 3 of 5. If random
      directions look just as class-pure, (c) means nothing."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=0
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp0_squares_results.json'
NFRESH=48
KS=[32,128,512,2048,9216]

def token_class(s):
    t=s.strip()
    if not t: return 'space'
    if t[0].isdigit(): return 'digit'
    if all(not c.isalnum() for c in t): return 'punct'
    if t[0].isupper(): return 'capitalized'
    if s.startswith(' '): return 'space_word'
    return 'subword'

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mlp=m.transformer.h[LJ].mlp
    L=mlp.Left.weight.float(); R=mlp.Right.weight.float()
    Dw=mlp.Down.weight.float(); b=mlp.Down_bias.float()
    H=L.shape[0]
    Pw=(L+R); Mw=(L-R)                     # (H, D) each
    W=torch.cat([Pw,Mw],0)                 # (2H, D)  the features
    sign=torch.cat([torch.ones(H,device=DEV),
                    -torch.ones(H,device=DEV)])
    Dcat=torch.cat([Dw,Dw],1)*0.25         # (D, 2H)
    Dsig=Dcat*sign[None,:]
    NF=W.shape[0]
    print(f'mlp0: {H} hidden units -> {NF} squared features',
          flush=True)
    fresh=cl.fineweb_rows(NFRESH)

    def feats(X):
        return (X.float()@W.T)**2          # (B,T,2H)

    # exactness + feature second moments
    cap={}; sq_mean=torch.zeros(NF,device=DEV); n=0; errs=[]
    hs=[mlp.register_forward_pre_hook(
        lambda mo_,a_: cap.__setitem__('X',a_[0]))]
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        X=cap['X']
        f=feats(X)
        recon=f@Dsig.T+b
        real=mlp(X).float()
        errs.append(float((recon-real).norm()/real.norm()))
        sq_mean+=f.reshape(-1,NF).sum(0); n+=f.shape[0]*f.shape[1]
    for h in hs: h.remove()
    sq_mean/=max(n,1)
    ex=max(errs)
    print(f'(0) square-form reconstruction {ex:.3e}',flush=True)
    p0=ex<=1e-5
    print(f"(0) EXACTNESS: {'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'exactness':ex},
                  open(OUT,'w'),indent=1); return
    # importance: |Down column| * mean square value
    imp=(Dsig.norm(dim=0)*sq_mean)
    order=imp.argsort(descending=True)
    print(f'top feature importances: '
          f'{[round(float(imp[t]),4) for t in order[:5]]} | median '
          f'{float(imp.median()):.4g}',flush=True)

    def run(keep=None):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if keep is not None:
                msk=torch.zeros(NF,device=DEV); msk[keep]=1.0
                def fh(mo,args,o_,msk=msk):
                    X=args[0]
                    f=feats(X)
                    f=f*msk+sq_mean[None,None,:]*(1-msk)
                    return (f@Dsig.T+b).to(o_.dtype)
                hs.append(mlp.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=run(None)
    print(f'baseline CE {base:.4f}',flush=True)
    gg=torch.Generator().manual_seed(7)
    curve={}
    for K in KS:
        c=run(order[:K])-base
        rnd=[]
        if K<NF:
            for _ in range(3):
                pick=torch.randperm(NF,generator=gg)[:K].to(DEV)
                rnd.append(round(run(pick)-base,4))
        curve[K]={'cost':round(c,4),'random':rnd}
        print(f'K={K:>5}: cost {c:+.4f} nats | random {rnd}',
              flush=True)
        json.dump(curve,open(OUT,'w'),indent=1)
    # names: tokens maximizing |w . E_t| for the leading features
    Etab=F.rms_norm(m.transformer.wte.weight.float(),(D,))
    names=[]; pure=0
    for t in order[:5]:
        w=W[int(t)]
        sc=(Etab@w).abs()
        top=sc.argsort(descending=True)[:10].tolist()
        toks=[cl.d1(int(x)) for x in top]
        cls=[token_class(s) for s in toks]
        best=max(set(cls),key=cls.count)
        cnt=cls.count(best)
        if cnt>=7: pure+=1
        names.append({'feature':int(t),'tokens':toks,
                      'dominant_class':best,'purity':cnt})
        print(f"  feature {int(t)}: {best} {cnt}/10 -> "
              f"{[repr(x) for x in toks[:6]]}",flush=True)
    rpure=0; rnames=[]
    for s in range(5):
        g2=torch.Generator(device=DEV).manual_seed(100+s)
        w=torch.randn(D,generator=g2,device=DEV)
        sc=(Etab@w).abs()
        top=sc.argsort(descending=True)[:10].tolist()
        cls=[token_class(cl.d1(int(x))) for x in top]
        best=max(set(cls),key=cls.count)
        if cls.count(best)>=7: rpure+=1
        rnames.append({'dominant_class':best,
                       'purity':cls.count(best)})
    hit=[K for K in KS if K<=512 and curve[K]['cost']<0.10]
    pa=bool(hit)
    pb=False
    if hit:
        K=hit[0]; r=min(curve[K]['random']) if curve[K]['random'] else 1
        pb=curve[K]['cost']*5<max(r,1e-9)
        print(f'(b) at K={K}: top-K {curve[K]["cost"]:+.4f} vs best '
              f'random {r:+.4f}')
    print(f"(a) some K<=512 costs < 0.10 nats: "
          f"{'HELD ('+str(hit[0])+')' if pa else 'FAILED'}")
    print(f'(c) {pure}/5 leading features are class-pure (>=7/10)')
    print(f"NULL ({rpure}/5 random directions are class-pure): "
          f"{'ok' if rpure<3 else 'VIOLATED'}")
    out={'exactness':ex,'n_features':NF,'baseline_ce':round(base,4),
         'curve':{str(k):v for k,v in curve.items()},
         'top_features':names,'random_features':rnames,
         'class_pure':pure,'random_class_pure':rpure,
         'pred_0':True,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pure>=3),'null_ok':bool(rpure<3),
         'first_K_under_0.10':hit[0] if hit else None,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
