"""FRONT TABLE -- build the stand-in and price it.
The benchmark question is whether a part of the model can be
replaced by interpretable computation. For block 0 the pieces are
now in place: its input is exactly the token embedding (535), it
compares tokens along 2 directions per head (535), its MLP writes
into about 64 directions (536), and the whole block's write is
carried by 64 principal directions to within 0.081 nats (539).
None of that is a stand-in. This run builds one and measures it.
The candidate object is a TABLE: one vector per vocabulary entry,
holding the write that block 0 emits for that token, optionally
projected to r dimensions. Substituting it means block 0 stops
attending to anything -- every position gets the write its own
token deserves, looked up. If that works, the first block of this
model IS a 50304 x r table and can be printed.
Four arms, all replacing block 0's combined write:
  true_r      the real write projected to its top r principal
              directions (539's measurement, recomputed here as
              the reference ceiling)
  table_r     the per-token mean write, projected to r dimensions
              -- the stand-in
  table_full  the per-token mean write at full rank, which is the
              classic lookup-table stand-in
  random_r    a random r-dimensional projection of the per-token
              table, as the control
The gap between true_r and table_r is exactly how much context
block 0 contributes beyond the current token, measured in nats.
REGISTERED PREDICTIONS:
  (0) SANITY: table_full costs less than deleting the write
      entirely (0.838 nats, from 539). A table that is worse than
      nothing is not a stand-in and VOIDS the run;
  (a) THE STAND-IN WORKS: table_64 costs under 0.30 nats;
  (b) CONTEXT IS SMALL: the gap between table_64 and true_64 is
      under 0.25 nats, i.e. what block 0 does with context beyond
      the current token is worth less than a quarter nat. If the
      gap is large, block 0 is not a table however narrow its
      interface, and that is the honest answer;
  (c) PROJECTION IS NEARLY FREE ON THE TABLE: table_64 costs at
      most 0.10 nats more than table_full, so the 64-dimensional
      restriction is not what breaks it;
  NULL: random_64 must cost at least three times table_64. If a
      random 64-dimensional table does as well, the principal
      directions are not the content.
Naming is reported alongside: for the top eight interface
directions, the ten highest-scoring FREQUENT tokens and their
dominant class, since each direction is literally a scoring
function over the vocabulary once the table exists."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'front_table_results.json'
NFRESH=48; RANKS=[16,64,256]
DET={'a','an','the','this','that','these','those','his','her',
     'its','their','our','your','my','some','any','no','each'}
PREP={'of','to','in','for','on','with','at','by','from','as',
      'into','about','over','after','before','between','through'}
PRON={'he','she','it','they','we','you','i','him','them','us','me'}

def fine_class(s):
    t=s.strip().lower()
    if not t: return 'space'
    if t in DET: return 'determiner'
    if t in PREP: return 'preposition'
    if t in PRON: return 'pronoun'
    if t[0].isdigit(): return 'digit'
    if all(not c.isalnum() for c in t): return 'punct'
    if s.strip()[:1].isupper(): return 'capitalized'
    if s.startswith(' '): return 'space_word'
    return 'subword'

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    V=m.transformer.wte.weight.shape[0]
    at0=m.transformer.h[0].attn; mlp0=m.transformer.h[0].mlp
    rows=cl.rows()
    # per-token mean write, accumulated over the census corpus
    tot=torch.zeros(V,D,device=DEV); cnt=torch.zeros(V,device=DEV)
    store={}
    ha=at0.register_forward_hook(
        lambda mo,i_,o_: store.__setitem__(
            'a',(o_[0] if isinstance(o_,tuple) else o_).detach().float()))
    acc=[]
    def fm(mo,i_,o_):
        acc.append((store['a']+o_.float()).reshape(-1,D))
        return o_
    hm=mlp0.register_forward_hook(fm)
    for i in range(0,rows.shape[0],4):
        bb=rows[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        acc.clear()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        W=acc[0]
        flat=idx.reshape(-1)
        tot.index_add_(0,flat,W); cnt.index_add_(
            0,flat,torch.ones_like(flat,dtype=torch.float))
    ha.remove(); hm.remove()
    seen=int((cnt>0).sum())
    table=tot/cnt.clamp_min(1).unsqueeze(1)
    print(f'{seen} of {V} vocabulary entries seen in the corpus',
          flush=True)
    # interface basis from fresh text
    fresh=cl.fineweb_rows(NFRESH)
    caps=[]
    ha=at0.register_forward_hook(
        lambda mo,i_,o_: store.__setitem__(
            'a',(o_[0] if isinstance(o_,tuple) else o_).detach().float()))
    hm=mlp0.register_forward_hook(
        lambda mo,i_,o_: (caps.append((store['a']+o_.float())
                          .reshape(-1,D).cpu()),o_)[1])
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    ha.remove(); hm.remove()
    Y=torch.cat(caps).to(DEV); mu=Y.mean(0)
    _,_,Vh=torch.linalg.svd(Y-mu,full_matrices=False)

    def price(mode=None,r=None,seed=0):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if mode is not None:
                if mode=='random':
                    g=torch.Generator(device=DEV).manual_seed(seed)
                    Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                                    device=DEV))
                    P=Q@Q.T
                elif r is not None:
                    Vr=Vh[:r]; P=Vr.T@Vr
                else:
                    P=None
                st={}
                hs.append(at0.register_forward_hook(
                    lambda mo,i_,o_,st=st: (st.__setitem__(
                        'a',(o_[0] if isinstance(o_,tuple) else o_)
                        .detach().float()),o_)[1]))
                def fh(mo,i_,o_,P=P,mode=mode,st=st,idx=idx):
                    a=st['a']
                    if mode=='true':
                        w=a+o_.float()
                    else:
                        w=table[idx]
                    new=mu+(w-mu)@P if P is not None else w
                    return (new-a).to(o_.dtype)
                hs.append(mlp0.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=price()
    print(f'baseline CE {base:.4f}',flush=True)
    res={'table_full':round(price('table')-base,4)}
    print(f"table_full: {res['table_full']:+.4f}",flush=True)
    for r in RANKS:
        res[f'true_{r}']=round(price('true',r)-base,4)
        res[f'table_{r}']=round(price('table',r)-base,4)
        print(f"r={r:>4}: true {res[f'true_{r}']:+.4f} | table "
              f"{res[f'table_{r}']:+.4f} | context gap "
              f"{res[f'table_{r}']-res[f'true_{r}']:+.4f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    rnd=[round(price('random',64,s)-base,4) for s in (1,2,3)]
    res['random_64']=rnd
    print(f'random_64: {rnd}',flush=True)
    p0=res['table_full']<0.838
    va=res['table_64']<0.30
    gap=res['table_64']-res['true_64']
    vb=gap<0.25
    vc=(res['table_64']-res['table_full'])<=0.10
    nul=min(rnd)>=3*max(res['table_64'],1e-6)
    print(f"\n(0) table_full {res['table_full']:+.4f} < 0.838 "
          f"(deleting the write): {'HELD' if p0 else 'FAILED -- VOID'}")
    print(f"(a) table_64 {res['table_64']:+.4f} < 0.30: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) context gap at r=64: {gap:+.4f} < 0.25: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"(c) table_64 - table_full = "
          f"{res['table_64']-res['table_full']:+.4f} <= 0.10: "
          f"{'HELD' if vc else 'FAILED'}")
    print(f"NULL (random_64 {min(rnd):+.4f} >= 3x table_64): "
          f"{'ok' if nul else 'VIOLATED'}")
    # naming the interface directions
    cntv=torch.bincount(rows.reshape(-1),minlength=V)
    freq=cntv.argsort(descending=True)[:5000]
    Tab=table[freq]
    names=[]; pure=0
    for k in range(8):
        sc=(Tab-mu)@Vh[k]
        top=sc.abs().argsort(descending=True)[:10].tolist()
        toks=[cl.d1(int(freq[t])) for t in top]
        cls=[fine_class(s) for s in toks]
        best=max(set(cls),key=cls.count); c=cls.count(best)
        if c>=7: pure+=1
        names.append({'dir':k,'dominant_class':best,'purity':c,
                      'tokens':toks})
        print(f'  interface dir {k}: {best} {c}/10 -> '
              f'{[repr(x) for x in toks[:6]]}',flush=True)
    print(f'{pure}/8 interface directions are class-pure')
    out={'baseline_ce':round(base,4),'results':res,
         'context_gap_64':round(gap,4),'vocab_seen':seen,
         'interface_names':names,'class_pure':pure,
         'pred_0':bool(p0),'pred_a':bool(va),'pred_b':bool(vb),
         'pred_c':bool(vc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
