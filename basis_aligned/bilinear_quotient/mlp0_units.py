"""MLP0 UNITS -- the right atoms, and the honest benchmark number.
533 verified the exact form -- mlp0 is a signed sum of 9216
squared linear features to 7.8e-7 -- and then chose the wrong
atoms. The squares come in 4608 PAIRS: one hidden unit contributes
(1/4)Down[:,j]((P_j.x)^2 - (M_j.x)^2), and because (a+b)^2 and
(a-b)^2 are both about a^2+b^2, those two terms are large and
nearly equal with the unit's real output in their small
difference. Ranking squares individually splits pairs, which is
why the 32 LOUDEST squares cost 1.69 nats while 32 RANDOM squares
cost 0.81.
The natural atom is the unit. This run redoes the compression
curve over 4608 units with both squares kept together, and adds
the comparison the benchmark actually needs: the SAME measurement
for the per-token table stand-in, on the same rows, so "how many
interpretable atoms reproduce this layer" and "how good is the
lookup table" are finally on one scale. The 1.0-nat figure this
program quotes for the table comes from a different measurement
and cannot be compared to anything here.
Naming is also redone. 533's top-token lists were polluted by rare
embeddings ('ModLoader', '////////'), the standard artifact of
ranking by max |w.E_t| across a whole vocabulary. Here the token
scan is restricted to the 5000 most frequent tokens in the census
corpus, and the class taxonomy is finer -- determiners,
prepositions and pronouns are separated rather than all counted as
"space_word", which 533 showed was hiding two genuinely different
and coherent features.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: keeping all 4608 units reproduces the model
      exactly (cost < 1e-3 nats). Failure means the mean-fill
      machinery is broken and VOIDS the run;
  (a) COMPRESSIBLE: some K <= 256 units (5.6%) holds the cost
      below 0.10 nats;
  (b) BEATS RANDOM: at that K, top-K costs less than a third of
      random-K, three draws. 533's square-level ranking was WORSE
      than random at small K, so this bar is a real test of
      whether the unit is the right atom;
  (c) BEATS THE TABLE: at that same K, the unit stand-in costs
      less than the per-token table stand-in measured on the same
      rows. This is the benchmark claim -- an algebraic stand-in
      with a few hundred atoms beating a fitted lookup table;
  (d) NAMEABLE: among the 5000 most frequent tokens, at least 3 of
      the 5 leading units have >= 7 of their 10 top-scoring tokens
      in one fine class.
  NULL: 5 random directions must reach the naming bar fewer than
      3 times, on the same frequent-token restriction."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=0
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp0_units_results.json'
NFRESH=48
KS=[16,64,256,1024,4608]
DET={'a','an','the','this','that','these','those','his','her',
     'its','their','our','your','my','some','any','no','each',
     'every','another','both','all'}
PREP={'of','to','in','for','on','with','at','by','from','as',
      'into','about','over','after','before','between','through',
      'during','under','against','without','within','onto',
      'toward','towards','upon','among','across','behind','near'}
PRON={'he','she','it','they','we','you','i','him','them','us',
      'me','who','which','what','his','hers','theirs','itself',
      'himself','herself','themselves'}

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
    mlp=m.transformer.h[LJ].mlp
    L=mlp.Left.weight.float(); R=mlp.Right.weight.float()
    Dw=mlp.Down.weight.float(); b=mlp.Down_bias.float()
    H=L.shape[0]
    fresh=cl.fineweb_rows(NFRESH)
    # frequent tokens from the census corpus
    rows=cl.rows()
    cnt=torch.bincount(rows.reshape(-1),
                       minlength=m.transformer.wte.weight.shape[0])
    freq=cnt.argsort(descending=True)[:5000].tolist()
    freqset=torch.tensor(freq,device=DEV)
    # unit importance and mean value, from data
    cap={}; hs=[mlp.register_forward_pre_hook(
        lambda mo_,a_: cap.__setitem__('X',a_[0]))]
    hsum=torch.zeros(H,device=DEV); hsq=torch.zeros(H,device=DEV)
    n=0
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        X=cap['X'].float()
        h=(X@L.T)*(X@R.T)
        hsum+=h.reshape(-1,H).sum(0); hsq+=(h.reshape(-1,H)**2).sum(0)
        n+=h.shape[0]*h.shape[1]
    for hh in hs: hh.remove()
    hmean=hsum/max(n,1)
    hvar=(hsq/max(n,1)-hmean**2).clamp_min(0)
    imp=Dw.norm(dim=0)*hvar.sqrt()      # signed contribution scale
    order=imp.argsort(descending=True)
    print(f'{H} hidden units | top importances '
          f'{[round(float(imp[t]),3) for t in order[:5]]} median '
          f'{float(imp.median()):.3f}',flush=True)

    def run(keep=None,table=False):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if table:
                Etab=F.rms_norm(m.transformer.wte.weight.float(),(D,))
                Ht=(Etab@L.T)*(Etab@R.T)
                Ot=Ht@Dw.T+b                     # (V, D) per-token
                def fh(mo,args,o_,Ot=Ot):
                    return Ot[idx].to(o_.dtype)
                hs.append(mlp.register_forward_hook(fh))
            elif keep is not None:
                msk=torch.zeros(H,device=DEV); msk[keep]=1.0
                def fh(mo,args,o_,msk=msk):
                    X=args[0].float()
                    h=(X@L.T)*(X@R.T)
                    h=h*msk+hmean[None,None,:]*(1-msk)
                    return (h@Dw.T+b).to(o_.dtype)
                hs.append(mlp.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for hh in hs: hh.remove()
        return float(ce.mean())

    base=run(None)
    tab=run(table=True)-base
    print(f'baseline CE {base:.4f} | per-token table stand-in '
          f'{tab:+.4f} nats',flush=True)
    gg=torch.Generator().manual_seed(7)
    curve={}
    for K in KS:
        c=run(order[:K])-base
        rnd=[]
        if K<H:
            for _ in range(3):
                pick=torch.randperm(H,generator=gg)[:K].to(DEV)
                rnd.append(round(run(pick)-base,4))
        curve[K]={'cost':round(c,4),'random':rnd}
        print(f'K={K:>5}: cost {c:+.4f} | random {rnd}',flush=True)
        json.dump(curve,open(OUT,'w'),indent=1)
    p0=abs(curve[H]['cost'])<1e-3
    print(f"(0) all units reproduces the model "
          f"({curve[H]['cost']:+.5f}): "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'curve':curve},
                  open(OUT,'w'),indent=1); return
    # naming over frequent tokens
    Etab=F.rms_norm(m.transformer.wte.weight.float(),(D,))[freqset]
    names=[]; pure=0
    for t in order[:5]:
        j=int(t)
        # the unit's two feature directions
        for tag,w in (('P',L[j]+R[j]),('M',L[j]-R[j])):
            pass
        w=L[j]+R[j]
        sc=(Etab@w).abs()
        top=sc.argsort(descending=True)[:10].tolist()
        toks=[cl.d1(int(freqset[x])) for x in top]
        cls=[fine_class(s) for s in toks]
        best=max(set(cls),key=cls.count); c=cls.count(best)
        if c>=7: pure+=1
        names.append({'unit':j,'tokens':toks,'dominant_class':best,
                      'purity':c})
        print(f"  unit {j}: {best} {c}/10 -> "
              f"{[repr(x) for x in toks[:6]]}",flush=True)
    rpure=0
    for s in range(5):
        g2=torch.Generator(device=DEV).manual_seed(200+s)
        w=torch.randn(D,generator=g2,device=DEV)
        sc=(Etab@w).abs()
        top=sc.argsort(descending=True)[:10].tolist()
        cls=[fine_class(cl.d1(int(freqset[x]))) for x in top]
        if cls.count(max(set(cls),key=cls.count))>=7: rpure+=1
    hit=[K for K in KS if K<=256 and curve[K]['cost']<0.10]
    pa=bool(hit); pb=pc=False
    if hit:
        K=hit[0]; r=min(curve[K]['random']) if curve[K]['random'] else 1
        pb=curve[K]['cost']*3<max(r,1e-9)
        pc=curve[K]['cost']<tab
        print(f'(b) at K={K}: top-K {curve[K]["cost"]:+.4f} vs best '
              f'random {r:+.4f}')
        print(f'(c) at K={K}: units {curve[K]["cost"]:+.4f} vs '
              f'per-token table {tab:+.4f}')
    print(f"(a) some K<=256 costs < 0.10 nats: "
          f"{'HELD ('+str(hit[0])+')' if pa else 'FAILED'}")
    print(f'(d) {pure}/5 leading units class-pure over frequent '
          f'tokens')
    print(f"NULL ({rpure}/5 random directions): "
          f"{'ok' if rpure<3 else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'table_cost':round(tab,4),
         'curve':{str(k):v for k,v in curve.items()},
         'top_units':names,'class_pure':pure,
         'random_class_pure':rpure,'pred_0':True,'pred_a':bool(pa),
         'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pure>=3),'null_ok':bool(rpure<3),
         'first_K_under_0.10':hit[0] if hit else None,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
