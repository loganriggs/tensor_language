"""DIRECTION NAMES 2 -- readability with a matched control and a
base rate.
551 found the nameability metric broken: 9 of 16 RANDOM directions
passed the "7 of 10 top tokens share a class" test, because the
random control was drawn in the wrong space (embeddings, not the
centred per-token write) and the criterion ignored class base
rates among frequent tokens. Both are fixed here.
  MATCHED CONTROL. The random directions are drawn in the SAME
    centred per-token write space as the real principal directions
    -- a random unit vector in that space, scored by the same
    projection of the same matrix -- so the only thing removed is
    the principal-direction structure.
  BASE-RATE CRITERION. A direction is named only if its dominant
    class among the top 10 tokens is ENRICHED against that class's
    frequency in the 5000-token vocabulary at a binomial tail
    probability below 0.01. A class that is 40% of the vocabulary
    needs far more than 7 of 10 to count; a rare class needs
    fewer.
REGISTERED PREDICTIONS:
  (0) DISJOINT bases, via cl.assert_disjoint. Overlap VOIDS;
  (a) REAL BEATS RANDOM: the six blocks average at least 3x as
      many named directions as the matched random control. This
      is the bar 551 failed and the precondition for any count to
      mean something;
  (b) BLOCK 0 IS READABLE: at least 5 of its top 16 directions are
      named under the strict criterion;
  (c) THE CURVE with depth, reported with the random baseline
      subtracted. No bar -- the shape is the result;
  NULL: the matched random control yields fewer than 2 named of 16
      averaged over the blocks. If random directions in the write
      space are named, the criterion is still too loose."""
import json, time, math, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NB=6; NDIR=16
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'direction_names2_results.json'
NFIT=800; NPRICE=96
DET={'a','an','the','this','that','these','those','his','her',
     'its','their','our','your','my','some','any','no','each',
     'every','another','both','all'}
PREP={'of','to','in','for','on','with','at','by','from','as',
      'into','about','over','after','before','between','through',
      'during','under','against','without','within','onto'}
PRON={'he','she','it','they','we','you','i','him','them','us',
      'me','who','which','what','hers','theirs','itself'}

def binom_tail(k,n,p):
    # P(X >= k) for X ~ Binomial(n,p)
    return sum(math.comb(n,i)*p**i*(1-p)**(n-i)
               for i in range(k,n+1))

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
    allrows=cl.rows()
    fit=allrows[:NFIT]; price_rows=allrows[NFIT:NFIT+NPRICE]
    ok,ns=cl.assert_disjoint(fit,price_rows,label='direction_names')
    if not ok:
        json.dump({'pred_0':False,'shared':ns},
                  open(OUT,'w'),indent=1); return
    cnt=torch.bincount(fit.reshape(-1),minlength=V)
    freq=cnt.argsort(descending=True)[:5000]
    freq_cls=[fine_class(cl.d1(int(t))) for t in freq.tolist()]
    from collections import Counter
    baserate={c:n/len(freq_cls) for c,n in Counter(freq_cls).items()}
    print('class base rates among frequent tokens:',
          {c:round(r,3) for c,r in sorted(baserate.items(),
           key=lambda x:-x[1])},flush=True)
    def named_by(cls):
        best=max(set(cls),key=cls.count); c=cls.count(best)
        p=baserate.get(best,0.01)
        return (best,c,binom_tail(c,10,p)<0.01)
    st={}
    res={}; examples={}
    for b in range(NB):
        sink=[]
        at=m.transformer.h[b].attn; mlp=m.transformer.h[b].mlp
        h1=at.register_forward_hook(
            lambda mo,i_,o_,b=b: st.__setitem__(
                b,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))
        h2=mlp.register_forward_hook(
            lambda mo,i_,o_,b=b: (sink.append(
                (st[b]+o_.float()).reshape(-1,D)),o_)[1])
        tok=torch.zeros(V,D,device=DEV); tc=torch.zeros(V,device=DEV)
        for i in range(0,NFIT,4):
            bb=fit[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            sink.clear()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            W=sink[0]; flat=idx.reshape(-1)
            tok.index_add_(0,flat,W)
            tc.index_add_(0,flat,torch.ones_like(flat,
                                                 dtype=torch.float))
        h1.remove(); h2.remove()
        tokmean=tok/tc.clamp_min(1).unsqueeze(1)
        mu=(tok.sum(0)/tc.sum())
        # basis from the per-token means, which is what a
        # token-indexed reading of the interface would use
        Yc=tokmean[freq]-mu
        _,_,Vh=torch.linalg.svd(Yc,full_matrices=False)
        named=[]; pure=0
        for k in range(NDIR):
            sc=Yc@Vh[k]
            top=sc.abs().argsort(descending=True)[:10].tolist()
            toks=[cl.d1(int(freq[t])) for t in top]
            cls=[fine_class(s) for s in toks]
            best,c,isnamed=named_by(cls)
            if isnamed: pure+=1
            named.append({'dir':k,'class':best,'purity':c,
                          'named':isnamed,'tokens':toks[:6]})
        res[b]={'nameable':pure,'directions':named}
        print(f'block {b}: {pure} of {NDIR} directions nameable',
              flush=True)
        for d in named[:4]:
            print(f"   dir {d['dir']}: {d['class']} {d['purity']}/10 "
                  f"-> {[repr(x) for x in d['tokens'][:5]]}",
                  flush=True)
        # matched control: random directions IN the write space
        gg=torch.Generator(device=DEV).manual_seed(100+b)
        rp=0
        for _ in range(NDIR):
            w=torch.randn(D,generator=gg,device=DEV)
            w=w/w.norm()
            sc=Yc@w
            top=sc.abs().argsort(descending=True)[:10].tolist()
            cls=[fine_class(cl.d1(int(freq[t]))) for t in top]
            if named_by(cls)[2]: rp+=1
        res[b]['random_named']=rp
        print(f'   matched random control: {rp} of {NDIR} named',
              flush=True)
        json.dump({str(k):v for k,v in res.items()},
                  open(OUT,'w'),indent=1)
        del tok,tokmean,Yc
    counts=[res[b]['nameable'] for b in range(NB)]
    rand=[res[b]['random_named'] for b in range(NB)]
    real_avg=sum(counts)/NB; rand_avg=sum(rand)/NB
    va=real_avg>=3*max(rand_avg,1e-6)
    vb=counts[0]>=5
    nul=rand_avg<2
    net=[counts[b]-rand[b] for b in range(NB)]
    print(f"\n(a) real {real_avg:.1f} >= 3x random {rand_avg:.1f} "
          f"avg named: {'HELD' if va else 'FAILED'}")
    print(f"(b) block 0 has {counts[0]} named (strict) >= 5: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"(c) named by block {counts} | random {rand} | net {net}")
    print(f"NULL (random avg {rand_avg:.1f} < 2): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'named_by_block':counts,'random_by_block':rand,
         'net_by_block':net,'real_avg':round(real_avg,2),
         'random_avg':round(rand_avg,2),
         'blocks':{str(k):v for k,v in res.items()},
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
