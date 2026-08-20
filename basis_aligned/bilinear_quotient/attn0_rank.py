"""ATTN0 RANK -- how many directions does the first attention
layer actually read?
The folding is exactly clean at layer 0 and worth stating as a
fact before measuring anything. Block 0 computes x = lam0*x0 +
lam1*x0 = 12.1875 E with E = rms_norm(wte[t]), and rms_norm is
scale invariant, so attention layer 0's input is EXACTLY the
token's own embedding. No context, no mixing, no approximation.
Everything attn0 computes is therefore a function of token
identities and relative position alone -- which is why 495-era
work found it to be exactly a bigram table at 0.000 nats.
That result said attn0 IS a table. It did not say how big the
table has to be. The algebra bounds it: for head h the score
factor is
    E_t^T W_q,h^T R_d W_k,h E_s   (d = t - s, R_d the rotation)
so each factor is a bilinear form of rank at most 128, and the
head reads its input through a 128-dimensional bottleneck twice
over. The question this run answers is how much of that 128 is
load-bearing.
Method: truncate each head's query and key maps to rank r by SVD,
for both QK factors, and price the whole model. Rank 128 is the
untouched layer and is the sanity check. Rank r < 128 is an
interpretable claim -- "this head compares tokens along r
directions" -- and the cost curve says which r is honest.
Naming comes free at layer 0 because the input is the embedding:
the top query direction of a head is a scoring function over the
vocabulary, computable with no data at all.
REGISTERED PREDICTIONS:
  (0) THE FOLDING IS EXACT: the captured attn0 input equals
      rms_norm(wte[t]) to 1e-6 relative, checked directly. This is
      the premise of everything else and failure VOIDS the run;
  (a) LOW RANK: truncating every head's Q and K maps to rank 16 of
      128 costs less than 0.10 nats on the whole model;
  (b) BEATS RANDOM: at rank 16, the SVD truncation costs less than
      a third of what projecting onto 16 RANDOM directions costs,
      three draws;
  (c) THE EXTREME POINT: the cost at rank 4 is reported, and the
      rank at which cost first exceeds 0.10 nats is named. No bar
      -- this is the number the benchmark wants;
  (d) NAMEABLE: for the highest-cost head, the top two query
      directions and the top two key directions each have >= 7 of
      their 10 highest-scoring FREQUENT tokens in one fine class
      (determiner, preposition, pronoun, digit, punctuation,
      capitalized, subword). At least 2 of the 4 must clear it.
  NULL: rank 128 must cost under 1e-3 nats. If the untruncated
      reconstruction is not free, the SVD machinery is wrong and
      every other number is meaningless."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=0; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn0_rank_results.json'
NFRESH=48
RANKS=[2,4,8,16,32,64,128]
DET={'a','an','the','this','that','these','those','his','her',
     'its','their','our','your','my','some','any','no','each',
     'every','another','both','all'}
PREP={'of','to','in','for','on','with','at','by','from','as',
      'into','about','over','after','before','between','through',
      'during','under','against','without','within','onto',
      'toward','towards','upon','among','across'}
PRON={'he','she','it','they','we','you','i','him','them','us',
      'me','who','which','what','hers','theirs','itself'}

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
    at=m.transformer.h[LJ].attn
    fresh=cl.fineweb_rows(NFRESH)
    # (0) the folding check
    cap={}
    h0=at.register_forward_pre_hook(
        lambda mo_,a_: cap.__setitem__('X',a_[0]))
    bb=fresh[:4,:257].to(DEV); idx=bb[:,:-1].contiguous()
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    h0.remove()
    E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
    rel=float((cap['X'].float()-E).norm()/E.norm())
    print(f'(0) attn0 input vs rms_norm(wte): {rel:.3e}',flush=True)
    p0=rel<=1e-6
    print(f"(0) FOLDING EXACT: {'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'rel':rel},open(OUT,'w'),indent=1)
        return
    WQ=[at.c_q,at.c_k,at.c_q2,at.c_k2]
    orig={i:w.weight.data.clone() for i,w in enumerate(WQ)}
    # per-head SVD of each map
    def truncated(r,random=False,seed=0):
        new=[]
        for i,w in enumerate(WQ):
            W=orig[i].float().clone()
            for h in range(NH):
                blk=W[h*128:(h+1)*128]           # (128, D)
                if random:
                    g=torch.Generator(device=DEV).manual_seed(
                        seed*100+i*10+h)
                    Vr=torch.randn(r,D,generator=g,device=DEV)
                    Q,_=torch.linalg.qr(Vr.T)
                    W[h*128:(h+1)*128]=blk@Q@Q.T
                else:
                    U,S,Vh=torch.linalg.svd(blk,full_matrices=False)
                    W[h*128:(h+1)*128]=(U[:,:r]*S[:r])@Vh[:r]
            new.append(W)
        return new

    def price(new=None):
        if new is not None:
            for i,w in enumerate(WQ):
                w.weight.data.copy_(new[i].to(w.weight.dtype))
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            b2=fresh[i:i+4,:257].to(DEV)
            ix=b2[:,:-1].contiguous(); tg=b2[:,1:].reshape(-1)
            B=b2.shape[0]
            xx=F.rms_norm(m.transformer.wte(ix),(D,)); x0=xx; v1=None
            for blk in m.transformer.h: xx,v1=blk(xx,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(xx,(D,)))
                              /30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
        if new is not None:
            for i,w in enumerate(WQ):
                w.weight.data.copy_(orig[i].to(w.weight.dtype))
        return float(ce.mean())

    base=price()
    print(f'baseline CE {base:.4f}',flush=True)
    curve={}
    for r in RANKS:
        c=price(truncated(r))-base
        rnd=[]
        if r<128:
            for s in (1,2,3):
                rnd.append(round(price(truncated(r,True,s))-base,4))
        curve[r]={'cost':round(c,4),'random':rnd}
        print(f'rank {r:>4}: cost {c:+.4f} | random {rnd}',
              flush=True)
        json.dump(curve,open(OUT,'w'),indent=1)
    # naming for the head with the largest deletion cost
    atlas=json.load(open(PT+'head_atlas_results.json'))['atlas']
    cand={k:v['delete_cost'] for k,v in atlas.items()
          if k.startswith('0.')}
    tophead=int(max(cand,key=cand.get).split('.')[1]) if cand else 0
    print(f'\nnaming head 0.{tophead} (atlas delete cost '
          f'{cand.get(f"0.{tophead}",0):.5f})',flush=True)
    rows=cl.rows()
    cnt=torch.bincount(rows.reshape(-1),
                       minlength=m.transformer.wte.weight.shape[0])
    freq=cnt.argsort(descending=True)[:5000]
    Et=F.rms_norm(m.transformer.wte.weight.float(),(D,))[freq]
    names=[]; pure=0
    for tag,i in (('query',0),('key',1)):
        blk=orig[i].float()[tophead*128:(tophead+1)*128]
        U,S,Vh=torch.linalg.svd(blk,full_matrices=False)
        for k in range(2):
            w=Vh[k]
            sc=(Et@w)
            top=sc.abs().argsort(descending=True)[:10].tolist()
            toks=[cl.d1(int(freq[t])) for t in top]
            cls=[fine_class(s) for s in toks]
            best=max(set(cls),key=cls.count); c=cls.count(best)
            if c>=7: pure+=1
            names.append({'side':tag,'dir':k,'sv':round(float(S[k]),3),
                          'dominant_class':best,'purity':c,
                          'tokens':toks})
            print(f"  {tag} dir {k} (sv {float(S[k]):.2f}): {best} "
                  f"{c}/10 -> {[repr(x) for x in toks[:6]]}",
                  flush=True)
    nul=abs(curve[128]['cost'])<1e-3
    va,_=cl.score_bar('a',0.10-curve[16]['cost'],1e-9)
    r16=min(curve[16]['random']) if curve[16]['random'] else 1
    vb,_=cl.score_bar('b',r16-3*max(curve[16]['cost'],0),1e-9)
    first=next((r for r in RANKS if curve[r]['cost']<0.10),None)
    print(f"(c) rank 4 costs {curve[4]['cost']:+.4f}; cost first "
          f"falls under 0.10 nats at rank {first}")
    print(f'(d) {pure}/4 leading directions are class-pure')
    print(f"NULL (rank 128 costs {curve[128]['cost']:+.5f} < 1e-3): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'folding_rel':rel,'baseline_ce':round(base,4),
         'curve':{str(k):v for k,v in curve.items()},
         'named_head':f'0.{tophead}','directions':names,
         'class_pure':pure,'first_rank_under_0.10':first,
         'pred_0':True,'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_d':bool(pure>=2),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
