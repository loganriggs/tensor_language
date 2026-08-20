"""FRONT TABLE 2 -- what variable is block 1 a function of?
542 replaced block 0 with a printed table: 50304 rows, 64 columns,
0.18 nats, columns that are determiners, punctuation, initial
capitals, digits and sentence openers. It worked because block 0's
input is EXACTLY the token embedding (535), so the indexing
variable was known in advance rather than guessed.
Block 1 has no such guarantee. Its input is the residual after
block 0, which is the embedding plus what attention layer 0 and
mlp0 wrote -- and attn0 was shown long ago to be exactly a bigram
table. So the natural guess is that block 1's write is a function
of the TOKEN PAIR, and the way to find out is to build both tables
and price them.
  unigram   one row per current token
  bigram    one row per (previous, current) pair seen in the
            corpus, backing off to the unigram row for pairs that
            were never observed
  ceiling   the real write projected to the same 64 dimensions
            (541 measured 0.378 for block 1)
  shuffled  the unigram table indexed by a RANDOM token, as the
            control that the indexing variable is doing the work
Tables are built in the 64-dimensional interface basis, so a
bigram table over the corpus's observed pairs costs 64 floats per
pair rather than 1152.
The comparison is the point. If the bigram table closes most of
the gap that the unigram table leaves, block 1's write is a
function of the token pair and the folding programme has a second
replaceable block. If it does not, block 1 depends on something
wider than two tokens, and the honest answer is that the
substitution route stops at block 0 -- which would also explain
why 541 found block 1 the hardest of the six to compress (0.378
nats at rank 64, five times block 0's 0.081).
REGISTERED PREDICTIONS:
  (0) SANITY: the ceiling arm reproduces 541's block-1 figure of
      0.3781 to within 0.03 nats. Failure means this is not the
      same measurement and VOIDS the run;
  (a) A TABLE AT ALL: the unigram table for block 1 costs under
      0.60 nats;
  (b) THE PAIR HELPS: the bigram table beats the unigram table by
      at least 0.10 nats. This is the claim that the indexing
      variable is the token pair;
  (c) IT CLOSES THE GAP: the bigram table's distance to the
      ceiling is less than half the unigram table's distance.
      Reported either way, since the fraction of the gap closed is
      the quantity that says how much of block 1 is bigram;
  NULL: the shuffled-index table must cost at least twice the
      unigram table. If indexing on a random token does as well,
      neither table is using its variable."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; BLK=1; R=64
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'front_table2_results.json'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    V=m.transformer.wte.weight.shape[0]
    at=m.transformer.h[BLK].attn; mlp=m.transformer.h[BLK].mlp
    rows=cl.rows(); fresh=cl.fineweb_rows(NFRESH)
    st={}
    def cap_hooks(sink):
        h1=at.register_forward_hook(
            lambda mo,i_,o_: st.__setitem__(
                'a',(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))
        h2=mlp.register_forward_hook(
            lambda mo,i_,o_: (sink.append(
                (st['a']+o_.float()).reshape(-1,D)),o_)[1])
        return [h1,h2]
    # interface basis from fresh text
    sink=[]; hs=cap_hooks(sink)
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hs: h.remove()
    Y=torch.cat(sink); mu=Y.mean(0)
    _,_,Vh=torch.linalg.svd(Y-mu,full_matrices=False)
    B64=Vh[:R]                                   # (R, D)
    del sink,Y
    # accumulate unigram and bigram tables in the 64-dim basis
    uni=torch.zeros(V,R,device=DEV); ucnt=torch.zeros(V,device=DEV)
    pairs={}; pkeys=[]
    pac=[]; pcn=[]
    pair_index={}
    for i in range(0,rows.shape[0],4):
        bb=rows[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        sink=[]; hs=cap_hooks(sink)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        W=((sink[0]-mu)@B64.T)                   # (B*T, R)
        cur=idx.reshape(-1)
        prev=torch.cat([torch.zeros(idx.shape[0],1,dtype=idx.dtype,
                                    device=DEV),idx[:,:-1]],1) \
             .reshape(-1)
        uni.index_add_(0,cur,W)
        ucnt.index_add_(0,cur,torch.ones_like(cur,dtype=torch.float))
        key=(prev.long()*V+cur.long()).cpu()
        for j,k in enumerate(key.tolist()):
            e=pair_index.get(k)
            if e is None:
                pair_index[k]=len(pac); pac.append(W[j]); pcn.append(1)
            else:
                pac[e]=pac[e]+W[j]; pcn[e]+=1
    uni=uni/ucnt.clamp_min(1).unsqueeze(1)
    P=torch.stack(pac)/torch.tensor(pcn,device=DEV,
                                    dtype=torch.float).unsqueeze(1)
    print(f'{int((ucnt>0).sum())} tokens and {len(pair_index)} '
          f'token pairs tabulated',flush=True)

    def price(mode=None,seed=0):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            Bn=bb.shape[0]; hs=[]
            if mode is not None:
                cur=idx.reshape(-1)
                if mode=='shuffled':
                    g=torch.Generator(device=DEV).manual_seed(seed)
                    cur=cur[torch.randperm(cur.numel(),generator=g,
                                           device=DEV)]
                prev=torch.cat([torch.zeros(Bn,1,dtype=idx.dtype,
                                device=DEV),idx[:,:-1]],1).reshape(-1)
                if mode=='bigram':
                    keys=(prev.long()*V+cur.long()).cpu().tolist()
                    rowsel=torch.tensor(
                        [pair_index.get(k,-1) for k in keys],
                        device=DEV)
                    hit=rowsel>=0
                    C=uni[cur].clone()
                    if int(hit.sum()): C[hit]=P[rowsel[hit]]
                else:
                    C=uni[cur]
                lookup=(mu+C@B64).view(Bn,T,D)
                sl={}
                hs.append(at.register_forward_hook(
                    lambda mo,i_,o_,sl=sl: (sl.__setitem__(
                        'a',(o_[0] if isinstance(o_,tuple) else o_)
                        .detach().float()),o_)[1]))
                def fh(mo,i_,o_,sl=sl,lookup=lookup,mode=mode):
                    a=sl['a']
                    if mode=='ceiling':
                        w=a+o_.float()
                        new=mu+((w-mu)@B64.T)@B64
                    else:
                        new=lookup
                    return (new-a).to(o_.dtype)
                hs.append(mlp.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+Bn]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').view(Bn,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=price()
    ceil_=price('ceiling')-base
    unic=price('unigram')-base
    bic=price('bigram')-base
    shuf=[round(price('shuffled',s)-base,4) for s in (1,2,3)]
    print(f'baseline CE {base:.4f}',flush=True)
    print(f'ceiling (real write, rank {R}): {ceil_:+.4f}')
    print(f'unigram table: {unic:+.4f} | bigram table: {bic:+.4f}')
    print(f'shuffled-index table: {shuf}')
    p0=abs(ceil_-0.3781)<=0.03
    va=unic<0.60
    vb=(unic-bic)>=0.10
    gap_u=unic-ceil_; gap_b=bic-ceil_
    vc=gap_b<0.5*gap_u
    nul=min(shuf)>=2*max(unic,1e-6)
    print(f"\n(0) ceiling {ceil_:+.4f} vs 541's 0.3781: "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'ceiling':ceil_},
                  open(OUT,'w'),indent=1); return
    print(f"(a) unigram under 0.60: {'HELD' if va else 'FAILED'}")
    print(f"(b) bigram beats unigram by {unic-bic:+.4f} >= 0.10: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"(c) gap to ceiling: unigram {gap_u:+.4f} -> bigram "
          f"{gap_b:+.4f} ({100*(1-gap_b/max(gap_u,1e-9)):.0f}% "
          f"closed): {'HELD' if vc else 'FAILED'}")
    print(f"NULL (shuffled {min(shuf):+.4f} >= 2x unigram): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'block':BLK,'rank':R,
         'ceiling':round(ceil_,4),'unigram':round(unic,4),
         'bigram':round(bic,4),'shuffled':shuf,
         'n_tokens':int((ucnt>0).sum()),'n_pairs':len(pair_index),
         'gap_unigram':round(gap_u,4),'gap_bigram':round(gap_b,4),
         'fraction_closed':round(1-gap_b/max(gap_u,1e-9),3),
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'pred_c':bool(vc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
