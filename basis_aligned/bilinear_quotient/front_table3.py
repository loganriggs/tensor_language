"""FRONT TABLE 3 -- how far does the table route reach?
542 replaced block 0 with a table indexed by the token (0.18
nats). 544 replaced block 1 with a table indexed by the token pair
(0.52, closing 73% of what a unigram table leaves). 546 showed the
two compose additively at 0.67 and that each should be fitted
against the ORIGINAL network rather than the partially replaced
one.
Both indexing variables were derivable in advance: block 0's input
is exactly the embedding, and the only thing between block 1 and
the embedding is an exact bigram table. Block 2 has no such
derivation. Between it and the embedding sits attention layer 1,
which FAILS the bigram-table test at +1.470 where attn0 passes at
0.000 -- so whatever attn1 computes is not a function of two
tokens, and block 2's write may not be either.
The question is therefore how the required context grows with
depth, and the way to answer it is to build the ladder and see
where it stops paying:
  unigram   one row per current token
  bigram    one row per (previous, current) pair
  trigram   one row per (t-2, t-1, t) triple
each backing off to the shorter context when a longer one was
never observed, all built in block 2's own 64-dimensional
interface basis over the census corpus, and all fitted against the
real network per 546.
If the trigram adds little over the bigram, block 2 depends on
something a local window cannot capture, and the honest statement
is that the table route reaches two blocks. If it keeps paying,
the ladder continues and the next question is how far.
REGISTERED PREDICTIONS:
  (0) SANITY: the ceiling arm -- the real write at rank 64 --
      reproduces 541's block-2 figure of 0.1589 to within 0.03.
      Failure VOIDS the run;
  (a) THE PAIR STILL HELPS: the bigram table beats the unigram
      table by at least 0.10 nats, as it did at block 1;
  (b) THE TRIPLE STILL HELPS: the trigram table beats the bigram
      table by at least 0.05 nats. This is the claim that context
      keeps growing smoothly with depth. A failure here is the
      more interesting outcome and means the local-window route
      stops at two tokens;
  (c) THE LADDER: report the fraction of the unigram-to-ceiling
      gap closed by each rung. No bar;
  NULL: the shuffled-index table must cost at least twice the
      unigram table."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; BLK=2; R=64
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'front_table3_results.json'
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
    pac=[]; pcn=[]; pair_index={}
    tac=[]; tcn=[]; tri_index={}
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
        prev2=torch.cat([torch.zeros(idx.shape[0],2,dtype=idx.dtype,
                                     device=DEV),idx[:,:-2]],1) \
              .reshape(-1)
        key=(prev.long()*V+cur.long()).cpu()
        tkey=(prev2.long()*V*V+prev.long()*V+cur.long()).cpu()
        for j,k in enumerate(key.tolist()):
            e=pair_index.get(k)
            if e is None:
                pair_index[k]=len(pac); pac.append(W[j]); pcn.append(1)
            else:
                pac[e]=pac[e]+W[j]; pcn[e]+=1
        for j,k in enumerate(tkey.tolist()):
            e=tri_index.get(k)
            if e is None:
                tri_index[k]=len(tac); tac.append(W[j]); tcn.append(1)
            else:
                tac[e]=tac[e]+W[j]; tcn[e]+=1
    uni=uni/ucnt.clamp_min(1).unsqueeze(1)
    P=torch.stack(pac)/torch.tensor(pcn,device=DEV,
                                    dtype=torch.float).unsqueeze(1)
    Tm=torch.stack(tac)/torch.tensor(tcn,device=DEV,
                                     dtype=torch.float).unsqueeze(1)
    print(f'{int((ucnt>0).sum())} tokens, {len(pair_index)} pairs '
          f'and {len(tri_index)} triples tabulated',flush=True)

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
                if mode=='trigram':
                    prev2=torch.cat([torch.zeros(Bn,2,
                        dtype=idx.dtype,device=DEV),idx[:,:-2]],1) \
                        .reshape(-1)
                    tk=(prev2.long()*V*V+prev.long()*V
                        +cur.long()).cpu().tolist()
                    pk=(prev.long()*V+cur.long()).cpu().tolist()
                    ts=torch.tensor([tri_index.get(k,-1) for k in tk],
                                    device=DEV)
                    ps=torch.tensor([pair_index.get(k,-1) for k in pk],
                                    device=DEV)
                    C=uni[cur].clone()
                    ph=ps>=0
                    if int(ph.sum()): C[ph]=P[ps[ph]]
                    th=ts>=0
                    if int(th.sum()): C[th]=Tm[ts[th]]
                elif mode=='bigram':
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
    tric=price('trigram')-base
    shuf=[round(price('shuffled',s)-base,4) for s in (1,2,3)]
    print(f'baseline CE {base:.4f}',flush=True)
    print(f'ceiling (real write, rank {R}): {ceil_:+.4f}')
    print(f'unigram {unic:+.4f} | bigram {bic:+.4f} | trigram '
          f'{tric:+.4f}')
    print(f'shuffled-index table: {shuf}')
    p0=abs(ceil_-0.1589)<=0.03
    va=(unic-bic)>=0.10
    vb=(bic-tric)>=0.05
    gap_u=unic-ceil_; gap_b=bic-ceil_; gap_t=tric-ceil_
    nul=min(shuf)>=2*max(unic,1e-6)
    print(f"\n(0) ceiling {ceil_:+.4f} vs 541's 0.3781: "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'ceiling':ceil_},
                  open(OUT,'w'),indent=1); return
    print(f"(a) bigram beats unigram by {unic-bic:+.4f} >= 0.10: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) trigram beats bigram by {bic-tric:+.4f} >= 0.05: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"(c) gap to ceiling {gap_u:+.4f} -> {gap_b:+.4f} -> "
          f"{gap_t:+.4f} | closed "
          f"{100*(1-gap_b/max(gap_u,1e-9)):.0f}% then "
          f"{100*(1-gap_t/max(gap_u,1e-9)):.0f}%")
    print(f"NULL (shuffled {min(shuf):+.4f} >= 2x unigram): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'block':BLK,'rank':R,
         'ceiling':round(ceil_,4),'unigram':round(unic,4),
         'bigram':round(bic,4),'trigram':round(tric,4),
         'shuffled':shuf,'n_triples':len(tri_index),
         'n_tokens':int((ucnt>0).sum()),'n_pairs':len(pair_index),
         'gap_unigram':round(gap_u,4),'gap_bigram':round(gap_b,4),
         'gap_trigram':round(gap_t,4),
         'fraction_closed_bigram':round(1-gap_b/max(gap_u,1e-9),3),
         'fraction_closed_trigram':round(1-gap_t/max(gap_u,1e-9),3),
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
