"""FRONT TABLE COMPOSE -- two tables at once, and the refitting
question that composition raises.
542 replaced block 0's write with a table indexed by the token
(0.18 nats). 544 replaced block 1's with a table indexed by the
token pair (0.52, closing 73% of what a unigram table leaves).
545 closed the competing route: the best joint rank allocation for
the whole front costs 1.18, so substitution wins by a wide margin
one block at a time. Whether it wins two blocks at a time is not
yet measured, and 543 gives a concrete reason to doubt that the
costs add -- compressing blocks interacts, sometimes with
cancellation.
Composition also raises a methodological question this program has
not faced. Block 1's table was fitted against the REAL block 0. If
block 0 is replaced by its table, block 1's input changes, and the
table fitted in the original model is being applied in a model
that no longer exists. The honest alternative is to rebuild block
1's table with block 0's replacement already in place -- a
self-consistent, sequential fit. If refitting matters, every
layered stand-in this program has built by fitting components
independently is optimistic.
Arms:
  b0            block 0's token table alone      (reference, 0.18)
  b1            block 1's pair table alone       (reference, 0.52)
  naive         both, each fitted against the real model
  sequential    block 0's table, then block 1's table REFITTED
                with block 0's replacement active
  shuffled      both tables with shuffled indices (control)
All tables are built in each block's own 64-dimensional interface
basis over the census corpus, with backoff to the unigram row for
unseen pairs, exactly as in 542 and 544.
REGISTERED PREDICTIONS:
  (0) THE ARMS REPRODUCE: b0 and b1 alone come within 0.03 nats of
      0.182 and 0.522. Failure means this is not the same
      construction and VOIDS the run;
  (a) COMPOSITION IS NOT FREE: naive composition costs at least
      1.2x the sum of the two individual costs (0.704). 543 found
      interaction between compressed blocks and this asks whether
      substitution interacts the same way;
  (b) REFITTING HELPS: sequential composition beats naive by at
      least 0.15 nats. This is the methodological claim -- that
      fitting a stand-in against a model whose earlier parts have
      already been replaced is materially different from fitting
      it against the original;
  (c) THE HEADLINE NUMBER: report the cost of the best two-block
      substitution against the 1.18 that the best rank allocation
      of all six blocks achieves. No bar;
  NULL: the shuffled-index arm must cost at least three times the
      naive arm. Both tables share the same machinery, so if
      shuffling the index is cheap the composition is not using
      its variables."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; R=64
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'front_table_compose_results.json'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    V=m.transformer.wte.weight.shape[0]
    rows=cl.rows(); fresh=cl.fineweb_rows(NFRESH)
    blocks={0:(m.transformer.h[0].attn,m.transformer.h[0].mlp),
            1:(m.transformer.h[1].attn,m.transformer.h[1].mlp)}
    st={}

    def write_hooks(b,sink):
        at,mlp=blocks[b]
        h1=at.register_forward_hook(
            lambda mo,i_,o_,b=b: st.__setitem__(
                b,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))
        h2=mlp.register_forward_hook(
            lambda mo,i_,o_,b=b: (sink.append(
                (st[b]+o_.float()).reshape(-1,D)),o_)[1])
        return [h1,h2]

    def sub_hooks(b,lookup):
        """Replace block b's write with the given per-position
        lookup (B,T,D)."""
        at,mlp=blocks[b]; sl={}
        h1=at.register_forward_hook(
            lambda mo,i_,o_: (sl.__setitem__(
                'a',(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()),o_)[1])
        def fh(mo,i_,o_):
            return (lookup-sl['a']).to(o_.dtype)
        h2=mlp.register_forward_hook(fh)
        return [h1,h2]

    def basis_for(b,pre=None):
        sink=[]; hs=write_hooks(b,sink)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            extra=pre(idx) if pre else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in extra: h.remove()
        for h in hs: h.remove()
        Y=torch.cat(sink); mu=Y.mean(0)
        _,_,Vh=torch.linalg.svd(Y-mu,full_matrices=False)
        return mu,Vh[:R]

    def build(b,mu,B64,pair,pre=None):
        uni=torch.zeros(V,R,device=DEV)
        cnt=torch.zeros(V,device=DEV)
        pidx={}; pac=[]; pcn=[]
        for i in range(0,rows.shape[0],4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            sink=[]; hs=write_hooks(b,sink)
            extra=pre(idx) if pre else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in extra: h.remove()
            for h in hs: h.remove()
            W=(sink[0]-mu)@B64.T
            cur=idx.reshape(-1)
            uni.index_add_(0,cur,W)
            cnt.index_add_(0,cur,torch.ones_like(cur,
                                                 dtype=torch.float))
            if pair:
                prev=torch.cat([torch.zeros(idx.shape[0],1,
                    dtype=idx.dtype,device=DEV),idx[:,:-1]],1) \
                    .reshape(-1)
                for j,k in enumerate((prev.long()*V+cur.long())
                                     .cpu().tolist()):
                    e=pidx.get(k)
                    if e is None:
                        pidx[k]=len(pac); pac.append(W[j]); pcn.append(1)
                    else:
                        pac[e]=pac[e]+W[j]; pcn[e]+=1
        uni=uni/cnt.clamp_min(1).unsqueeze(1)
        Pm=(torch.stack(pac)/torch.tensor(pcn,device=DEV,
            dtype=torch.float).unsqueeze(1)) if pac else None
        return uni,pidx,Pm

    def lookup_for(idx,mu,B64,uni,pidx,Pm,pair,shuffle=0):
        Bn=idx.shape[0]; cur=idx.reshape(-1)
        if shuffle:
            g=torch.Generator(device=DEV).manual_seed(shuffle)
            cur=cur[torch.randperm(cur.numel(),generator=g,
                                   device=DEV)]
        C=uni[cur]
        if pair and Pm is not None:
            prev=torch.cat([torch.zeros(Bn,1,dtype=idx.dtype,
                device=DEV),idx[:,:-1]],1).reshape(-1)
            keys=(prev.long()*cur.new_tensor(V).long()
                  +cur.long()).cpu().tolist()
            sel=torch.tensor([pidx.get(k,-1) for k in keys],
                             device=DEV)
            hit=sel>=0
            if int(hit.sum()): C=C.clone(); C[hit]=Pm[sel[hit]]
        return (mu+C@B64).view(Bn,T,D)

    print('fitting block 0 table (token index)',flush=True)
    mu0,B0=basis_for(0)
    uni0,_,_=build(0,mu0,B0,pair=False)
    print('fitting block 1 table (pair index, against the real '
          'model)',flush=True)
    mu1,B1=basis_for(1)
    uni1,pidx1,P1=build(1,mu1,B1,pair=True)

    def pre0(idx,shuffle=0):
        return sub_hooks(0,lookup_for(idx,mu0,B0,uni0,None,None,
                                      False,shuffle))
    print('refitting block 1 table WITH block 0 replaced',
          flush=True)
    mu1s,B1s=basis_for(1,pre=pre0)
    uni1s,pidx1s,P1s=build(1,mu1s,B1s,pair=True,pre=pre0)

    def price(mode):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            Bn=bb.shape[0]; hs=[]
            sh=1 if mode=='shuffled' else 0
            if mode in ('b0','naive','sequential','shuffled'):
                hs+=sub_hooks(0,lookup_for(idx,mu0,B0,uni0,None,
                                           None,False,sh))
            if mode in ('b1','naive','shuffled'):
                hs+=sub_hooks(1,lookup_for(idx,mu1,B1,uni1,pidx1,
                                           P1,True,sh))
            if mode=='sequential':
                hs+=sub_hooks(1,lookup_for(idx,mu1s,B1s,uni1s,
                                           pidx1s,P1s,True,0))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+Bn]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').view(Bn,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=price('none')
    res={k:round(price(k)-base,4)
         for k in ('b0','b1','naive','sequential','shuffled')}
    print(f'\nbaseline CE {base:.4f}')
    for k,v in res.items(): print(f'  {k:>11}: {v:+.4f}')
    p0=(abs(res['b0']-0.182)<=0.03 and abs(res['b1']-0.522)<=0.03)
    s=res['b0']+res['b1']
    va=res['naive']>=1.2*s
    vb=(res['naive']-res['sequential'])>=0.15
    nul=res['shuffled']>=3*max(res['naive'],1e-6)
    print(f"\n(0) arms reproduce 542/544 (0.182, 0.522): "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'results':res},
                  open(OUT,'w'),indent=1); return
    print(f"(a) naive {res['naive']:+.4f} >= 1.2 x sum "
          f"({s:+.4f}): {'HELD' if va else 'FAILED'}")
    print(f"(b) sequential refit gains "
          f"{res['naive']-res['sequential']:+.4f} >= 0.15: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"(c) best two-block substitution "
          f"{min(res['naive'],res['sequential']):+.4f} against 1.18 "
          f"for the best rank allocation of all six blocks")
    print(f"NULL (shuffled {res['shuffled']:+.4f} >= 3x naive): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'results':res,
         'sum_individual':round(s,4),
         'refit_gain':round(res['naive']-res['sequential'],4),
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
