"""PREFIX ACCUMULATION -- why doesn't compression compose?
540 and 541 both found the same shape. Every early component is
compressible on its own and nothing is compressible together:
weight truncation gives 1.78 nats jointly, interface projection
1.31 at rank 64 where the six individual costs sum to 0.84.
I stated the pattern and guessed at a cause -- that the residual
stream is a shared channel and the slack at one layer is used by
another. That is one of two explanations and they are
distinguishable.
  ACCUMULATION. Each compression injects a small independent
    error and the errors pile up through the remaining depth.
    Then the incremental cost of adding a block to the compressed
    set is about that block's own individual cost, and the total
    is roughly the sum -- possibly amplified by depth, but with no
    interaction between blocks.
  INTERACTION. What one block discards is read by another, so
    compressing two blocks costs more than compressing each. Then
    the incremental cost of adding a block EXCEEDS its individual
    cost, and by more the later it is added.
Method: at a fixed rank of 64, project the interfaces of a growing
PREFIX of blocks -- {0}, {0,1}, ... {0..5} -- and record the
increments. Then the reverse order, {5}, {5,4}, ... so each block
is measured both as an early addition and as a late one. Every
projection is the same operation 541 used, so the endpoints must
agree with it.
REGISTERED PREDICTIONS:
  (0) SANITY: the full six-block prefix at rank 64 reproduces
      541's joint number of 1.3137 to within 0.02 nats, and every
      prefix at rank 1152 costs under 1e-3. Failure means this run
      is not measuring the same thing and VOIDS it;
  (a) INTERACTION, NOT ACCUMULATION: for at least 4 of the 5
      additions in the forward order, the incremental cost of
      adding block k exceeds that block's own individual cost
      (0.081, 0.378, 0.159, 0.079, 0.083, 0.060 from 541);
  (b) POSITION MATTERS: adding block 0 LAST -- to a set that
      already has blocks 1 through 5 compressed -- costs at least
      twice what compressing block 0 alone costs (0.081). If
      interaction is real, the cost of compressing a block should
      depend on what else is already compressed;
  (c) THE CURVE: report both accumulation curves and every
      increment. No bar -- the shape is the result;
  NULL: the two orders must reach the SAME total, since the final
      set is identical. A discrepancy above 0.02 nats means the
      measurement is order-dependent for a reason that is not the
      model, and the run is reported as broken.
If (a) fails and the increments match the individual costs, the
answer is accumulation, the joint numbers are a depth artifact
rather than a fact about sharing, and 541's closing paragraph
about a shared channel should be withdrawn in turn."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'prefix_accumulation_results.json'
NFRESH=48; NB=6; RANK=64
INDIV={0:0.0811,1:0.3781,2:0.1589,3:0.0788,4:0.0825,5:0.0597}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    caps={i:{'a':[],'m':[]} for i in range(NB)}
    hs=[]
    for i in range(NB):
        hs.append(m.transformer.h[i].attn.register_forward_hook(
            (lambda i: lambda mo,i_,o_: caps[i]['a'].append(
                (o_[0] if isinstance(o_,tuple) else o_)
                .detach().float().reshape(-1,D).cpu()))(i)))
        hs.append(m.transformer.h[i].mlp.register_forward_hook(
            (lambda i: lambda mo,i_,o_: caps[i]['m'].append(
                o_.detach().float().reshape(-1,D).cpu()))(i)))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hs: h.remove()
    basis={}; means={}
    for i in range(NB):
        Y=(torch.cat(caps[i]['a'])+torch.cat(caps[i]['m'])).to(DEV)
        means[i]=Y.mean(0)
        _,_,Vh=torch.linalg.svd(Y-means[i],full_matrices=False)
        basis[i]=Vh; caps[i]=None

    def hooks_for(blocks,r):
        hs=[]
        for i in blocks:
            V=basis[i][:r]; P=V.T@V; mu=means[i]; st={}
            hs.append(m.transformer.h[i].attn.register_forward_hook(
                (lambda st: lambda mo,i_,o_: (st.__setitem__(
                    'a',(o_[0] if isinstance(o_,tuple) else o_)
                    .detach().float()),o_)[1])(st)))
            def fm(mo,i_,o_,P=P,mu=mu,st=st):
                a=st['a']; tot=a+o_.float()
                return ((mu+(tot-mu)@P)-a).to(o_.dtype)
            hs.append(m.transformer.h[i].mlp
                      .register_forward_hook(fm))
        return hs

    def price(blocks=(),r=RANK):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=hooks_for(blocks,r) if blocks else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=price()
    print(f'baseline CE {base:.4f} | individual costs at rank '
          f'{RANK}: {INDIV}',flush=True)
    curves={}
    for name,order in (('forward',list(range(NB))),
                       ('reverse',list(reversed(range(NB))))):
        pref=[]; prev=0.0; rows=[]
        for k in order:
            pref.append(k)
            c=price(tuple(pref))-base
            inc=c-prev
            rows.append({'added':k,'set':list(pref),
                         'cost':round(c,4),'increment':round(inc,4),
                         'individual':INDIV[k],
                         'ratio':round(inc/max(INDIV[k],1e-6),2)})
            print(f"{name}: +block {k} -> set {pref} cost {c:+.4f} "
                  f"(increment {inc:+.4f} vs individual "
                  f"{INDIV[k]:+.4f}, ratio "
                  f"{inc/max(INDIV[k],1e-6):.2f})",flush=True)
            prev=c
        curves[name]=rows
        json.dump(curves,open(OUT,'w'),indent=1)
    full_fwd=curves['forward'][-1]['cost']
    full_rev=curves['reverse'][-1]['cost']
    fullrank=price(tuple(range(NB)),1152)-base
    p0=(abs(full_fwd-1.3137)<=0.02 and abs(fullrank)<1e-3)
    print(f"\n(0) full prefix {full_fwd:+.4f} vs 541's 1.3137, and "
          f"full-rank prefix {fullrank:+.5f}: "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'full_fwd':full_fwd,
                   'fullrank':fullrank,'curves':curves},
                  open(OUT,'w'),indent=1); return
    exceed=[r for r in curves['forward'][1:]
            if r['increment']>r['individual']]
    va=len(exceed)>=4
    b0_last=curves['reverse'][-1]
    vb=b0_last['increment']>=2*INDIV[0]
    nul=abs(full_fwd-full_rev)<=0.02
    print(f"(a) increments exceed individual costs in "
          f"{len(exceed)} of 5 additions: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) block 0 added LAST costs "
          f"{b0_last['increment']:+.4f} vs {INDIV[0]:+.4f} alone "
          f"(ratio {b0_last['ratio']}): {'HELD' if vb else 'FAILED'}")
    print(f"NULL (both orders reach the same total: "
          f"{full_fwd:+.4f} vs {full_rev:+.4f}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'rank':RANK,
         'individual':INDIV,'curves':curves,
         'full_forward':round(full_fwd,4),
         'full_reverse':round(full_rev,4),
         'full_rank_check':round(fullrank,5),
         'n_increments_exceeding':len(exceed),
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
