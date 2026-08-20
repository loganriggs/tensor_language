"""FIXED QUERY CENSUS -- classify all 162 heads by whether their
query is fixed (selection) or content-dependent (matching).
573 established a one-measurement classifier: replace a head's
query with a single fixed (position-average) query; if the head is
reproduced, it is a fixed-query selection head, if it breaks, it is
a content-matching head. This applies that test to EVERY head in
the model, target-free, to produce a global map of the attention
repertoire.
For each head, its pre-rotary query at every position is replaced
by that head's own mean query over all positions (a single fixed
vector), everything else untouched, and the whole-model
cross-entropy cost is measured. A head whose CE barely moves has a
fixed query (its per-position query variation is not load-bearing);
a head whose CE jumps has a content-dependent query.
This is grounded (the classifier is validated on the four verified
heads) and global (one census, not per-head drilling).
REGISTERED PREDICTIONS:
  (0) IDENTITY FREE: replacing a head's query with the REAL query
      (no-op control) costs < 1e-3 nats, verifying the machinery;
  (a) STRUCTURAL HEADS ARE FIXED: the three verified fixed-query
      heads (13.8, 10.7, 12.6) each cost < 0.02 nats under the mean
      query -- their queries are fixed, so mean-filling is nearly
      free;
  (b) INDUCTION HEADS ARE NOT: head 8.3 (digit induction) and the
      induction band cost more than the structural heads;
  (c) THE MAP: report the distribution of mean-query costs across
      all 162 heads and the fraction that are fixed-query
      (cost < 0.02) vs content (cost > 0.05). No bar -- the
      distribution is the result;
  NULL: for the fixed-query heads, a RANDOM-direction query of
      matched norm must cost more than the mean query -- the mean
      is a meaningful fixed query, not just any constant."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9; NL=18
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fixed_query_census_results.json'
NFRESH=32

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    # PASS 1: mean pre-rotary query per (layer, head)
    qsum={(li,h):torch.zeros(128,device=DEV)
          for li in range(NL) for h in range(NH)}
    qn=0; caps={}
    hooks=[]
    for li in range(NL):
        at=m.transformer.h[li].attn
        hooks.append(at.register_forward_pre_hook(
            (lambda li: lambda mo_,a_: caps.__setitem__(li,a_[0]))(li)))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for li in range(NL):
            at=m.transformer.h[li].attn
            qp=F.rms_norm(at.c_q(caps[li]).view(B,T,NH,128),
                          (128,)).float()
            for h in range(NH):
                qsum[(li,h)]+=qp[:,:,h].reshape(-1,128).sum(0)
        qn+=B*T
    for hk in hooks: hk.remove()
    qmean={k:v/max(qn,1) for k,v in qsum.items()}

    def price(mode,li=None,h=None,seed=0):
        """mode: None | 'mean' | 'rand' -- replace head (li,h) query."""
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if mode is not None:
                at=m.transformer.h[li].attn
                def fh(mo,args,o_,at=at,h=h):
                    y,v1r=o_; X=args[0]
                    v1b=args[1] if args[1] is not None else v1r
                    z,vm=cl.head_parts(li,X,v1b); z=z.clone()
                    cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
                    if mode=='mean':
                        qp=qmean[(li,h)][None,None].expand(B,T,128)
                    elif mode=='rand':
                        g=torch.Generator(device=DEV).manual_seed(seed)
                        w=torch.randn(128,generator=g,device=DEV)
                        w=w/w.norm()*qmean[(li,h)].norm()
                        qp=w[None,None].expand(B,T,128)
                    else:
                        qp=F.rms_norm(at.c_q(X).view(B,T,NH,128),
                                      (128,))[:,:,h].float()
                    qrot=are(qp[:,:,None].expand(B,T,NH,128)
                             .contiguous(),cq,sq)[:,:,h]
                    def rk(W): return are(F.rms_norm(
                        W(X).view(B,T,NH,128),(128,)),cq,sq)[:,:,h].float()
                    s1=torch.einsum('bqd,bkd->bqk',qrot.float(),
                                    rk(at.c_k))/128
                    s2=torch.einsum('bqd,bkd->bqk',rk(at.c_q2),
                                    rk(at.c_k2))/128
                    sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
                    z[:,h]=torch.einsum('bqk,bkd->bqd',sc,
                                        vm[:,:,h].float())
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for hk in hs: hk.remove()
        return float(ce.mean())

    base=price(None)
    # identity check on one head
    ident=price('real',13,8)-base
    print(f'baseline {base:.4f} | identity check (real query '
          f'reinjected) {ident:+.5f}',flush=True)
    costs={}
    for li in range(NL):
        for h in range(NH):
            costs[f'{li}.{h}']=round(price('mean',li,h)-base,4)
        print(f'layer {li} done',flush=True)
        json.dump({'costs':costs},open(OUT,'w'),indent=1)
    order=sorted(costs,key=lambda k:costs[k])
    fixed=[k for k in costs if costs[k]<0.02]
    content=[k for k in costs if costs[k]>0.05]
    # null: random query for the fixed heads
    rnd={}
    for k in ('13.8','10.7','12.6'):
        li,h=int(k.split('.')[0]),int(k.split('.')[1])
        rnd[k]=round(price('rand',li,h,1)-base,4)
    p0=abs(ident)<1e-3
    pa=all(costs[k]<0.02 for k in ('13.8','10.7','12.6'))
    pb=costs.get('8.3',0)>max(costs['13.8'],costs['10.7'],costs['12.6'])
    nul=all(rnd[k]>costs[k] for k in rnd)
    print(f"\n(0) identity free ({ident:+.5f}): "
          f"{'HELD' if p0 else 'FAILED'}")
    print(f"(a) structural heads fixed (13.8 {costs['13.8']}, "
          f"10.7 {costs['10.7']}, 12.6 {costs['12.6']} < 0.02): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) induction 8.3 ({costs.get('8.3')}) > structural: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) {len(fixed)}/162 fixed-query (<0.02), "
          f"{len(content)}/162 content (>0.05)")
    print(f"    most content-dependent: "
          f"{[(k,costs[k]) for k in order[::-1][:8]]}")
    print(f"NULL (random query costs more than mean for fixed heads "
          f"{rnd}): {'ok' if nul else 'CHECK'}")
    out={'baseline':round(base,4),'identity':round(ident,5),
         'costs':costs,'n_fixed':len(fixed),'n_content':len(content),
         'most_content':[(k,costs[k]) for k in order[::-1][:12]],
         'most_fixed':[(k,costs[k]) for k in order[:12]],
         'random_query_fixed_heads':rnd,
         'pred_0':bool(p0),'pred_a':bool(pa),'pred_b':bool(pb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
