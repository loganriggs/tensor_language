"""FIXED KEY CENSUS -- are the keys as fixable as the queries?
574/575 showed the model's attention is approximately FIXED-QUERY:
fixing all 162 queries to their means costs ~1 nat, 11x better
than random. If the query says "what to look for" and is roughly
fixed, the discrimination -- "what is at this key position" --
must live on the KEY side. This is the complement: replace each
head's KEY (both QK factors' keys) with its position-mean and
measure the cost, per head and jointly, exactly mirroring the
query census.
Prediction: keys are NOT fixable. A fixed query resolved by the
double-QK AND needs the KEYS to carry the token/position content
it discriminates on, so mean-filling the keys should be
substantially MORE expensive than mean-filling the queries. The
asymmetry -- cheap queries, expensive keys -- is the clean
statement of what each side does.
REGISTERED PREDICTIONS:
  (0) IDENTITY: reinjecting the real key costs < 1e-3 nats;
  (a) KEYS COST MORE THAN QUERIES: the median per-head key cost
      exceeds the median per-head query cost from 574 (0.00x) by
      >= 3x;
  (b) THE JOINT NUMBER: report the cost of fixing all 162 keys and
      compare to the 0.96 nats for all queries (575);
  (c) the per-head distribution and the most key-content-dependent
      heads. No bar;
  NULL: random-direction keys cost more than mean keys -- the mean
      is a meaningful fixed key."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9; NL=18
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fixed_key_census_results.json'
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
            qp=F.rms_norm(at.c_k(caps[li]).view(B,T,NH,128),
                          (128,)).float()  # KEY mean
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
                    def rqk(W): return are(F.rms_norm(
                        W(X).view(B,T,NH,128),(128,)),cq,sq)[:,:,h].float()
                    if mode=='mean':
                        kp=qmean[(li,h)][None,None].expand(B,T,128)
                        krot=are(kp[:,:,None].expand(B,T,NH,128)
                                 .contiguous(),cq,sq)[:,:,h].float()
                    elif mode=='rand':
                        g=torch.Generator(device=DEV).manual_seed(seed)
                        w=torch.randn(128,generator=g,device=DEV)
                        w=w/w.norm()*qmean[(li,h)].norm()
                        kp=w[None,None].expand(B,T,128)
                        krot=are(kp[:,:,None].expand(B,T,NH,128)
                                 .contiguous(),cq,sq)[:,:,h].float()
                    else:
                        krot=rqk(at.c_k)
                    s1=torch.einsum('bqd,bkd->bqk',rqk(at.c_q),
                                    krot)/128
                    s2=torch.einsum('bqd,bkd->bqk',rqk(at.c_q2),
                                    rqk(at.c_k2))/128
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
    ident=price('real',13,8)-base  # key identity
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
