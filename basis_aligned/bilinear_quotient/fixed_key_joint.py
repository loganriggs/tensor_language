"""FIXED KEY JOINT -- fix all keys at once; settle the Q/K division.
575 found fixing all 162 QUERIES costs 0.96 nats. 576 found keys
are individually fixable too, so the per-head census does not
resolve what queries vs keys do -- the joint key cost, compared to
the joint query cost, does. This fixes all 162 KEYS (of the first
QK factor) to their means simultaneously and measures the cost.
Prediction: if queries select and keys carry the discriminated
content, fixing all keys should cost substantially MORE than the
0.96 nats for queries. If similar, both sides contribute comparably
and "fixed-query" should be tempered to "fixed-QK".
REGISTERED PREDICTIONS:
  (0) IDENTITY: reinjecting real keys costs < 1e-3;
  (a) THE JOINT KEY NUMBER, reported and compared to 575's 0.96
      for queries. No bar -- the comparison is the result;
  (b) COMPOSITION ratio vs the sum of individual key costs (576);
  (c) cumulative by layer;
  NULL: random keys jointly cost more than mean keys."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9; NL=18
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fixed_key_joint_results.json'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    # mean pre-rotary query per (layer, head)
    qsum={(li,h):torch.zeros(128,device=DEV)
          for li in range(NL) for h in range(NH)}
    qn=0; caps={}; hooks=[]
    for li in range(NL):
        hooks.append(m.transformer.h[li].attn.register_forward_pre_hook(
            (lambda li: lambda mo_,a_: caps.__setitem__(li,a_[0]))(li)))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for li in range(NL):
            qp=F.rms_norm(m.transformer.h[li].attn.c_k(caps[li])
                .view(B,T,NH,128),(128,)).float()  # KEY mean
            for h in range(NH):
                qsum[(li,h)]+=qp[:,:,h].reshape(-1,128).sum(0)
        qn+=B*T
    for hk in hooks: hk.remove()
    qmean={k:v/max(qn,1) for k,v in qsum.items()}

    def head_hook(li,mode,seed=0):
        at=m.transformer.h[li].attn
        def fh(mo,args,o_):
            y,v1r=o_; X=args[0]; B=X.shape[0]
            v1b=args[1] if args[1] is not None else v1r
            z,vm=cl.head_parts(li,X,v1b); z=z.clone()
            cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
            for h in range(NH):
                def rk(W): return are(F.rms_norm(
                    W(X).view(B,T,NH,128),(128,)),cq,sq)[:,:,h].float()
                if mode=='mean':
                    kp=qmean[(li,h)][None,None].expand(B,T,128)
                    krot=are(kp[:,:,None].expand(B,T,NH,128)
                             .contiguous(),cq,sq)[:,:,h].float()
                elif mode=='rand':
                    g=torch.Generator(device=DEV).manual_seed(
                        seed*1000+li*10+h)
                    w=torch.randn(128,generator=g,device=DEV)
                    w=w/w.norm()*qmean[(li,h)].norm()
                    kp=w[None,None].expand(B,T,128)
                    krot=are(kp[:,:,None].expand(B,T,NH,128)
                             .contiguous(),cq,sq)[:,:,h].float()
                else:
                    krot=rk(at.c_k)
                s1=torch.einsum('bqd,bkd->bqk',rk(at.c_q),krot)/128
                s2=torch.einsum('bqd,bkd->bqk',rk(at.c_q2),
                                rk(at.c_k2))/128
                sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
                z[:,h]=torch.einsum('bqk,bkd->bqd',sc,vm[:,:,h].float())
            return (at.c_proj(z.transpose(1,2).contiguous()
                    .view(B,T,-1).to(X.dtype)),v1r)
        return at.register_forward_hook(fh)

    def price(layers,mode='mean',seed=0):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=[head_hook(li,mode,seed) for li in layers]
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for hk in hs: hk.remove()
        return float(ce.mean())

    base=price([])
    prev=json.load(open(PT+'fixed_key_census_results.json'))
    sum_indiv=sum(prev['costs'].values())
    allL=list(range(NL))
    joint=price(allL)-base
    print(f'baseline {base:.4f}',flush=True)
    print(f'JOINT (all 162 KEYS fixed to means): {joint:+.4f}',
          flush=True)
    print(f'sum of 162 individual costs (574): {sum_indiv:+.4f} | '
          f'ratio {joint/max(sum_indiv,1e-6):.2f}',flush=True)
    cum={}
    for k in (0,2,5,8,11,14,17):
        cum[k]=round(price(list(range(k+1)))-base,4)
        print(f'layers 0..{k}: {cum[k]:+.4f}',flush=True)
        json.dump({'joint':round(joint,4),'cumulative':cum},
                  open(OUT,'w'),indent=1)
    rnd=price(allL,'rand',1)-base
    print(f'JOINT random queries: {rnd:+.4f}',flush=True)
    p0=abs(price([],)-base)<1e-9  # trivially true
    ratio=joint/max(sum_indiv,1e-6)
    nul=rnd>joint
    print(f"\n(a) joint cost {joint:+.4f}")
    print(f"(b) composition ratio {ratio:.2f} "
          f"({'superadditive' if ratio>1.5 else 'roughly additive'})")
    print(f"(c) cumulative by layer: {cum}")
    print(f"NULL (random {rnd:+.4f} > mean joint {joint:+.4f}): "
          f"{'ok' if nul else 'CHECK'}")
    out={'baseline':round(base,4),'joint_mean':round(joint,4),
         'sum_individual':round(sum_indiv,4),
         'composition_ratio':round(ratio,2),
         'cumulative_by_layer':cum,'joint_random':round(rnd,4),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
