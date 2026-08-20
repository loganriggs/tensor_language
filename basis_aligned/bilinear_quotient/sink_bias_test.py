"""SINK BIAS TEST -- 433: head 5.7 is the costliest head in the
model (+0.916 nats to delete, 8x the next) and it reads POSITION
0 for 99.8% of queries (neighbour 5.6: 5.3%), where the value
norm is 730 against 197 elsewhere. This architecture has NO
softmax, so a sink cannot be absorbing normalisation pressure --
the head instead adds nearly the same vector at every position.
If that is right, the model's most important attention head is a
LEARNED BIAS GENERATOR that 1152 numbers replace.
Arms at layer 5 head 7: zero / mean (its write replaced by its
own per-batch mean vector) / crossmean (mean taken from OTHER
rows -- token-independence) / rowmean (per-row mean, the weakest
constant).
REGISTERED PREDICTIONS:
  (a) BIAS: the mean arm costs <= 0.10 nats overall (deletion is
      ~0.92) -- the head is a constant-bias adder;
  (b) TOKEN-INDEPENDENT: the cross-row mean arm costs <= 0.15;
  (c) sanity: the zero arm reproduces > 0.5."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sink_bias_test_results.json'
NR=32; LJ=5; HD=7
ARMS=['zero','mean','crossmean','rowmean']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={a:[0.0,0] for a in ARMS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        def run(mode):
            hs=[]
            if mode is not None:
                at=m.transformer.h[LJ].attn
                def fh(mo_,args,o_,at=at,mode=mode):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                    kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                    qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                    q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),
                                  (128,))
                    k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),
                                  (128,))
                    q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    if mode=='zero': z[:,HD]=0
                    elif mode=='mean':
                        z[:,HD]=z[:,HD].mean(dim=(0,1),
                                             keepdim=True)
                    elif mode=='rowmean':
                        z[:,HD]=z[:,HD].mean(dim=1,keepdim=True)
                    elif mode=='crossmean':
                        mu=z[:,HD].mean(dim=1,keepdim=True)
                        z[:,HD]=torch.roll(mu,1,dims=0) \
                            .expand_as(z[:,HD])
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h in hs: h.remove()
            return ce
        base=run(None)
        for a in ARMS:
            acc[a][0]+=run(a)-base; acc[a][1]+=1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()}
    print('dCE overall:',CE)
    pa=abs(CE['mean'])<=0.10
    pb=abs(CE['crossmean'])<=0.15
    pc=CE['zero']>0.5
    out={'dce_overall':CE,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    for nm,v in (('a','mean arm <=0.10 (constant bias)'),
                 ('b','cross-row mean <=0.15'),
                 ('c','zero arm >0.5 (sanity)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
