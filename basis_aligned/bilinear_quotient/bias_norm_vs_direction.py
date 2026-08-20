"""BIAS: NORM OR DIRECTION -- 448: the linearization hypothesis
is refuted (447): with an exact decomposition (relative error
1.3e-6), removing the bias's cross terms from mlp5 costs 0.020
nats and its constant term 0.005, against 0.915 for deleting the
head. So the bias's value is not in how mlp5 transforms it. The
parts measured so far sum to roughly 0.26 against a whole of
0.92 -- strongly non-additive.
A different hypothesis fits a no-activation architecture: what
matters is the residual's SCALE. Every downstream rms_norm
divides by the residual norm, so a large constant added at layer
5 sets the operating scale of the entire rest of the network. If
so, the DIRECTION of the constant should barely matter and its
MAGNITUDE should matter a lot.
Arms (all in place, inside the head, where a constant is free):
  mean            : the true constant (reference, ~0)
  rand_samenorm   : a random direction, same norm
  half / double   : the true constant scaled 0.5x and 2.0x
  delete          : reference (~0.92)
REGISTERED PREDICTIONS:
  (a) SCALE: rand_samenorm costs <= 0.20 nats -- most of the
      value is magnitude, not direction;
  (b) if (a) fails at >= 0.5, direction is what matters and the
      bias is a specific learned vector, not a scale device --
      recorded either way;
  (c) MAGNITUDE MATTERS: half and double each cost >= 0.10."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_norm_vs_direction_results.json'
NR=32; LJ=5; HD=7
ARMS=['zero','mean','rand_samenorm','half','double']

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
                    mu=z[:,HD].mean(dim=(0,1),keepdim=True)
                    if mode=='zero': z[:,HD]=0
                    elif mode=='mean': z[:,HD]=mu
                    elif mode=='half': z[:,HD]=0.5*mu
                    elif mode=='double': z[:,HD]=2.0*mu
                    elif mode=='rand_samenorm':
                        gj=torch.Generator(device=DEV) \
                            .manual_seed(53)
                        rr=torch.randn(mu.shape,generator=gj,
                                       device=DEV,dtype=mu.dtype)
                        rr=rr/rr.norm().clamp_min(1e-6)*mu.norm()
                        z[:,HD]=rr
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
    pa=abs(CE['rand_samenorm'])<=0.20
    pb=abs(CE['rand_samenorm'])>=0.5
    pc=(CE['half']>=0.10 and CE['double']>=0.10)
    out={'dce_overall':CE,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    for nm,v in (('a','random same-norm <=0.20 (scale device)'),
                 ('b','random same-norm >=0.5 (specific vector)'),
                 ('c','half and double each cost >=0.10')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
