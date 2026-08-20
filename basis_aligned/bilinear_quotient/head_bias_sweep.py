"""HEAD BIAS SWEEP -- 435: head 5.7 costs +0.915 nats to delete
but replacing its write with a CONSTANT (its own mean, or a mean
taken from other rows) costs -0.005 -- free, marginally better
than intact. The model's most expensive attention head is a
learned bias generator: 1152 numbers replace it exactly. How
common is that? Sweep all 162 heads: deletion cost vs
mean-replacement cost, same rows, same metric.
REGISTERED PREDICTIONS:
  (a) BIAS ADDERS EXIST BROADLY: >= 10 heads have deletion cost
      >= 0.02 nats while their mean-replacement cost is <= 0.005
      (their entire contribution is a constant);
  (b) 5.7 is the extreme: the largest (deletion - mean) gap in
      the model;
  (c) report the layer profile of the bias-adder set, and the
      total nats recoverable by replacing every such head with
      its constant."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_bias_sweep_results.json'
NR=16
ARMS=['zero','mean']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={f'{lj}.{h}':{a:[0.0,0] for a in ARMS}
         for lj in range(18) for h in range(9)}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        def run(mode,LJ=None,HD=None):
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
        for lj in range(18):
            for hd in range(9):
                k=f'{lj}.{hd}'
                for a in ARMS:
                    acc[k][a][0]+=run(a,lj,hd)-base
                    acc[k][a][1]+=1
        print(f'batch {i} done ({time.time()-t0:.0f}s)',flush=True)
    CE={k:{a:round(v[a][0]/max(v[a][1],1),5) for a in ARMS}
        for k,v in acc.items()}
    bias=[k for k in CE if CE[k]['zero']>=0.02
          and abs(CE[k]['mean'])<=0.005]
    gaps={k:round(CE[k]['zero']-CE[k]['mean'],5) for k in CE}
    top=max(gaps,key=gaps.get)
    import collections
    prof=dict(sorted(collections.Counter(
        int(k.split('.')[0]) for k in bias).items()))
    recover=round(sum(CE[k]['zero']-CE[k]['mean'] for k in bias),4)
    pa=len(bias)>=10
    pb=(top=='5.7')
    out={'dce':CE,'bias_adders':sorted(bias),
         'n_bias_adders':len(bias),'layer_profile':prof,
         'largest_gap_head':top,'largest_gap':gaps[top],
         'nats_recoverable':recover,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'runtime_s':time.time()-t0}
    print(f'bias adders: {len(bias)} heads {sorted(bias)}')
    print(f'layer profile {prof} | largest gap {top} '
          f'{gaps[top]} | nats recoverable {recover}')
    for nm,v in (('a','>=10 bias-adder heads'),
                 ('b','5.7 has the largest gap'),
                 ('c','profile reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
