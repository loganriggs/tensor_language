"""BIAS INJECTION DEPTH -- 444: with the scale bug fixed (443),
injecting head 5.7's constant at block 6 costs +1.017 nats and at
the final residual +1.375 -- both WORSE than simply deleting the
head (+0.915) -- while sink_bias_test showed that replacing the
head's write with the same constant IN PLACE (inside layer 5,
before mlp5 runs) is free (-0.005). So the bias must be present
before layer 5's MLP consumes it; delivered even one sublayer
late it is worse than absent. Measure that decay precisely: inject
the constant at increasing depth and price each.
Injection points: in-place (inside the head, the free reference),
after mlp5, after block 6, after block 9, after block 13, final
residual. Head deleted in every arm except the reference.
REGISTERED PREDICTIONS:
  (a) MONOTONE DECAY: cost rises monotonically with injection
      depth (in-place < after-mlp5 < block6 < block9 < block13 <
      final);
  (b) EARLY CONSUMPTION: the after-mlp5 arm already costs >= 0.5
      nats, i.e. most of the bias's value is consumed by layer
      5's own MLP;
  (c) sanity: the in-place arm reproduces ~0 and deletion ~0.92."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_injection_depth_results.json'
NR=32

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    prev=json.load(open(PT+'bias_semantics_results.json'))
    toptok=prev['top20_tokens']
    W=m.lm_head.weight.float()
    enc=cl.enc()
    topi=[enc.encode_single_token(t) if hasattr(
              enc,'encode_single_token') else None
          for t in toptok]
    topi=[t for t in topi if t is not None]
    POINTS=['inplace','after_mlp5','block6','block9','block13','final']
    acc={a:[0.0,0] for a in ['delete']+POINTS}
    lg_acc={a:0.0 for a in ('full','delete','direct')}
    nb=0
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4; cap={}
        h=m.transformer.h[LJ].attn.register_forward_pre_hook(
            lambda mo_,args: cap.__setitem__('X',args[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        h.remove()
        # the head's own write, and its per-batch constant
        at=m.transformer.h[LJ].attn; X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        def rot(w):
            return are(F.rms_norm(w(X).view(B,T,9,128),
                       (128,))[:,:,HD][:,:,None],cos,sin)[:,:,0]
        qf,kf=rot(at.c_q),rot(at.c_k); q2,k2=rot(at.c_q2),rot(at.c_k2)
        # /128 per QK factor, as the model does (omitted in the
        # first version: made the injected constant 16384x too
        # large, writeup 443)
        pat=((torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())/128)
             *(torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())/128)) \
            *torch.tril(torch.ones(T,T,device=DEV))
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()*(1-at.lamb)
        Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
        const=(torch.einsum('bqk,bkd->bqd',pat,v)@Wp.T) \
            .mean(dim=(0,1))            # one vector
        def run(mode):
            hs=[]
            if mode is not None:
                def fh(mo_,args,o_,at=at,mode=mode):
                    y,v1r=o_
                    X2=args[0]
                    v1b=args[1] if args[1] is not None else v1r
                    vv=at.c_v(X2).view(B,T,9,128)
                    vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
                    c2,s2=at.rotary(at.c_q(X2).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X2).view(B,T,9,128),(128,)),c2,s2)
                    qq,kk=r2(at.c_q),r2(at.c_k)
                    q22,k22=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                                    kk.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                                     k22.float())/128
                    p2=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
                    if mode=='inplace':
                        zz[:,HD]=zz[:,HD].mean(dim=(0,1),
                                               keepdim=True)
                    else:
                        zz[:,HD]=0
                    return (at.c_proj(zz.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X2.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
                if mode in ('block6','block9','block13'):
                    bl={'block6':6,'block9':9,'block13':13}[mode]
                    def ph(mo_,args):
                        return (args[0]+const.to(args[0].dtype),
                                )+tuple(args[1:])
                    hs.append(m.transformer.h[bl]
                              .register_forward_pre_hook(ph))
                elif mode=='after_mlp5':
                    def mh(mo,i_,o_):
                        return o_+const.to(o_.dtype)
                    hs.append(m.transformer.h[LJ].mlp
                              .register_forward_hook(mh))
            xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0b=xx
            v1b=None
            for blk in m.transformer.h: xx,v1b=blk(xx,v1b,x0b)
            if mode=='final': xx=xx+const.to(xx.dtype)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(xx,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h_ in hs: h_.remove()
            return ce,lg.mean(dim=(0,1)).cpu()
        base,lgf=run(None)
        for a in ['delete']+POINTS:
            c,lg=run(a)
            acc[a][0]+=c-base; acc[a][1]+=1
            if a in ('delete','final'):
                lg_acc[a if a=='delete' else 'direct'] \
                    +=float(lg[topi].mean())
        lg_acc['full']+=float(lgf[topi].mean()); nb+=1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()}
    LG={a:round(v/max(nb,1),4) for a,v in lg_acc.items()}
    seq=[CE[p] for p in POINTS]
    pa=all(seq[i]<=seq[i+1]+1e-4 for i in range(len(seq)-1))
    pb=CE['after_mlp5']>=0.5
    pc=(abs(CE['inplace'])<=0.05 and CE['delete']>0.5)
    out={'dce':CE,'order':POINTS,'top20_mean_logit':LG,
         'n_top_tokens':len(topi),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('dCE by injection depth:',
          {p:CE[p] for p in POINTS},'| delete',CE['delete'])
    for nm,v in (('a','cost rises monotonically with depth'),
                 ('b','after-mlp5 already >=0.5 nats'),
                 ('c','in-place ~0 and deletion ~0.92')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
