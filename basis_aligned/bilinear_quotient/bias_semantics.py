"""BIAS SEMANTICS -- 441: the sink chain is fully named -- at
position 0 the MLP chain (m0 0.44 -> m2 -> m3 -> m4) builds a
vector whose norm is 10.5x a normal write and whose DIRECTION is
the same across texts (mean pairwise cosine 0.998, and 0.998 even
between rows with different first tokens); head 5.7 broadcasts it
to every position; deleting it costs 0.92 nats. Remaining
question: what is the bias FOR? Read it in output space.
Method: take the head's mean write (the constant), pass it through
the final norm and unembedding, and look at which tokens it pushes
-- against a norm-matched random vector as the null. Then check
causally: with the head deleted, do exactly those tokens lose
probability?
REGISTERED PREDICTIONS:
  (a) CONCENTRATED: the bias's logit profile is far more
      concentrated than a norm-matched random vector -- top-20
      share of total |logit| mass at least 2x the random null;
  (b) COHERENT: the top-20 pushed tokens form a nameable class
      (reported verbatim, judged in the writeup, not by the
      script);
  (c) CAUSAL MATCH: deleting head 5.7 lowers the mean logit of
      the bias's top-20 pushed tokens by more than it lowers a
      random 20-token control (ratio >= 2)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_semantics_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    # 1) recover the constant: head 5.7's mean write
    means=[]
    logits_base=[]; logits_abl=[]
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        h=m.transformer.h[LJ].attn.register_forward_pre_hook(
            lambda mo_,args: cap.__setitem__('X',args[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        h.remove()
        lg0=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))
        at=m.transformer.h[LJ].attn; X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        def rot(w):
            return are(F.rms_norm(w(X).view(B,T,9,128),
                       (128,))[:,:,HD][:,:,None],cos,sin)[:,:,0]
        qf,kf=rot(at.c_q),rot(at.c_k)
        q2,k2=rot(at.c_q2),rot(at.c_k2)
        pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
             *torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())) \
            *torch.tril(torch.ones(T,T,device=DEV))
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()*(1-at.lamb)
        z=torch.einsum('bqk,bkd->bqd',pat,v)
        Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
        means.append((z@Wp.T).mean(dim=1))
        # ablated logits
        def fh(mo_,args,o_,at=at):
            y,v1r=o_
            X2=args[0]
            v1b=args[1] if args[1] is not None else v1r
            vv=at.c_v(X2).view(B,T,9,128)
            vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
            c2,s2=at.rotary(at.c_q(X2).view(B,T,9,128))
            qq=are(F.rms_norm(at.c_q(X2).view(B,T,9,128),(128,)),
                   c2,s2)
            kk=are(F.rms_norm(at.c_k(X2).view(B,T,9,128),(128,)),
                   c2,s2)
            q22=are(F.rms_norm(at.c_q2(X2).view(B,T,9,128),(128,)),
                    c2,s2)
            k22=are(F.rms_norm(at.c_k2(X2).view(B,T,9,128),(128,)),
                    c2,s2)
            sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                            kk.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                             k22.float())/128
            p2=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
            zz[:,HD]=0
            return (at.c_proj(zz.transpose(1,2).contiguous()
                    .view(B,T,-1).to(X2.dtype)),v1r)
        hh=at.register_forward_hook(fh)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hh.remove()
        lg1=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))
        logits_base.append(lg0.float().mean(dim=(0,1)).cpu())
        logits_abl.append(lg1.float().mean(dim=(0,1)).cpu())
        print(f'batch {i} done',flush=True)
    bias=torch.cat(means).mean(dim=0)
    lb=torch.stack(logits_base).mean(0)
    la=torch.stack(logits_abl).mean(0)
    # 2) read the bias in output space
    W=m.lm_head.weight.float()
    def profile(vec):
        lg=(F.rms_norm(vec[None],(D,))@W.T)[0].float()
        a=lg.abs()
        top=a.topk(20)
        return lg,float(top.values.sum()/a.sum())
    lgb,share=profile(bias.to(DEV))
    g=torch.Generator(device=DEV).manual_seed(9)
    nulls=[]
    for _ in range(5):
        r=torch.randn(D,generator=g,device=DEV)
        r=r/r.norm()*bias.norm().to(DEV)
        nulls.append(profile(r)[1])
    nullshare=sum(nulls)/len(nulls)
    topi=lgb.topk(20).indices.tolist()
    toptok=[cl.d1(t) for t in topi]
    boti=(-lgb).topk(10).indices.tolist()
    # 3) causal match
    d=(la-lb)
    drop_top=float(d[topi].mean())
    gg=torch.Generator().manual_seed(11)
    ctrl=torch.randperm(len(d),generator=gg)[:20]
    drop_ctrl=float(d[ctrl].mean())
    pa=share>=2*nullshare
    pc=abs(drop_top)>=2*abs(drop_ctrl)
    out={'bias_norm':round(float(bias.norm()),2),
         'top20_logit_share':round(share,4),
         'null_share':round(nullshare,4),
         'top20_tokens':toptok,
         'bottom10_tokens':[cl.d1(t) for t in boti],
         'delta_logit_top20':round(drop_top,4),
         'delta_logit_control20':round(drop_ctrl,4),
         'pred_a':bool(pa),'pred_b':True,'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'bias norm {out["bias_norm"]} | top-20 share '
          f'{share:.4f} vs null {nullshare:.4f}')
    print('pushes:',toptok)
    print('suppresses:',out['bottom10_tokens'])
    print(f'deleting the head: top-20 logits {drop_top:+.4f} vs '
          f'control {drop_ctrl:+.4f}')
    for nm,v in (('a','logit profile >=2x more concentrated'),
                 ('b','tokens reported for naming'),
                 ('c','causal match >=2x control')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
