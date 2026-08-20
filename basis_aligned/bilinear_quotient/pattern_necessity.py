"""PATTERN NECESSITY -- 424: swapping a6.h3's read pattern for a
sibling head's costs nothing (-0.002 nats) while swapping its
values costs as much as deletion (423). Two readings again:
either read POSITIONS are functionally irrelevant for this head,
or sibling patterns at the same layer are so similar that the
swap was a weak perturbation. Decide it with patterns that are
definitely wrong: uniform-over-prefix, within-row reversed, and
a pattern taken from a DIFFERENT ROW (wrong context entirely).
Values always the head's own; primary metric dCE (the 422 audit
retired argmax shift for cross-arm work).
REGISTERED PREDICTIONS:
  (a) DECISIVE: the uniform-pattern arm costs <= 0.02 nats at
      match positions -- read positions are functionally
      irrelevant for a6.h3;
  (b) the cross-row pattern arm also costs <= 0.02 (even
      wrong-context positions are fine);
  (c) DIAGNOSTIC (no bar, decides how to read 423): mean rank
      correlation between h3's and h0's patterns -- if >= 0.5 the
      423 swap was weak and its conclusion needs that caveat;
  (d) sanity: the zero arm reproduces ~0.05 at match positions."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'pattern_necessity_results.json'
NR=32
SIBCORR=[]
PAIRS=[(6,3,0)]
ARMS=['zero','patswap','unif','rev','crossrow']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={f'{lj}.{h1}': {a:{'m':0.0,'nm':0,'all':0.0,'nall':0}
         for a in ARMS} for lj,h1,_ in PAIRS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        mmask=torch.zeros(B,T,dtype=torch.bool)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                ism=t in last and last[t]+1<q
                last[t]=q
                if ism and q>=8: mmask[b,q]=True
        def run(lj=None,h1=None,h0=None,mode=None):
            hs=[]
            if mode is not None:
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,h1=h1,h0=h0,mode=mode):
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
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    if mode=='zero': z[:,h1]=0
                    elif mode=='patswap':
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pat[:,h0],vm[:,:,h1].float())
                    elif mode=='unif':
                        u=tril/tril.sum(-1,keepdim=True)
                        sc9=pat[:,h1].abs().sum(-1,keepdim=True)
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            u[None]*sc9,vm[:,:,h1].float())
                    elif mode=='rev':
                        pr=torch.flip(pat[:,h1],dims=[-1])
                        pr=torch.roll(pr,shifts=1,dims=-1)*tril
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pr,vm[:,:,h1].float())
                    elif mode=='crossrow':
                        pc=pat[:,h1][torch.tensor([1,2,3,0])]
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pc,vm[:,:,h1].float())
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
                               reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
            return ce
        ce0=run()
        if not SIBCORR:
            atx=m.transformer.h[6].attn
            capx={}
            hx=atx.register_forward_pre_hook(
                lambda mo_,args: capx.__setitem__('X',args[0]))
            xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0x=xx
            v1x=None
            for blk in m.transformer.h: xx,v1x=blk(xx,v1x,x0x)
            hx.remove()
            X=capx['X']
            cosx,sinx=atx.rotary(atx.c_q(X).view(B,T,9,128))
            def pf(hh):
                qf=F.rms_norm(atx.c_q(X).view(B,T,9,128),
                              (128,))[:,:,hh]
                kf=F.rms_norm(atx.c_k(X).view(B,T,9,128),
                              (128,))[:,:,hh]
                q2=F.rms_norm(atx.c_q2(X).view(B,T,9,128),
                              (128,))[:,:,hh]
                k2=F.rms_norm(atx.c_k2(X).view(B,T,9,128),
                              (128,))[:,:,hh]
                qf=are(qf[:,:,None],cosx,sinx)[:,:,0]
                kf=are(kf[:,:,None],cosx,sinx)[:,:,0]
                q2=are(q2[:,:,None],cosx,sinx)[:,:,0]
                k2=are(k2[:,:,None],cosx,sinx)[:,:,0]
                return ((torch.einsum('bqd,bkd->bqk',qf.float(),
                    kf.float())*torch.einsum('bqd,bkd->bqk',
                    q2.float(),k2.float()))
                    *torch.tril(torch.ones(T,T,device=DEV)))
            p3,p0=pf(3),pf(0)
            cs=[]
            for b in range(B):
                for q in range(16,T,8):
                    r3=p3[b,q,:q].abs().argsort().argsort().float()
                    r0=p0[b,q,:q].abs().argsort().argsort().float()
                    cs.append(float(torch.corrcoef(
                        torch.stack([r3,r0]))[0,1]))
            SIBCORR.append(round(sum(cs)/len(cs),3))
        for lj,h1,h0 in PAIRS:
            k=f'{lj}.{h1}'
            for a in ARMS:
                d=run(lj,h1,h0,a)-ce0
                acc[k][a]['m']+=float(d[mmask].sum())
                acc[k][a]['nm']+=int(mmask.sum())
                acc[k][a]['all']+=float(d.sum())
                acc[k][a]['nall']+=d.numel()
        print(f'batch {i} done',flush=True)
    out={}
    for k,arms in acc.items():
        out[k]={a:{'dce_match':round(v['m']/max(v['nm'],1),4),
                   'dce_all':round(v['all']/max(v['nall'],1),4)}
                for a,v in arms.items()}
        print(f"{k}: "+" | ".join(
            f"{a} match {out[k][a]['dce_match']} all "
            f"{out[k][a]['dce_all']}" for a in ARMS),flush=True)
    c3=out['6.3']
    pa=abs(c3['unif']['dce_match'])<=0.02
    pb=abs(c3['crossrow']['dce_match'])<=0.02
    pc=True   # diagnostic, reported below
    pd=c3['zero']['dce_match']>0.02
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c_diagnostic':True,'pred_d':bool(pd),
                'sibling_pattern_rankcorr':SIBCORR[0]})
    print(f"sibling pattern rank corr (h3 vs h0): {SIBCORR[0]}")
    for nm,v in (('a','uniform pattern <=0.02 (positions irrelevant)'),
                 ('b','cross-row pattern <=0.02'),
                 ('c_diagnostic','sibling corr reported'),
                 ('d','zero arm reproduces >0.02')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
