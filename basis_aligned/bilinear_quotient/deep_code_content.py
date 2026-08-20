"""DEEP CODE CONTENT -- 398 follow-up: head 5.5's match code is
ladder + a4 (0.362 -> 0.860, one layer = 78% of the gap). Is a4
writing NEW content, or RELAYING ladder content from other
positions? Recompute a4's write with its real patterns but VALUES
read from the ladder residual (attention as a spatial relay of
mlp-ladder code); also per-head restriction of a4's write.
Arms for head 5.5's trigger (all = ladder + variant-a4-write):
  full     : a4's real write (reproduces 0.860);
  h0..h8   : a4's write restricted to one of its 9 heads;
  ladval   : a4 recomputed -- real patterns, values from the
             ladder residual at layer 4 (real v1 kept).
REGISTERED PREDICTIONS:
  (a) sanity: full arm within 0.02 of 0.860;
  (b) one a4 head carries >=70% of the full arm's lift over
      ladder (the relay is a specific head);
  (c) ladval recovers >=80% of the full arm's lift -- the deep
      match code is STILL mlp-ladder content, spatially relayed
      by attention (if HELD, the induction code story closes:
      one code, built by MLPs, moved by attention)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'deep_code_content_results.json'
LI,HD=5,5; AJ=4
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    ARMS=['full']+[f'h{h}' for h in range(9)]+['ladval']
    res={a:{'hit':0,'n':0} for a in ARMS}
    res['ladder']={'hit':0,'n':0}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        aout={}; mout={}; attin={}; a4args={}
        hs=[]
        for lj in range(LI):
            def ah(mo,i_,o_,lj=lj): aout[lj]=o_[0].detach().float()
            def mh(mo,i_,o_,lj=lj): mout[lj]=o_.detach().float()
            hs.append(m.transformer.h[lj].attn
                      .register_forward_hook(ah))
            hs.append(m.transformer.h[lj].mlp
                      .register_forward_hook(mh))
        def p4(mo_,args): a4args['X']=args[0]; a4args['v1']=args[1]
        hs.append(m.transformer.h[AJ].attn
                  .register_forward_pre_hook(p4))
        def p5(mo_,args): attin['X']=args[0]
        hs.append(m.transformer.h[LI].attn
                  .register_forward_pre_hook(p5))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        # ladder residual chain up to entry of block AJ and LI
        def ladder_to(upto):
            xr=E.clone()
            for lj in range(upto):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                xr=xr+mout[lj]
            return xr
        # a4 write variants
        at4=m.transformer.h[AJ].attn
        X4=a4args['X']; v14=a4args['v1']
        B=X4.shape[0]
        lam4=m.transformer.h[AJ].lambdas.detach().float()
        xl4=ladder_to(AJ)
        Xl4=F.rms_norm(lam4[0]*xl4+lam4[1]*E,(D,)) \
            .to(X4.dtype)
        def a4_write(vsrc):
            v=at4.c_v(vsrc).view(B,T,9,128)
            vm=(1-at4.lamb)*v+at4.lamb*v14.view_as(v)
            cos,sin=at4.rotary(at4.c_q(X4).view(B,T,9,128))
            qf=F.rms_norm(at4.c_q(X4).view(B,T,9,128),(128,))
            kf=F.rms_norm(at4.c_k(X4).view(B,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at4.c_q2(X4).view(B,T,9,128),(128,))
            k2=F.rms_norm(at4.c_k2(X4).view(B,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            patm=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',patm,vm.float())
            return z          # (B,9,T,128)
        z_real=a4_write(X4)
        z_lad=a4_write(Xl4)
        def proj(z):
            return at4.c_proj(z.transpose(1,2).contiguous()
                              .view(B,T,-1).to(X4.dtype)).float()
        writes={'full':proj(z_real),'ladval':proj(z_lad)}
        for h in range(9):
            zh=torch.zeros_like(z_real); zh[:,h]=z_real[:,h]
            writes[f'h{h}']=proj(zh)
        writes['ladder']=None
        # code at LI: ladder chain with a4 variant inserted
        codes={}
        for a,w in writes.items():
            xr=E.clone()
            for lj in range(LI):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                if lj==AJ and w is not None: xr=xr+w
                xr=xr+mout[lj]
            lam5=m.transformer.h[LI].lambdas.detach().float()
            codes[a]=F.rms_norm(lam5[0]*xr+lam5[1]*E,(D,))
        at=m.transformer.h[LI].attn
        X=attin['X']
        a9,b9=HD*128,(HD+1)*128
        cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
        qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,HD]
        kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,HD]
        q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,HD]
        k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,HD]
        qf=are(qf[:,:,None],cos,sin)[:,:,0]
        kf=are(kf[:,:,None],cos,sin)[:,:,0]
        q2=are(q2[:,:,None],cos,sin)[:,:,0]
        k2=are(k2[:,:,None],cos,sin)[:,:,0]
        tril=torch.tril(torch.ones(T,T,device=DEV))
        pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
             *torch.einsum('bqd,bkd->bqk',q2.float(),
                           k2.float()))*tril
        for a,c in codes.items():
            fq1=F.rms_norm(c@at.c_q.weight.float()[a9:b9].T,(128,))
            fk1=F.rms_norm(c@at.c_k.weight.float()[a9:b9].T,(128,))
            fq2=F.rms_norm(c@at.c_q2.weight.float()[a9:b9].T,
                           (128,))
            fk2=F.rms_norm(c@at.c_k2.weight.float()[a9:b9].T,
                           (128,))
            fq1=are(fq1[:,:,None],cos,sin)[:,:,0]
            fk1=are(fk1[:,:,None],cos,sin)[:,:,0]
            fq2=are(fq2[:,:,None],cos,sin)[:,:,0]
            fk2=are(fk2[:,:,None],cos,sin)[:,:,0]
            fpat=(torch.einsum('bqd,bkd->bqk',fq1,fk1)
                  *torch.einsum('bqd,bkd->bqk',fq2,fk2))*tril
            for b in range(4):
                toks=ROWS[i+b,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]
                    ism=t in last and last[t]+1<q
                    last[t]=q
                    if not ism or q<8: continue
                    st=res[a]
                    kreal=int(pat[b,q,:q].abs().argmax())
                    kfold=int(fpat[b,q,:q].abs().argmax())
                    st['hit']+=int(kreal==kfold); st['n']+=1
        print(f'batch {i} done',flush=True)
    outj={a:round(st['hit']/max(st['n'],1),3)
          for a,st in res.items()}
    base=outj['ladder']; lift=outj['full']-base
    hl={f'h{h}':round(outj[f'h{h}']-base,3) for h in range(9)}
    besth=max(hl,key=hl.get)
    pa=abs(outj['full']-0.860)<=0.02
    pb=hl[besth]>=0.7*max(lift,1e-6)
    pc=(outj['ladval']-base)>=0.8*max(lift,1e-6)
    out={'arms':outj,'head_lifts':hl,'best_head':besth,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"ladder {base} full {outj['full']} "
          f"ladval {outj['ladval']} | head lifts {hl}")
    for nm,v in (('a','full within 0.02 of 0.860'),
                 ('b','one head >=70% of lift'),
                 ('c','ladval >=80% of lift (relay)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
