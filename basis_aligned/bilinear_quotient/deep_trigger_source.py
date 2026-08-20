"""DEEP TRIGGER SOURCE -- 397: the MLP ladder carries the early
induction triggers (0.998/0.859/0.741) but not the deep ones (5.5
at 0.362, 8.4 at 0.504): the deep match code contains
attention-carried content. Localize it: for each deep head, add
ONE attention layer's real residual write to the ladder at a time
(x~ chain: lambda mixes + all real MLP writes + a_j real write at
block j only) and measure the lift over the plain ladder.
REGISTERED PREDICTIONS:
  (a) for head 8.4, j=4 gives the largest single-layer lift
      (387's m0|a4 side term named a4 as the deep match input);
  (b) CONCENTRATION fork (report either way): for at least one
      deep head, the best single a_j recovers >=50% of the
      ladder-to-real gap (concentrated carrier); else the deep
      code is diffusely attention-written;
  (c) early-head control: for 2.5, no single a_j lifts hit by
      more than 0.05 (its residual gap is not a single attention
      layer)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'deep_trigger_source_results.json'
HEADS=[(2,5),(5,5),(8,4)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    maxli=max(l for l,_ in HEADS)
    arms={f'{li}.{hd}':['ladder']+[f'+a{j}' for j in range(li)]
          for li,hd in HEADS}
    res={k:{a:{'hit':0,'n':0} for a in v} for k,v in arms.items()}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        aout={}; mout={}; attin={}
        hs=[]
        for lj in range(maxli):
            def ah(mo,i_,o_,lj=lj): aout[lj]=o_[0].detach().float()
            def mh(mo,i_,o_,lj=lj): mout[lj]=o_.detach().float()
            hs.append(m.transformer.h[lj].attn
                      .register_forward_hook(ah))
            hs.append(m.transformer.h[lj].mlp
                      .register_forward_hook(mh))
        for li,hd in HEADS:
            def ph(mo_,args,li=li): attin[li]=args[0]
            hs.append(m.transformer.h[li].attn
                      .register_forward_pre_hook(ph))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        def chain(addj,upto):
            xr=E.clone()
            code=None
            for lj in range(upto+1):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                if lj==upto:
                    return F.rms_norm(xr,(D,))
                if lj==addj: xr=xr+aout[lj]
                xr=xr+mout[lj]
        for li,hd in HEADS:
            at=m.transformer.h[li].attn
            X=attin[li]
            a9,b9=hd*128,(hd+1)*128
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            tril=torch.tril(torch.ones(T,T,device=DEV))
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float()))*tril
            for a in arms[f'{li}.{hd}']:
                addj=-1 if a=='ladder' else int(a[2:])
                c=chain(addj,li)
                fq1=F.rms_norm(c@at.c_q.weight.float()
                               [a9:b9].T,(128,))
                fk1=F.rms_norm(c@at.c_k.weight.float()
                               [a9:b9].T,(128,))
                fq2=F.rms_norm(c@at.c_q2.weight.float()
                               [a9:b9].T,(128,))
                fk2=F.rms_norm(c@at.c_k2.weight.float()
                               [a9:b9].T,(128,))
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
                        st=res[f'{li}.{hd}'][a]
                        kreal=int(pat[b,q,:q].abs().argmax())
                        kfold=int(fpat[b,q,:q].abs().argmax())
                        st['hit']+=int(kreal==kfold); st['n']+=1
        print(f'batch {i} done',flush=True)
    outj={}
    for k,armd in res.items():
        outj[k]={a:round(st['hit']/max(st['n'],1),3)
                 for a,st in armd.items()}
        base=outj[k]['ladder']
        lifts={a:round(v-base,3) for a,v in outj[k].items()
               if a!='ladder'}
        outj[k]['lifts']=lifts
        best=max(lifts,key=lifts.get)
        outj[k]['best']=best
        print(f"{k}: ladder {base} | best {best} "
              f"+{lifts[best]} | lifts {lifts}",flush=True)
    pa=outj['8.4']['best']=='+a4'
    conc={}
    for k in ('5.5','8.4'):
        gap=1.0-outj[k]['ladder']
        conc[k]=max(outj[k]['lifts'].values())/max(gap,1e-6)
    pb=any(v>=0.5 for v in conc.values())
    pc=max(outj['2.5']['lifts'].values())<=0.05
    out={'heads':outj,'gap_recovered':
         {k:round(v,3) for k,v in conc.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    for nm,v in (('a','8.4 best lift is a4'),
                 ('b','>=50% gap from one a_j (some deep head)'),
                 ('c','2.5 control: no a_j lift >0.05')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
