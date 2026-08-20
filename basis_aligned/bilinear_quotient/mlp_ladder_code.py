"""MLP LADDER CODE -- 396: locate2 resolved m1-mediated (a1's
residual write adds nothing to 2.5's trigger; m1's real write
closes most of it), and 393 showed a0's write irrelevant at 1.4.
Generalization under test: THE INDUCTION MATCH CODE IS A
CUMULATIVE MLP LADDER -- reconstruct the residual with ALL
attention residual writes removed but every real MLP write kept
(x~_{l+1} = lambda-mix_l(x~_l) + m_l_real; m_l_real captured from
the intact forward, so attention still contextualizes MLP inputs),
and predict each head's pattern from it. Inverse control: keep
attention writes, drop MLP writes.
REGISTERED PREDICTIONS:
  (a) ladder hit >=0.85 for 2.5 and >=0.75 for 3.5 (beats
      locate2's arm C: the ladder carries the code at least as
      well as block-local surgery);
  (b) DEPTH REACH fork (report either way): deep heads 5.5, 8.4
      ladder hit >=0.5 -- if HELD the entire induction trigger
      story is one sentence (MLP-chain identity code); if FAILED
      the deep match code needs attention-carried content;
  (c) inverse control: attention-only residual (MLP writes
      dropped) hits <0.2 on all heads;
  (d) 1.4 ladder hit >=0.99 (393 consistency)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp_ladder_code_results.json'
HEADS=[(1,4),(2,5),(3,5),(5,5),(8,4)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    maxli=max(l for l,_ in HEADS)
    res={f'{li}.{hd}':{a:{'hit':0,'n':0} for a in
         ('ladder','attonly')} for li,hd in HEADS}
    corr={f'{li}.{hd}':[] for li,hd in HEADS}
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
        # reconstruct residuals: ladder (real MLP writes, no attn
        # writes) and attonly (real attn writes, no MLP writes)
        codes={'ladder':{},'attonly':{}}
        xl=E.clone(); xa=E.clone()
        for lj in range(maxli+1):
            blk=m.transformer.h[lj]
            lam=blk.lambdas.detach().float()
            xl=lam[0]*xl+lam[1]*E
            xa=lam[0]*xa+lam[1]*E
            codes['ladder'][lj]=F.rms_norm(xl,(D,))
            codes['attonly'][lj]=F.rms_norm(xa,(D,))
            if lj<maxli:
                xl=xl+mout[lj]
                xa=xa+aout[lj]
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
            for a in ('ladder','attonly'):
                c=codes[a][li]
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
                        if a=='ladder' and q>=16:
                            rr=pat[b,q,:q].abs().argsort() \
                                .argsort().float()
                            rf=fpat[b,q,:q].abs().argsort() \
                                .argsort().float()
                            corr[f'{li}.{hd}'].append(float(
                                torch.corrcoef(torch.stack(
                                    [rr,rf]))[0,1]))
        print(f'batch {i} done',flush=True)
    outj={}
    for k,arms in res.items():
        outj[k]={a:round(st['hit']/max(st['n'],1),3)
                 for a,st in arms.items()}
        outj[k]['ladder_corr']=round(
            sum(corr[k])/max(len(corr[k]),1),3)
        print(f"{k}: ladder {outj[k]['ladder']} "
              f"(corr {outj[k]['ladder_corr']}) | "
              f"attonly {outj[k]['attonly']}",flush=True)
    pa=outj['2.5']['ladder']>=0.85 and outj['3.5']['ladder']>=0.75
    pb=outj['5.5']['ladder']>=0.5 and outj['8.4']['ladder']>=0.5
    pc=all(outj[k]['attonly']<0.2 for k in outj)
    pd=outj['1.4']['ladder']>=0.99
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'pred_d':bool(pd)}
    for nm,v in (('a','2.5>=0.85 and 3.5>=0.75'),
                 ('b','deep reach >=0.5 (fork)'),
                 ('c','attn-only control <0.2 all'),
                 ('d','1.4>=0.99 consistency')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
