"""FOLD GAP LOCATE 2 -- 393 follow-up: heads 2.5 and 3.5 cap at
58/43% even with full block-0 context; their match codes accrete
writes from layers 1-2. Isolate WHICH component closes each
trigger with per-component arms. For head 2.5 (block-1 gap; all
arms start from the REAL block-1 input x1):
  A: lam-mix(x1) + m1_fold          (baseline = locate's arm3)
  B: lam-mix(x1) + a1_real + m1_fold  (a1's residual write added)
  C: lam-mix(x1) + m1_real            (m1's real write, a1's own
                                       residual dropped)
  D: real block-2 input               (sanity)
For head 3.5, the same four arms one block up (start from REAL
block-2 input; a2/m2 the variable components) -- the A-arm's gain
over locate's 43.3% separately attributes block-1 context.
REGISTERED PREDICTIONS:
  (a) sanity: D-arm hit >= 0.98 on both heads;
  (b) 2.5: max(B,C) hit >= 0.9 -- single-component closure; FORK
      recorded either way: B>C means the trigger reads a1's write
      directly, C>B means m1-mediated context;
  (c) 3.5: A-arm (real through block 1, m2 fold) >= 0.7 -- most of
      3.5's gap is block-1 context (387's side-term prediction);
  (d) monotone: A <= max(B,C) <= D on both heads."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fold_gap_locate2_results.json'
NR=16
CFG={'2.5':{'li':2,'hd':5,'blk':1},'3.5':{'li':3,'hd':5,'blk':2}}

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    res={k:{a:{'hit':0,'n':0,'corr':[]} for a in 'ABCD'}
         for k in CFG}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        binp={}; aout={}; mout={}; attin={}
        hs=[]
        for bl in (1,2,3):
            def bh(mo_,args,bl=bl): binp[bl]=args[0].detach().float()
            hs.append(m.transformer.h[bl]
                      .register_forward_pre_hook(bh))
        for bl in (1,2):
            def ah(mo,i_,o_,bl=bl): aout[bl]=o_[0].detach().float()
            def mh(mo,i_,o_,bl=bl): mout[bl]=o_.detach().float()
            hs.append(m.transformer.h[bl].attn
                      .register_forward_hook(ah))
            hs.append(m.transformer.h[bl].mlp
                      .register_forward_hook(mh))
        for k,c in CFG.items():
            def ph(mo_,args,k=k): attin[k]=args[0]
            hs.append(m.transformer.h[c['li']].attn
                      .register_forward_pre_hook(ph))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for k,c in CFG.items():
            li,hd,bl=c['li'],c['hd'],c['blk']
            at=m.transformer.h[li].attn
            blkb=m.transformer.h[bl]
            lam=blkb.lambdas.detach().float()
            xm=lam[0]*binp[bl]+lam[1]*E
            mfold=blkb.mlp(F.rms_norm(xm,(D,)).to(
                blkb.mlp.Down.weight.dtype)).float()
            # B keeps m_bl at fold; recomputing it on the
            # a-included input would make B identical to D
            xnext={'A':xm+mfold,
                   'B':xm+aout[bl]+mfold,
                   'C':xm+mout[bl],
                   'D':binp[bl+1]}
            lamn=m.transformer.h[li].lambdas.detach().float()
            X=attin[k]
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
            for a,xv in xnext.items():
                xr=xv
                for lj in range(bl+1,li):
                    bj=m.transformer.h[lj]
                    lj_lam=bj.lambdas.detach().float()
                    xr=lj_lam[0]*xr+lj_lam[1]*E
                    xr=xr+bj.mlp(F.rms_norm(xr,(D,)).to(
                        bj.mlp.Down.weight.dtype)).float()
                code=F.rms_norm(lamn[0]*xr+lamn[1]*E,(D,))
                fq1=F.rms_norm(code@at.c_q.weight.float()
                               [a9:b9].T,(128,))
                fk1=F.rms_norm(code@at.c_k.weight.float()
                               [a9:b9].T,(128,))
                fq2=F.rms_norm(code@at.c_q2.weight.float()
                               [a9:b9].T,(128,))
                fk2=F.rms_norm(code@at.c_k2.weight.float()
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
                        st=res[k][a]
                        kreal=int(pat[b,q,:q].abs().argmax())
                        kfold=int(fpat[b,q,:q].abs().argmax())
                        st['hit']+=int(kreal==kfold); st['n']+=1
                        if q>=16:
                            rr=pat[b,q,:q].abs().argsort() \
                                .argsort().float()
                            rf=fpat[b,q,:q].abs().argsort() \
                                .argsort().float()
                            st['corr'].append(float(
                                torch.corrcoef(torch.stack(
                                    [rr,rf]))[0,1]))
        print(f'batch {i} done',flush=True)
    outj={}
    for k,arms in res.items():
        outj[k]={a:{'hit':round(st['hit']/max(st['n'],1),3),
                    'corr':round(sum(st['corr'])
                                 /max(len(st['corr']),1),3)}
                 for a,st in arms.items()}
        print(f"{k}: "+" | ".join(
            f"{a} hit {outj[k][a]['hit']}" for a in 'ABCD'),
            flush=True)
    pa=all(outj[k]['D']['hit']>=0.98 for k in CFG)
    h25=outj['2.5']
    pb=max(h25['B']['hit'],h25['C']['hit'])>=0.9
    pc=outj['3.5']['A']['hit']>=0.7
    pd=all(outj[k]['A']['hit']<=max(outj[k]['B']['hit'],
           outj[k]['C']['hit'])<=outj[k]['D']['hit'] for k in CFG)
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'pred_d':bool(pd),
         'fork_2.5':'a1-direct' if h25['B']['hit']>h25['C']['hit']
                    else 'm1-mediated'}
    for nm,v in (('a','sanity D>=0.98 both'),
                 ('b','2.5 single-component closure >=0.9'),
                 ('c','3.5 A-arm >=0.7 (block-1 context)'),
                 ('d','monotone A<=max(B,C)<=D')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    print(f"2.5 fork: {out['fork_2.5']}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
