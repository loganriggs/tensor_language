"""FOLD GAP LOCATE -- 392: the fold-trigger arc capped at
66/53/36% early-band hits over three registered attempts; the
token-computable residual is exhausted. Localize the missing
information with an arm ladder (all arms use rotary at real
positions; patterns compared to the real head pattern):
  arm1 = v3 fold code (attention-free forward; baseline);
  arm2 = REAL m0 write + lambda-mixed wte, a0's own write dropped,
         mlp folds above block 0 (tests: is the gap m0's
         contextual INPUT rather than a0's residual write?);
  arm3 = real residual through block 0 (a0 kept), attention-free
         above (for layer-1 heads this is exact by construction --
         sanity anchor; for layers 2-3 it isolates a1+/m1+ noise).
REGISTERED PREDICTIONS:
  (a) arm2 early-band ((1,4),(2,5),(3,5)) hit >=80% on >=2 of 3
      (the gap is mostly m0's contextual enrichment);
  (b) arm2 rank corr >=0.6 on >=2 of 3;
  (c) monotone ladder: arm1 <= arm2 <= arm3 hit on every early
      head;
  (d) sanity: arm3 hit >=0.98 for head (1,4)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fold_gap_locate_results.json'
HEADS=[(1,4),(2,5),(3,5)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    maxli=max(l for l,_ in HEADS)
    res={f'{li}.{hd}':{a:{'hit':0,'n':0,'corr':[]} for a in
         ('arm1','arm2','arm3')} for li,hd in HEADS}
    cap={}
    def mkpre(li):
        def h(mo_,args): cap[li]=(args[0],args[1])
        return h
    m0out={}
    def m0h(mo,i_,o_): m0out['y']=o_.detach().float()
    blk1in={}
    def b1h(mo_,args): blk1in['x']=args[0].detach().float()
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li,_ in HEADS]
        hs.append(m.transformer.h[0].mlp.register_forward_hook(m0h))
        hs.append(m.transformer.h[1].register_forward_pre_hook(b1h))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        lam0=m.transformer.h[0].lambdas.detach().float()
        xm0=(lam0[0]+lam0[1])*E                 # block0 pre-attn mix
        starts={'arm1':xm0+m.transformer.h[0].mlp(
                    F.rms_norm(xm0,(D,)).to(x.dtype)).float(),
                'arm2':xm0+m0out['y'],
                'arm3':blk1in['x']}
        # attention-free continuation per arm: codes[arm][li]
        codes={a:{} for a in starts}
        for a,xa in starts.items():
            xr=xa.clone()
            for li in range(1,maxli+1):
                blk=m.transformer.h[li]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                codes[a][li]=F.rms_norm(xr,(D,))
                xr=xr+blk.mlp(F.rms_norm(xr,(D,)).to(
                    blk.mlp.Down.weight.dtype)).float()
        for li,hd in HEADS:
            at=m.transformer.h[li].attn
            X,_=cap[li]
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
            for a in starts:
                c=codes[a][li]
                fq1=F.rms_norm(c@at.c_q.weight.float()[a9:b9].T,
                               (128,))
                fk1=F.rms_norm(c@at.c_k.weight.float()[a9:b9].T,
                               (128,))
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
                        st=res[f'{li}.{hd}'][a]
                        kreal=int(pat[b,q,:q].abs().argmax())
                        kfold=int(fpat[b,q,:q].abs().argmax())
                        st['hit']+=int(kreal==kfold); st['n']+=1
                        if q>=16:
                            rr=pat[b,q,:q].abs().argsort() \
                                .argsort().float()
                            rf=fpat[b,q,:q].abs().argsort() \
                                .argsort().float()
                            st['corr'].append(float(torch.corrcoef(
                                torch.stack([rr,rf]))[0,1]))
        print(f'batch {i} done',flush=True)
    outj={}
    for k,arms in res.items():
        outj[k]={}
        for a,st in arms.items():
            outj[k][a]={'hit':round(st['hit']/max(st['n'],1),3),
                        'corr':round(sum(st['corr'])
                                     /max(len(st['corr']),1),3)}
        print(f"{k}: "+" | ".join(
            f"{a} hit {outj[k][a]['hit']} corr {outj[k][a]['corr']}"
            for a in ('arm1','arm2','arm3')),flush=True)
    early=['1.4','2.5','3.5']
    pa=sum(outj[k]['arm2']['hit']>=0.8 for k in early)>=2
    pb=sum(outj[k]['arm2']['corr']>=0.6 for k in early)>=2
    pc=all(outj[k]['arm1']['hit']<=outj[k]['arm2']['hit']
           <=outj[k]['arm3']['hit'] for k in early)
    pd=outj['1.4']['arm3']['hit']>=0.98
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'pred_d':bool(pd)}
    for nm,v in (('a','arm2 hit>=80% on >=2'),
                 ('b','arm2 corr>=0.6 on >=2'),
                 ('c','monotone ladder all'),
                 ('d','arm3 sanity 1.4 >=0.98')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
