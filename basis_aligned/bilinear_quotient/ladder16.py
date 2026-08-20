"""LADDER 16 -- 411: r.3.0's late heads (16.8, 16.2) compare
m0|m0 -- the same identity code as the induction band (410's
decomp, near-unanimous argmax). Test whether the bounded-relay
reconstruction predicts their patterns at layer 16: chain_k as in
402 (chain_0 = pure MLP ladder; chain_k relays with values from
chain_{k-1}), evaluated at every 8th query position (r.3.0 reads
are not match-conditioned), with a match/non-match split reported.
REGISTERED PREDICTIONS:
  (a) k=3 hit >= 0.70 for both heads (the one-code story reaches
      layer 16);
  (b) monotone in k for both heads;
  (c) shuffled-value null at k=1: lift < 25% of true k=1 lift;
  (d) report: hit at seen-before positions vs fresh positions
      (are these heads seen-before-conditional?)."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ladder16_results.json'
NR=16; KMAX=3
HEADS=[(16,8),(16,2)]
EARLY=[]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    maxli=max(l for l,_ in HEADS)
    ARMS=[f'k{k}' for k in range(KMAX+1)]+['k1shuf']
    res={f'{li}.{hd}':{a:{'hit':0,'n':0} for a in ARMS}
         for li,hd in HEADS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        mout={}; attin={}; pre={}
        hs=[]
        for lj in range(maxli):
            def mh(mo,i_,o_,lj=lj): mout[lj]=o_.detach().float()
            def phj(mo_,args,lj=lj): pre[lj]=(args[0],args[1])
            hs.append(m.transformer.h[lj].mlp
                      .register_forward_hook(mh))
            hs.append(m.transformer.h[lj].attn
                      .register_forward_pre_hook(phj))
        for li,hd in HEADS:
            def ph(mo_,args,li=li): attin[li]=args[0]
            hs.append(m.transformer.h[li].attn
                      .register_forward_pre_hook(ph))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        B=4
        def a_relay(j,src,shuf=False):
            at=m.transformer.h[j].attn
            Xj,v1j=pre[j]
            Xs=F.rms_norm(src,(D,)) \
                .to(m.transformer.wte.weight.dtype)
            if shuf: Xs=Xs[torch.tensor([1,2,3,0])]
            v=at.c_v(Xs).view(B,T,9,128)
            vm=v if v1j is None else \
                (1-at.lamb)*v+at.lamb*v1j.view_as(v)
            cos,sin=at.rotary(at.c_q(Xj).view(B,T,9,128))
            qf=F.rms_norm(at.c_q(Xj).view(B,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(Xj).view(B,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(Xj).view(B,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(Xj).view(B,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            patm=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',patm,vm.float())
            return at.c_proj(z.transpose(1,2).contiguous()
                             .view(B,T,-1).to(Xj.dtype)).float()
        def build(vals,shuf=False):
            """vals: None (pure ladder) or dict lj->residual value
            source (chain_{k-1}'s pre-attn residual at block lj).
            Returns (codes per layer, residual per block)."""
            xr=E.clone(); out={}; resid={}
            for lj in range(maxli+1):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                resid[lj]=xr.clone()
                out[lj]=F.rms_norm(xr,(D,))
                if lj<maxli:
                    if vals is not None:
                        xr=xr+a_relay(lj,vals[lj],shuf)
                    xr=xr+mout[lj]
            return out,resid
        chains={}
        codes0,resid=build(None)
        chains['k0']=codes0
        prev=resid
        for k in range(1,KMAX+1):
            ck,resid=build(prev)
            chains[f'k{k}']=ck
            if k==1:
                cs,_=build(prev,shuf=True)
                chains['k1shuf']=cs
            prev=resid
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
            for a in ARMS:
                c=chains[a][li]
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
                    toks=ROWS[i+b,:T].tolist()
                    seen=set(); seenmask=[]
                    for q in range(T):
                        seenmask.append(toks[q] in seen)
                        seen.add(toks[q])
                    for q in range(8,T,8):
                        st=res[f'{li}.{hd}'][a]
                        kreal=int(pat[b,q,:q].abs().argmax())
                        kfold=int(fpat[b,q,:q].abs().argmax())
                        ok=int(kreal==kfold)
                        st['hit']+=ok; st['n']+=1
                        sk='seen' if seenmask[q] else 'fresh'
                        st[sk]=st.get(sk,0)+ok
                        st[sk+'_n']=st.get(sk+'_n',0)+1
        print(f'batch {i} done',flush=True)
    outj={}
    for k9,arms in res.items():
        outj[k9]={a:round(st['hit']/max(st['n'],1),3)
                  for a,st in arms.items()}
        stk=arms[f'k{KMAX}']
        outj[k9]['k3_seen']=round(stk.get('seen',0)
                                  /max(stk.get('seen_n',0),1),3)
        outj[k9]['k3_fresh']=round(stk.get('fresh',0)
                                   /max(stk.get('fresh_n',0),1),3)
        print(f"{k9}: "+" | ".join(f"{a} {outj[k9][a]}"
              for a in ARMS),flush=True)
    pa=all(outj[k][f'k{KMAX}']>=0.70 for k in outj)
    pb=all(outj[k]['k0']<=outj[k]['k1']<=outj[k]['k2']
           <=outj[k]['k3'] for k in outj)
    fr=[]
    for k in outj:
        lift=outj[k]['k1']-outj[k]['k0']
        sl=outj[k]['k1shuf']-outj[k]['k0']
        if lift>0.02: fr.append(sl/lift)
    pc=(sum(fr)/max(len(fr),1))<0.25 if fr else False
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'pred_d':True,
         'shuf_lift_fraction':round(sum(fr)/max(len(fr),1),3)
         if fr else None}
    for nm,v in (('a','k3 >=0.70 both heads'),
                 ('b','monotone in k'),
                 ('c','shuffled null <25% of k1 lift'),
                 ('d','seen/fresh split reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
