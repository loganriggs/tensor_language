"""INDUCTION CHANNEL v2 -- exact v-mixing (v1 approximation was the one
caveat left on the third refutation). vm = (1-lamb)*v + lamb*v1 with v1
the LAYER-0 values, computed exactly from block 0's attention input.
Same bars as v1; THREE-STRIKES RULE registered: if (a)/(b) fail again,
the linear-reduction hypothesis class for induction transport closes at
scope, and the head's own bilinear computation stands as its minimal
description.
REGISTERED PREDICTIONS: (a) channel R^2 >= 0.3 for >=2/3 heads;
(b) channel >= 2x same-position control; (c) shuffled null <= 0.1;
(d) rank-32 within 0.05."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'induction_channel2_results.json'
CA=300; NB=16
HEADS=[(2,5),(3,8),(5,5)]

@torch.no_grad()
def main():
    t0=time.time()
    mod=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod.apply_rotary_emb
    T=256
    lset=sorted({li for li,_ in HEADS})
    Xs={li:[] for li in lset}
    X0s=[]
    hs=[m.transformer.h[0].attn.register_forward_pre_hook(
        lambda mo,args: X0s.append(args[0].detach().float()))]
    for li in lset:
        def mk(li=li):
            def h(mo,args): Xs[li].append(args[0].detach().float())
            return h
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    toks=[]
    for i in range(CA,CA+NB*4,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        toks.append(bb[:,:-1])
    for h in hs: h.remove()
    tt=torch.cat(toks)                      # (B,T)
    B=tt.shape[0]
    g=torch.Generator().manual_seed(0)
    def matchidx(tk):
        # j*[b,q] = last j<q with tk[j]==tk[q], then +1; -1 if none
        M=torch.full((tk.shape[0],T),-1,dtype=torch.long)
        for b_ in range(tk.shape[0]):
            last={}
            row=tk[b_].tolist()
            for qq in range(T):
                t=row[qq]
                if t in last and last[t]+1<qq:
                    M[b_,qq]=last[t]+1
                last[t]=qq
        return M
    MI=matchidx(tt.cpu())
    tsh=torch.stack([r[torch.randperm(T,generator=g)] for r in tt.cpu()])
    MIs=matchidx(tsh)
    out={'heads':{}}
    na=0; nb_=0; nc=0; nd=0
    for li,hd in HEADS:
        at=m.transformer.h[li].attn
        X=torch.cat(Xs[li])                 # (B,T,D)
        q=at.c_q(X).view(B,T,9,128); k=at.c_k(X).view(B,T,9,128)
        q2=at.c_q2(X).view(B,T,9,128); k2=at.c_k2(X).view(B,T,9,128)
        v=at.c_v(X).view(B,T,9,128)
        X0=torch.cat(X0s)
        v1=m.transformer.h[0].attn.c_v(X0).view(B,T,9,128)
        vm=(1-at.lamb)*v+at.lamb*v1
        cos,sin=at.rotary(q)
        qn=F.rms_norm(q,(128,)); kn=F.rms_norm(k,(128,))
        qn,kn=are(qn,cos,sin),are(kn,cos,sin)
        q2n=F.rms_norm(q2,(128,)); k2n=F.rms_norm(k2,(128,))
        q2n,k2n=are(q2n,cos,sin),are(k2n,cos,sin)
        sc=torch.einsum('bqd,bkd->bqk',qn[:,:,hd].float(),
                        kn[:,:,hd].float())/128
        sc2=torch.einsum('bqd,bkd->bqk',q2n[:,:,hd].float(),
                         k2n[:,:,hd].float())/128
        pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
        z=torch.einsum('bqk,bkd->bqd',pat,vm[:,:,hd].float())  # (B,T,128)
        el=(MI>=0)
        bi,qi=torch.nonzero(el,as_tuple=True)
        ji=MI[bi,qi]
        Xf=X.reshape(-1,D)
        src=Xf[(bi*T+ji).to(DEV)]
        same=Xf[(bi*T+qi).to(DEV)]
        jsh=MIs[bi,qi]
        ok=jsh>=0
        srcS=Xf[((bi[ok]*T)+jsh[ok]).to(DEV)]
        Y=z.reshape(-1,128)[(bi*T+qi).to(DEV)]
        n=len(Y); ntr=n//2
        def fitr2(A_,B2,rank=None):
            Atr,Btr=A_[:ntr],B2[:ntr]; Ate,Bte=A_[ntr:],B2[ntr:]
            lam=1e-2*ntr
            Wm=torch.linalg.solve(Atr.T@Atr+lam*torch.eye(D,device=DEV),
                                  Atr.T@Btr)
            if rank:
                U,S,Vh=torch.linalg.svd(Wm,full_matrices=False)
                Wm=(U[:,:rank]*S[:rank])@Vh[:rank]
            pr=Ate@Wm
            return 1-float(((pr-Bte)**2).sum())/\
                float(((Bte-Btr.mean(0))**2).sum())
        r2c=fitr2(src,Y); r2s=fitr2(same,Y)
        r2c32=fitr2(src,Y,rank=32)
        Ysh=Y[ok]; nsh=len(Ysh); nts=nsh//2
        lam=1e-2*nts
        Wm=torch.linalg.solve(srcS[:nts].T@srcS[:nts]
                              +lam*torch.eye(D,device=DEV),
                              srcS[:nts].T@Ysh[:nts])
        pr=srcS[nts:]@Wm
        r2n=1-float(((pr-Ysh[nts:])**2).sum())/\
            float(((Ysh[nts:]-Ysh[:nts].mean(0))**2).sum())
        print(f'L{li}h{hd}: channel R2 {r2c:.3f} (rank32 {r2c32:.3f}) | '
              f'same-pos {r2s:.3f} | shuffled {r2n:.3f} | n={n}',
              flush=True)
        out['heads'][f'L{li}h{hd}']={'channel':round(r2c,3),
            'channel_r32':round(r2c32,3),'same_pos':round(r2s,3),
            'shuffled':round(r2n,3),'n':n}
        if r2c>=0.3: na+=1
        if r2c>=2*max(r2s,1e-3): nb_+=1
        if r2n<=0.1: nc+=1
        if r2c-r2c32<=0.05: nd+=1
        Xs[li]=None
    pa=na>=2; pb=nb_>=2; pc=nc==3; pd=nd>=2
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    out['pred_c']=bool(pc); out['pred_d']=bool(pd)
    print(f"(a) channel R2>=0.3 for >=2 heads ({na}/3): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) >=2x same-pos ({nb_}/3): {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled <=0.1 (3/3 needed, {nc}/3): "
          f"{'HELD' if pc else 'FAILED'}")
    print(f"(d) rank-32 within 0.05 ({nd}/3): {'HELD' if pd else 'FAILED'}")
    out['note']='exact v-mixing; three-strikes rule applies'
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
