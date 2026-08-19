"""MECH REPLICATE -- first COMPUTATIONAL-grade attempt (user
standard: write the mechanism as code; the code must replicate model
behavior on the member datapoints). Target: bilin18's induction
behavior, the canonical mechanistic story. CODE HYPOTHESIS, written
down: each of the 9 census induction heads computes
    z_h(q) = alpha_h * vm_h(j+1)   if token[q] occurred before at j
    z_h(q) = 0                      otherwise
i.e., a one-hot read of the value at the position AFTER the previous
occurrence of the current token, scaled by one declared scalar per
head (alpha_h, fit on 32 held-in rows). Everything else about the
model stays real. If the induction story is the computation, running
the model with all 9 heads REPLACED BY THIS CODE must reproduce
behavior at match positions.
REGISTERED PREDICTIONS (eval on 64 held-out rows):
  (a) REPLICATION: CE at match positions under code-substitution
      within +0.05 of the real model;
  (b) per-position logit(continuation token) correlation real-vs-
      code >= 0.8 at match positions;
  (c) eligibility: non-match positions cost <= +0.10 (the code says
      the heads are silent there);
  (d) CONTROL: shuffled-match substitution (wrong j) breaks (a) by
      >= 3x."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_replicate_results.json'
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    ROWS=cl.rows()
    def match_idx(row):
        toks=row[:T].tolist(); last={}
        M=torch.full((T,),-1,dtype=torch.long)
        for q in range(T):
            t=toks[q]
            if t in last and last[t]+1<q: M[q]=last[t]+1
            last[t]=q
        return M
    MI=torch.stack([match_idx(ROWS[r]) for r in range(96)])
    def head_parts(li,X,v1):
        at=m.transformer.h[li].attn
        Bb=X.shape[0]
        v=at.c_v(X).view(Bb,T,9,128)
        vm=(1-at.lamb)*v+at.lamb*(v1.view_as(v) if v1 is not None
                                  else v)
        cos,sin=at.rotary(at.c_q(X).view(Bb,T,9,128))
        qf=F.rms_norm(at.c_q(X).view(Bb,T,9,128),(128,))
        kf=F.rms_norm(at.c_k(X).view(Bb,T,9,128),(128,))
        qf,kf=are(qf,cos,sin),are(kf,cos,sin)
        q2=F.rms_norm(at.c_q2(X).view(Bb,T,9,128),(128,))
        k2=F.rms_norm(at.c_k2(X).view(Bb,T,9,128),(128,))
        q2,k2=are(q2,cos,sin),are(k2,cos,sin)
        sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
        s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
        pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
        z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
        return z,vm.float()
    # ---- fit alpha_h on rows 0..31 (held-in) ----
    num={k:0.0 for k in IND}; den={k:0.0 for k in IND}
    cap={}
    def mkpre(li):
        def h(mo_,args): cap[li]=(args[0],args[1])
        return h
    for i in range(0,32,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        Mb=MI[i:i+4].to(DEV)
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li,_ in IND]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li9,blkm in enumerate(m.transformer.h): x,v1=blkm(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in IND:
            X,v1i=cap[li]
            z,vm=head_parts(li,X,v1i)
            ok=Mb>=0
            tgt=vm[torch.arange(4)[:,None],Mb.clamp_min(0),hd]
            zq=z[:,hd].transpose(0,1)[...,:]  # careful dims
            zh=z[:,hd]  # (B,T,128)? z dims (B,H,T,128)->z[:,hd]=(B,T,128)
            num[(li,hd)]+=float((zh[ok]*tgt[ok]).sum())
            den[(li,hd)]+=float((tgt[ok]*tgt[ok]).sum())
    ALPHA={k:num[k]/max(den[k],1e-6) for k in IND}
    print('alpha:',{f'{k}':round(v,3) for k,v in ALPHA.items()},
          flush=True)
    # ---- eval on rows 32..95 (held-out) ----
    byl={}
    for li,hd in IND: byl.setdefault(li,[]).append(hd)
    def code_hooks(Mb,shuffle=False):
        hs=[]
        if shuffle:
            g=torch.Generator().manual_seed(9)
            Mb=Mb.clone()
            for b in range(Mb.shape[0]):
                ok=(Mb[b]>=0).nonzero().squeeze(1)
                Mb[b,ok]=Mb[b,ok[torch.randperm(len(ok),
                                                generator=g)]]
        Mb=Mb.to(DEV)
        for li,hds in byl.items():
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,li=li,hds=hds,Mb=Mb):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_parts(li,X,v1)
                Bb=X.shape[0]
                ok=(Mb>=0).to(DEV)
                for hd in hds:
                    tgt=vm[torch.arange(Bb,device=DEV)[:,None],
                           Mb.clamp_min(0),hd]
                    zc=ALPHA[(li,hd)]*tgt
                    zc[~ok]=0
                    z[:,hd]=zc
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,T,-1).to(X.dtype))
                return (yn,v1r)
            hs.append(at.register_forward_hook(fh))
        return hs
    def run(rows0,rows1,hookfn):
        ces=[]; lts=[]
        for i in range(rows0,rows1,4):
            bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
            tg=bb[:,1:]
            Mb=MI[i:i+4]
            hs=hookfn(Mb) if hookfn else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            for h in hs: h.remove()
            ce=F.cross_entropy(lg.reshape(-1,lg.size(-1)),
                               tg.reshape(-1),reduction='none')
            ces.append(ce.view(4,T).cpu())
            # continuation-token logit at match positions
            cont=torch.zeros(4,T,dtype=torch.long)
            ok=Mb>=0
            for b in range(4):
                for q in ok[b].nonzero().squeeze(1).tolist():
                    cont[b,q]=ROWS[i+b][MI[i+b][q]]
            lts.append(lg.gather(-1,cont.to(DEV)[...,None])
                       .squeeze(-1).cpu())
        return torch.cat(ces),torch.cat(lts)
    okm=(MI[32:96]>=0)
    ceR,ltR=run(32,96,None)
    ceC,ltC=run(32,96,lambda Mb: code_hooks(Mb))
    ceS,_=run(32,96,lambda Mb: code_hooks(Mb,shuffle=True))
    dm=float((ceC-ceR)[okm].mean()); dn=float((ceC-ceR)[~okm].mean())
    ds=float((ceS-ceR)[okm].mean())
    lr=ltR[okm]; lc=ltC[okm]
    corr=float(torch.corrcoef(torch.stack([lr,lc]))[0,1])
    pa=dm<=0.05; pb=corr>=0.8; pc_=dn<=0.10; pd=ds>=3*max(dm,1e-3)
    out={'alpha':{f'{k[0]}.{k[1]}':round(v,3) for k,v in ALPHA.items()},
         'match_cost':round(dm,4),'nonmatch_cost':round(dn,4),
         'shuffled_cost':round(ds,4),'logit_corr':round(corr,4),
         'n_match':int(okm.sum()),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':bool(pd)}
    print(f'match cost {dm:+.4f} | nonmatch {dn:+.4f} | shuffled '
          f'{ds:+.4f} | logit corr {corr:.3f}')
    print(f"(a) replication <=+0.05: {'HELD' if pa else 'FAILED'}")
    print(f"(b) logit corr >=0.8: {'HELD' if pb else 'FAILED'}")
    print(f"(c) nonmatch <=+0.10: {'HELD' if pc_ else 'FAILED'}")
    print(f"(d) shuffled >=3x: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
