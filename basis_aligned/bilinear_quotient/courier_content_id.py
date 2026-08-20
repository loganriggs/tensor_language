"""COURIER CONTENT ID -- 418: a6.h3's hub effect is NOT its mean
write (417: mean-ablation leaves the generic shift at 0.107 vs
0.122) -- its downstream influence is position-specific content
everywhere. 399 showed its channel into 8.4 is ~95% RELAYED
LADDER CODE. Closing test: substitute a6.h3's VALUES with the
pure-MLP-ladder reconstruction (real patterns kept) in the live
model. If its content is relayed identity code, the substitution
should be nearly free for the whole downstream stack.
Arms: ladval (a6.h3 values from ladder residual at layer 6) /
shufval (same, row-shuffled values; null) / zero3 (reference).
REGISTERED PREDICTIONS:
  (a) IDENTITY-CODE CONTENT: ladval median generic shift <= 0.05
      (near the a6.h0-control floor of 0.037; vs 0.122 zeroed);
  (b) courier function preserved: ladval trio match shift <=
      0.10 (vs 0.18-0.34 zeroed);
  (c) specificity: shufval median generic shift >= 0.10 (right
      structure, wrong content is as bad as nothing)."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'courier_content_id_results.json'
NR=16
TRIO=[(7,3),(8,3),(8,4)]
DOWN=[(li,hd) for li in range(7,18) for hd in range(9)]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    gen={a:{f'{li}.{hd}':[0,0] for li,hd in DOWN}
         for a in ('zero3','ladval','shufval')}
    mat={a:{f'{li}.{hd}':[0,0] for li,hd in TRIO}
         for a in ('zero3','ladval','shufval')}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4
        mmask=torch.zeros(B,T,dtype=torch.bool)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                ism=t in last and last[t]+1<q
                last[t]=q
                if ism and q>=8: mmask[b,q]=True
        # ladder residual at layer 6 (pure MLP chain, per 397)
        mout={}
        hs0=[m.transformer.h[lj].mlp.register_forward_hook(
            (lambda lj: lambda mo,i_,o_: mout.__setitem__(
                lj,o_.detach().float()))(lj)) for lj in range(6)]
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs0: h.remove()
        xl=E.clone()
        for lj in range(6):
            blk=m.transformer.h[lj]
            lam=blk.lambdas.detach().float()
            xl=lam[0]*xl+lam[1]*E
            xl=xl+mout[lj]
        lam6=m.transformer.h[6].lambdas.detach().float()
        Xlad=F.rms_norm(lam6[0]*xl+lam6[1]*E,(D,)) \
            .to(m.transformer.wte.weight.dtype)
        def run(mode):
            caps={}
            hs=[]
            for li in range(7,18):
                def ph(mo_,args,li=li): caps[li]=args[0]
                hs.append(m.transformer.h[li].attn
                          .register_forward_pre_hook(ph))
            if mode is not None:
                kh=3
                at=m.transformer.h[6].attn
                def fh(mo_,args,o_,at=at,kh=kh,mode=mode):
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
                    sc2=torch.einsum('bqhd,bkhd->bhqk',
                                     q2.float(),k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    if mode in ('ladval','shufval'):
                        Xs=Xlad if mode=='ladval' else \
                            Xlad[torch.tensor([1,2,3,0])]
                        vl=at.c_v(Xs).view(B,T,9,128)
                        vml=(1-at.lamb)*vl+at.lamb*v1.view_as(vl)
                        z[:,kh]=torch.einsum('bqk,bkd->bqd',
                            pat[:,kh],vml[:,:,kh].float())
                    else:
                        z[:,kh]=0
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            argm={}
            for li in range(7,18):
                at=m.transformer.h[li].attn
                X=caps[li]
                cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))
                k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))
                q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                kf.float())
                sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                 k2.float())
                pat=(sc*sc2).abs()*torch.tril(
                    torch.ones(T,T,device=DEV))
                argm[li]=pat.argmax(-1).cpu()
            for h in hs: h.remove()
            return argm
        a0=run(None)
        for a in ('zero3','ladval','shufval'):
            a1=run(a)
            for li,hd in DOWN:
                sh=gen[a][f'{li}.{hd}']
                d0=a0[li][:,hd,8:]; d1=a1[li][:,hd,8:]
                sh[0]+=int((d0!=d1).sum()); sh[1]+=d0.numel()
            for li,hd in TRIO:
                sh=mat[a][f'{li}.{hd}']
                d0=a0[li][:,hd]; d1=a1[li][:,hd]
                mm=mmask
                sh[0]+=int((d0[mm]!=d1[mm]).sum())
                sh[1]+=int(mm.sum())
        print(f'batch {i} done',flush=True)
    out={}
    for a in ('zero3','ladval','shufval'):
        rates=sorted(v[0]/max(v[1],1) for v in gen[a].values())
        med=rates[len(rates)//2]
        trio={k:round(v[0]/max(v[1],1),3)
              for k,v in mat[a].items()}
        out[a]={'median_generic_shift':round(med,4),
                'trio_match_shift':trio}
        print(f"{a}: median generic {med:.4f} | trio match {trio}",
              flush=True)
    pa=out['ladval']['median_generic_shift']<=0.05
    pb=all(v<=0.10 for v in
           out['ladval']['trio_match_shift'].values())
    pc=out['shufval']['median_generic_shift']>=0.10
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','ladval generic shift <=0.05 (identity-code content)'),
                 ('b','ladval trio match <=0.10 (courier preserved)'),
                 ('c','shufval >=0.10 (content-specific)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
