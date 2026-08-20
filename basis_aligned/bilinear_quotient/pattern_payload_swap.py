"""PATTERN/PAYLOAD SWAP -- 421: a6.h3's payload composition is
NOT distinctive (420: m0 0.27 / first-layer-values 0.27 / m3 0.14
-- and the control head a6.h0 carries nearly the same mixture,
0.36/0.29/0.11). All layer-6 heads read the same residual, so what
must distinguish the courier is its PATTERN (where it reads), not
what it carries. Crossover test in the live model, head 3 vs the
control head 0 at layer 6:
  patswap : z3 = h0's pattern applied to h3's values
  valswap : z3 = h3's pattern applied to h0's values
  zero3   : reference
REGISTERED PREDICTIONS:
  (a) PATTERN CARRIES IT: patswap keeps the deep trio's match-read
      shift >= 0.15 (near zeroing's 0.18-0.34);
  (b) PAYLOAD IS FUNGIBLE: valswap keeps it <= 0.10;
  (c) sanity: neither arm exceeds zero3's shift on the trio."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'pattern_payload_swap_results.json'
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
         for a in ('zero3','patswap','valswap')}
    mat={a:{f'{li}.{hd}':[0,0] for li,hd in TRIO}
         for a in ('zero3','patswap','valswap')}
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
                    if mode=='patswap':
                        z[:,3]=torch.einsum('bqk,bkd->bqd',
                            pat[:,0],vm[:,:,3].float())
                    elif mode=='valswap':
                        z[:,3]=torch.einsum('bqk,bkd->bqd',
                            pat[:,3],vm[:,:,0].float())
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
        for a in ('zero3','patswap','valswap'):
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
    for a in ('zero3','patswap','valswap'):
        rates=sorted(v[0]/max(v[1],1) for v in gen[a].values())
        med=rates[len(rates)//2]
        trio={k:round(v[0]/max(v[1],1),3)
              for k,v in mat[a].items()}
        out[a]={'median_generic_shift':round(med,4),
                'trio_match_shift':trio}
        print(f"{a}: median generic {med:.4f} | trio match {trio}",
              flush=True)
    tp=out['patswap']['trio_match_shift']
    tv=out['valswap']['trio_match_shift']
    tz=out['zero3']['trio_match_shift']
    pa=all(v>=0.15 for v in tp.values())
    pb=all(v<=0.10 for v in tv.values())
    pc=all(tp[k]<=tz[k]+0.02 and tv[k]<=tz[k]+0.02 for k in tz)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','patswap keeps trio damage >=0.15 (pattern)'),
                 ('b','valswap <=0.10 (payload fungible)'),
                 ('c','neither exceeds zeroing')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
