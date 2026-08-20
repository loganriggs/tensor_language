"""PAYLOAD DECOMP -- 419: a6.h3 broadcasts position-specific
content to the whole late stack (417), but substituting its values
with the pure MLP-ladder code recovers only ~40% of the deletion
gap (418) -- so its payload is NOT just relayed identity code.
Decompose the payload EXACTLY: the value path is linear given the
per-position rms scalar, so a6.h3's write at q splits into writer
contributions
   write_w(q) = c_proj_h3( sum_k pat[q,k] * s_k * Wv (a_w x_w)[k] )
over wte + every component write below layer 6 (+ the v1 lambda
term as its own bucket). Report signed projection shares (sum to
1) and norm shares. Control: the same decomposition for a6.h0
(the clean control head, 416/417).
REGISTERED PREDICTIONS:
  (a) m0 is the single largest writer of a6.h3's payload with a
      projection share >= 0.25 (the relay is real);
  (b) NOT a pure relay: m0's share <= 0.60 and at least 3 writers
      exceed 0.05 (mixed payload -- explains 418's partial
      substitution);
  (c) contrast: a6.h0's payload profile differs -- its top writer
      is not m0, or m0's share is at least 0.10 lower."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'payload_decomp_results.json'
LJ=6; HEADS=[3,0]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    acc={h:{w:{'proj':0.0,'norm':0.0} for w in WR+['v1']}
         for h in HEADS}
    acc_tot={h:0.0 for h in HEADS}
    nseen=0
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4
        outs={}; pre={}
        hs=[]
        for lj in range(LJ):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(key=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[key]=y.detach().float()
                    return h
                hs.append(mod.register_forward_hook(mk()))
        def ph(mo_,args): pre['X']=args[0]; pre['v1']=args[1]
        hs.append(m.transformer.h[LJ].attn
                  .register_forward_pre_hook(ph))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        at=m.transformer.h[LJ].attn
        lam=m.transformer.h[LJ].lambdas.detach().float()
        X=pre['X']; v1c=pre['v1']
        # per-position rms scale: X = Xpre / rms(Xpre)
        Xpre=lam[0]*(E+sum(outs[w] for w in WR if w!='wte')) \
            +lam[1]*E
        scale=(X.float().norm(dim=-1,keepdim=True)
               /Xpre.norm(dim=-1,keepdim=True).clamp_min(1e-6))
        contribs={}
        for w in WR:
            base=(lam[0]+lam[1])*E if w=='wte' else lam[0]*outs[w]
            contribs[w]=base*scale
        Wv=at.c_v.weight.float()
        bias_v=at.c_v.bias.float() if at.c_v.bias is not None \
            else None
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
        kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
        qf,kf=are(qf,cos,sin),are(kf,cos,sin)
        q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))
        k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))
        q2,k2=are(q2,cos,sin),are(k2,cos,sin)
        sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
        pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
        Wp=at.c_proj.weight.float()
        for hd in HEADS:
            a9,b9=hd*128,(hd+1)*128
            Ph=Wp[:,a9:b9]                    # D x 128
            def head_write(vsrc):
                vh=(vsrc@Wv[a9:b9].T)          # B,T,128
                vm=(1-at.lamb)*vh
                z=torch.einsum('bqk,bkd->bqd',pat[:,hd],vm)
                return z@Ph.T
            parts={}
            for w in WR:
                parts[w]=head_write(contribs[w])
            # v1 lambda term (not a residual writer)
            v1h=v1c.view(B,T,9,128)[:,:,hd].float()*at.lamb
            zz=torch.einsum('bqk,bkd->bqd',pat[:,hd],v1h)
            parts['v1']=zz@Ph.T
            if bias_v is not None:
                bh=bias_v[a9:b9].expand(B,T,128)*(1-at.lamb)
                zb=torch.einsum('bqk,bkd->bqd',pat[:,hd],bh)
                parts['bias']=zb@Ph.T
            tot=sum(parts.values())
            tn=(tot*tot).sum(-1).clamp_min(1e-9)
            for w,pv in parts.items():
                if w not in acc[hd]: acc[hd][w]={'proj':0.0,
                                                 'norm':0.0}
                acc[hd][w]['proj']+=float(
                    ((pv*tot).sum(-1)/tn).sum())
                acc[hd][w]['norm']+=float(pv.norm(dim=-1).sum())
            acc_tot[hd]+=float(sum(
                p.norm(dim=-1) for p in parts.values()).sum())
        nseen+=B*T
        print(f'batch {i} done',flush=True)
    out={}
    for hd in HEADS:
        proj={w:round(v['proj']/nseen,4)
              for w,v in acc[hd].items()}
        norm={w:round(v['norm']/max(acc_tot[hd],1e-9),4)
              for w,v in acc[hd].items()}
        top=sorted(proj.items(),key=lambda kv:-abs(kv[1]))[:6]
        out[f'a6.h{hd}']={'proj_share':proj,'norm_share':norm,
                          'top_by_proj':top}
        print(f"a6.h{hd} top payload writers (proj share): {top}",
              flush=True)
    p3=out['a6.h3']['proj_share']
    m0=p3.get('m0',0.0)
    top1=max(p3.items(),key=lambda kv:abs(kv[1]))
    pa=(top1[0]=='m0' and m0>=0.25)
    pb=(m0<=0.60 and sum(1 for v in p3.values() if abs(v)>0.05)>=3)
    p0=out['a6.h0']['proj_share']
    top1_0=max(p0.items(),key=lambda kv:abs(kv[1]))
    pc=(top1_0[0]!='m0' or (m0-p0.get('m0',0.0))>=0.10)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','m0 top writer with share >=0.25'),
                 ('b','mixed payload (m0<=0.60, >=3 writers >5%)'),
                 ('c','control head profile differs')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
