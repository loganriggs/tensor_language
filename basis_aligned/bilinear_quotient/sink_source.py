"""SINK SOURCE -- 437: head 5.7 adds a learned constant taken from
position 0, where the value norm is 730 against 197 elsewhere
(432/435). Where does that vector come from, and what is it?
Decompose the residual entering layer 5 AT POSITION 0 into exact
writer contributions (the lambda mix and rms scale are
per-position scalars, so the split is exact), then check the
chain end to end: does the reconstructed constant match the head's
actual mean write?
Position 0 has no context by construction, so its content can only
come from the token embedding and the MLP chain -- which makes
this a clean test of whether the model's bias vector is built by
identifiable machinery or is a diffuse accumulation.
REGISTERED PREDICTIONS:
  (a) CONCENTRATED SOURCE: one writer carries >= 0.40 of the
      position-0 residual at layer 5 (projection share);
  (b) EARLY MACHINERY: that writer is wte or m0 (position 0 sees
      no attention context worth the name);
  (c) CHAIN CLOSES: the write produced by routing position 0's
      value through the head's own projection has cosine >= 0.9
      with 5.7's actual mean write (435's replacement vector);
  (d) report the top-5 writer shares and the BOS/first-token
      identity (which token sits at position 0 in these rows)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sink_source_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    shares={w:0.0 for w in WR}; n=0
    coss=[]; toks0=[]
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4
        toks0+= [int(t) for t in ROWS[i:i+4,0].tolist()]
        outs={}; pre={}
        hs=[]
        for lj in range(LJ):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(k9=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[k9]=y.detach().float()
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
        X=pre['X']
        # 2026-08-20 (writeup 503): exact per-writer coefficients.
        _p=cl.writer_parts(LJ,E,outs,'a')
        Xpre=sum(_p.values())
        scale=(X.float().norm(dim=-1,keepdim=True)
               /Xpre.norm(dim=-1,keepdim=True).clamp_min(1e-6))
        cl.check_parts(_p,X,label='sink_source')
        parts={w:_p[w]*scale for w in WR if w in _p}
        tot=sum(parts.values())
        p0=0   # position 0
        tn=(tot[:,p0]*tot[:,p0]).sum(-1).clamp_min(1e-9)
        for w in WR:
            shares[w]+=float(((parts[w][:,p0]*tot[:,p0]).sum(-1)
                              /tn).sum())
        n+=B
        # chain check: route position-0 value through the head
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()
        vm=(1-at.lamb)*v
        Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
        w0=(vm[:,p0]@Wp.T)              # what pos-0's value writes
        # actual mean write of the head
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))[:,:,HD]
        kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))[:,:,HD]
        q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))[:,:,HD]
        k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))[:,:,HD]
        qf=are(qf[:,:,None],cos,sin)[:,:,0]
        kf=are(kf[:,:,None],cos,sin)[:,:,0]
        q2=are(q2[:,:,None],cos,sin)[:,:,0]
        k2=are(k2[:,:,None],cos,sin)[:,:,0]
        pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
             *torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())) \
            *torch.tril(torch.ones(T,T,device=DEV))
        vfull=(1-at.lamb)*v
        z=torch.einsum('bqk,bkd->bqd',pat,vfull)
        actual=(z@Wp.T).mean(dim=1)     # mean write per row
        for b in range(B):
            coss.append(float(F.cosine_similarity(
                w0[b],actual[b],dim=0)))
        print(f'batch {i} done',flush=True)
    sh={w:round(v/max(n,1),4) for w,v in shares.items()}
    top=sorted(sh.items(),key=lambda kv:-abs(kv[1]))[:5]
    mc=sum(coss)/len(coss)
    import collections
    t0c=collections.Counter(toks0).most_common(3)
    pa=abs(top[0][1])>=0.40
    pb=top[0][0] in ('wte','m0')
    pc=mc>=0.9
    out={'pos0_writer_shares':sh,'top5':top,
         'chain_cosine':round(mc,3),
         'first_tokens':[(cl.d1(t),c) for t,c in t0c],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':True,'runtime_s':time.time()-t0}
    print(f'position-0 writers at layer 5: {top}')
    print(f'chain cosine (pos-0 value write vs head mean write): '
          f'{mc:.3f} | first tokens {out["first_tokens"]}')
    for nm,v in (('a','top writer share >=0.40'),
                 ('b','top writer is wte or m0'),
                 ('c','chain cosine >=0.9'),
                 ('d','shares reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
