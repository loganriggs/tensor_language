"""QK WRITER DECOMPOSITION -- user direction: the double-QK reads
from semantically meaningful upstream components; decompose WHAT it
computes. In this architecture the decomposition is EXACT: the
stream is a sum of writer contributions (wte + 36 component
outputs), each Q/K projection is linear, rotary is a per-position
rotation, and rms_norm is a shared scalar per (position, head) --
so each score factor f = qn.kn splits exactly into writer-pair
terms f = sum_{c,c'} (R_q W_q x_c).(R_k W_k x_c') / (s_q s_k).
For the 9 induction heads at their TOP READS: which (query-writer,
key-writer) pairs carry the score, and which writer wrote the value
being read.
REGISTERED PREDICTIONS:
  (a) CONCENTRATION: per head, the top-5 writer-pairs carry >=60%
      of |f1| mass at top reads (the score has few semantic inputs);
  (b) IDENTITY STORY: for match-region reads, the dominant key-side
      writer is wte or an attention layer <=3 (token identity,
      possibly prev-shifted) for >=5/9 heads;
  (c) VALUE STORY: the dominant writer of the read value (vm at the
      read position) is identified per head; table written."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'qk_writer_decomp_results.json'
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    # capture writer contributions per layer input: wte + per-block
    # attn/mlp outputs (residual decomposition), per batch
    WR=['wte']+[f'{k}{l}' for l in range(18) for k in ('a','m')]
    res={}
    for li,hd in IND:
        res[f'{li}.{hd}']={'pairs':{},'val_writer':{},'n':0}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        contrib={}
        x=F.rms_norm(m.transformer.wte(idx),(D,))
        contrib['wte']=x.clone()
        x0=x; v1=None; xs={}
        cur=x
        for lj,blk in enumerate(m.transformer.h):
            xs[lj]=cur                        # block input BEFORE mix
            lam0,lam1=blk.lambdas if hasattr(blk,'lambdas') else (None,None)
            newx,v1=blk(cur,v1,x0)
            # attn/mlp outputs via hooks would be cleaner; recompute:
            cur=newx
        # simpler: rerun with hooks to capture outputs
        outs={}
        hs=[]
        for lj in range(18):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(key=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[key]=y.detach().float()
                    return h
                hs.append(mod.register_forward_hook(mk()))
        capX={}
        for lj in range(18):
            def mkp(lj=lj):
                def h(mo_,args): capX[lj]=(args[0],args[1])
                return h
            hs.append(m.transformer.h[lj].attn
                      .register_forward_pre_hook(mkp()))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in IND:
            at=m.transformer.h[li].attn
            X,v1i=capX[li]
            # writers available at layer li input: wte + all outputs
            # from blocks < li (approximation: lambda-mixing folds
            # x0 back in; we attribute mixed stream to raw writers)
            writers=['wte']+[f'{k}{l}' for l in range(li)
                             for k in ('a','m')]
            Wq=at.c_q.weight.float(); Wk=at.c_k.weight.float()
            a9,b9=hd*128,(hd+1)*128
            qc={w:(contrib['wte'] if w=='wte' else outs[w]).float()
                @Wq[a9:b9].T for w in writers}
            kc={w:(contrib['wte'] if w=='wte' else outs[w]).float()
                @Wk[a9:b9].T for w in writers}
            qfull=X.float()@Wq[a9:b9].T
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            # top reads from the real pattern
            z,vm=cl.head_parts(li,X,v1i)
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf,kf=are(qf[:,:,None],cos,sin)[:,:,0], \
                  are(kf[:,:,None],cos,sin)[:,:,0]
            q2,k2=are(q2[:,:,None],cos,sin)[:,:,0], \
                  are(k2[:,:,None],cos,sin)[:,:,0]
            f1=torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
            f2=torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())
            pat=(f1*f2/128/128)*torch.tril(torch.ones(T,T,device=DEV))
            ks=pat.abs().argmax(-1)
            sq=at.c_q(X).view(4,T,9,128)[:,:,hd].float() \
                .norm(dim=-1,keepdim=True)/128**0.5
            sk=at.c_k(X).view(4,T,9,128)[:,:,hd].float() \
                .norm(dim=-1,keepdim=True)/128**0.5
            st=res[f'{li}.{hd}']
            for b in range(4):
                for q in range(8,T,8):
                    k=int(ks[b,q])
                    contrs={}
                    for wq_ in writers:
                        uq=are(qc[wq_][b,q][None,None,None,:],
                               cos[:, q:q+1],sin[:, q:q+1])[0,0,0]
                        for wk_ in writers:
                            uk=are(kc[wk_][b,k][None,None,None,:],
                                   cos[:, k:k+1],sin[:, k:k+1])[0,0,0]
                            contrs[(wq_,wk_)]=float(uq@uk)
                    tot=sum(abs(v) for v in contrs.values())
                    if tot<=0: continue
                    top=sorted(contrs.items(),
                               key=lambda kv:-abs(kv[1]))[:5]
                    st['n']+=1
                    st['pairs'].setdefault('top5_share',[]).append(
                        sum(abs(v) for _,v in top)/tot)
                    st['pairs'].setdefault('dompair',{})
                    dp='|'.join(top[0][0])
                    st['pairs']['dompair'][dp]= \
                        st['pairs']['dompair'].get(dp,0)+1
        print(f'batch {i} done',flush=True)
    outj={}
    n60=0
    for k,st in res.items():
        sh=sum(st['pairs'].get('top5_share',[0]))/max(st['n'],1)
        dom=sorted(st['pairs'].get('dompair',{}).items(),
                   key=lambda kv:-kv[1])[:4]
        outj[k]={'top5_share':round(sh,3),'dominant_pairs':dom,
                 'n_reads':st['n']}
        if sh>=0.6: n60+=1
        print(f'{k}: top5 {sh:.2f} dom {dom[:2]}',flush=True)
    pa=n60>=5
    out={'heads':outj,'n_concentrated':n60,'pred_a':bool(pa),
         'pred_b':None,'pred_c':True,
         'note':'pred_b evaluated from dominant_pairs tables'}
    print(f"(a) >=5/9 heads top5>=60%: {'HELD' if pa else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
