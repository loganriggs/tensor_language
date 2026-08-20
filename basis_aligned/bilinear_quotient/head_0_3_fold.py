"""HEAD 0.3 FOLD -- 475: the input-composition thread is closed
(474: neither writers nor bands isolate; dependence accumulates
smoothly). The program's two complete circuits both came from
chasing a specific anomaly instead, so take the next one: head 0.3
is the SECOND costliest head in the model (+0.112 nats, behind
only the sink at +0.916).
Layer 0 is special: attention there can only read token
embeddings, so a layer-0 head's pattern is a pure function of the
token sequence plus rotary position. That makes it EXACTLY
foldable in principle -- its score should be computable from a
(query token, key token) table and its output from a per-read-
token table, with no forward pass.
Arms:
  reads      : offset histogram and same-token rate at generic
               positions (is it previous-token, self, or diffuse?)
  fold       : predict the top read from weights + tokens +
               rotary alone, no residual stream
  value-table: replace the head's output with a table indexed by
               the READ token (its value is a function of that
               token alone at layer 0), priced in CE
REGISTERED PREDICTIONS:
  (a) SINGLE OFFSET: one offset carries >= 60% of its top reads;
  (b) EXACTLY FOLDABLE: the token+rotary fold reproduces the real
      top read >= 95% of the time (at layer 0 this is an identity
      up to numerics, so a miss means a bug, not a finding);
  (c) TABLE REPLACEMENT: swapping its value output for a
      per-read-token table costs <= 0.02 nats overall."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=0; HD=3
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_0_3_fold_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    at=m.transformer.h[LJ].attn
    off=({}); same=0; n=0; hit=0
    Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
    valsum={}; valcnt={}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4
        E=F.rms_norm(m.transformer.wte(idx),(D,))
        X=E                      # layer-0 attn input IS this
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        def rot(w):
            return are(F.rms_norm(w(X).view(B,T,9,128),
                       (128,))[:,:,HD][:,:,None],cos,sin)[:,:,0]
        qf,kf=rot(at.c_q),rot(at.c_k); q2,k2=rot(at.c_q2),rot(at.c_k2)
        tril=torch.tril(torch.ones(T,T,device=DEV))
        pat=((torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())/128)
             *(torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())/128)) \
            *tril
        # fold: rebuild the same quantities from tokens alone
        Ef=F.rms_norm(m.transformer.wte(idx),(D,))
        cosf,sinf=at.rotary(at.c_q(Ef).view(B,T,9,128))
        def rotf(w):
            return are(F.rms_norm(w(Ef).view(B,T,9,128),
                       (128,))[:,:,HD][:,:,None],cosf,sinf)[:,:,0]
        qff,kff=rotf(at.c_q),rotf(at.c_k)
        q2f,k2f=rotf(at.c_q2),rotf(at.c_k2)
        fpat=((torch.einsum('bqd,bkd->bqk',qff.float(),
                            kff.float())/128)
              *(torch.einsum('bqd,bkd->bqk',q2f.float(),
                             k2f.float())/128))*tril
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()*(1-at.lamb)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist()
            for q in range(4,T,4):
                k=int(pat[b,q,:q+1].abs().argmax())
                o=k-q
                off[o]=off.get(o,0)+1
                same+=int(toks[k]==toks[q]); n+=1
                hit+=int(k==int(fpat[b,q,:q+1].abs().argmax()))
                tk=int(ROWS[i+b,k])
                valsum[tk]=valsum.get(tk,0)+v[b,k]
                valcnt[tk]=valcnt.get(tk,0)+1
        print(f'batch {i} done',flush=True)
    top=sorted(off.items(),key=lambda kv:-kv[1])[:6]
    frac=top[0][1]/max(n,1)
    fold=hit/max(n,1)
    print(f'offsets {top} | dominant {frac:.3f} | fold match '
          f'{fold:.3f} | same-token {same/max(n,1):.3f}',flush=True)
    # value-table arm: replace the head's z with a per-read-token
    # table built from the same rows (in-sample; a fair upper bound)
    tabv={t:(valsum[t]/valcnt[t]) for t in valsum}
    def run(tab):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if tab is not None:
                def fh(mo_,args,o_,at=at):
                    y,v1r=o_
                    X2=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    vv=at.c_v(X2).view(B,T,9,128)
                    vm=(1-at.lamb)*vv+at.lamb*v1.view_as(vv)
                    c2,s2=at.rotary(at.c_q(X2).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X2).view(B,T,9,128),(128,)),c2,s2)
                    qq,kk=r2(at.c_q),r2(at.c_k)
                    q22,k22=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                                    kk.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                                     k22.float())/128
                    p2=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
                    rep=torch.zeros_like(z[:,HD])
                    for b in range(B):
                        for k in range(T):
                            tk=int(bb[b,k])
                            if tk in tab: rep[b,k]=tab[tk]
                    z[:,HD]=torch.einsum('bqk,bkd->bqd',
                                         p2[:,HD],rep)
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X2.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            tot+=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                 reduction='none').mean().item()
            cnt+=1
            for h in hs: h.remove()
        return tot/max(cnt,1)
    base=run(None); tabc=run(tabv)
    pa=frac>=0.60; pb=fold>=0.95; pc=(tabc-base)<=0.02
    out={'top_offsets':top,'dominant_offset_frac':round(frac,3),
         'fold_match':round(fold,3),
         'same_token_rate':round(same/max(n,1),3),
         'dce_value_table':round(tabc-base,4),
         'n_reads':n,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f'value-table dCE {tabc-base:+.4f}')
    for nm,v in (('a','one offset carries >=60%'),
                 ('b','token+rotary fold reproduces reads'),
                 ('c','per-read-token table costs <=0.02')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
