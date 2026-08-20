"""NEWLINE HEAD TRIGGER -- what makes head 12.6 fire?
The rhythm hypothesis died (writeup 500): 12.6 helps exactly as
much on line breaks that match the document's median line length
(+0.0875) as on ones that do not (+0.0909), so it is not doing
line arithmetic. What it clearly DOES do is push one token:
deleting it lowers the logit of "\\n" (id 198) by 0.137 at line
breaks while the best non-newline competitor moves by 0.004.
So the head is a newline pusher, and the open question is what
switches it on. Three hypotheses, each a registered bar:
  TOKEN-CONDITIONED. The push is a function of the current token
    (after a period, after a colon, after a list marker), which
    would make the head an elaborate bigram.
  DOCUMENT-GATED. The push is a function of whether this document
    is line-broken text at all -- the head detects verse, lists,
    tables, chat logs and turns on there, so the same trigger
    token gets a bigger push in newline-dense documents.
  NEITHER, i.e. position-by-position context.
Measured per position as the drop in the logit of token 198 when
12.6 is mean-ablated, which is the head's own causal contribution
to predicting a line break.
REGISTERED PREDICTIONS:
  (a) TOKEN-CONDITIONED: the mean push at the top-decile trigger
      token type is >= 3x the push at the median token type
      (types with >= 20 occurrences only);
  (b) DOCUMENT-GATED: holding the trigger token fixed to the top
      types, the push is >= 1.5x larger in documents whose newline
      density is above the median than below it;
  (c) IT IS A USABLE DETECTOR: the per-position push predicts
      "the next token is a newline" with AUC >= 0.75.
  NULL/CONTROL: the same AUC computed for head 12.2 -- the
      second-ranked head in the same layer, from 497 -- must be
      below 0.65. A same-layer head scoring as well would mean the
      measurement reflects the layer, not this head.
Reporting rule from 497 and 500: absolute pushes are reported
alongside every ratio, and no bar is scored on a ratio whose
denominator is near zero."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; NH=9; NLID=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_head_trigger_results.json'
NFRESH=64

def auc(pos,neg):
    if not pos or not neg: return float('nan')
    allv=sorted([(v,1) for v in pos]+[(v,0) for v in neg])
    r=0.0; i=0
    while i<len(allv):
        j=i
        while j<len(allv) and allv[j][0]==allv[i][0]: j+=1
        rank=(i+j+1)/2.0
        r+=sum(rank for v,l in allv[i:j] if l==1)
        i=j
    n1=len(pos); n0=len(neg)
    return (r-n1*(n1+1)/2)/(n1*n0)

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    at=m.transformer.h[LJ].attn

    def mk(HD):
        def fh(mo,args,o_):
            y,v1r=o_; X2=args[0]; B=X2.shape[0]
            v1b=args[1] if args[1] is not None else v1r
            vv=at.c_v(X2).view(B,T,NH,128)
            vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
            c2,s2=at.rotary(at.c_q(X2).view(B,T,NH,128))
            def r2(w):
                return are(F.rms_norm(w(X2).view(B,T,NH,128),
                                      (128,)),c2,s2)
            qq,kk=r2(at.c_q),r2(at.c_k)
            q22,k22=r2(at.c_q2),r2(at.c_k2)
            sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                            kk.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                             k22.float())/128
            p2=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
            zz[:,HD]=zz[:,HD].mean(dim=(0,1),keepdim=True)
            return (at.c_proj(zz.transpose(1,2).contiguous()
                    .view(B,T,-1).to(X2.dtype)),v1r)
        return fh

    def logits198(HD):
        out=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            hs=[at.register_forward_hook(mk(HD))] if HD is not None \
               else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            out[i:i+B]=lg[:,:,NLID].cpu()
            for h in hs: h.remove()
        return out

    base=logits198(None)
    push6=base-logits198(6)      # positive = head was pushing "\n"
    push2=base-logits198(2)
    nxt=fresh[:,1:257]
    isnl_next=(nxt==NLID)
    cur=fresh[:,:256]
    dens=isnl_next.float().mean(dim=1)
    medd=dens.median()
    hi=(dens>medd)
    print(f'{int(isnl_next.sum())} newline targets over {NFRESH} '
          f'rows | newline density median {medd:.4f}',flush=True)

    # (a) token-conditioned
    bytok={}
    for r in range(NFRESH):
        for q in range(T):
            bytok.setdefault(int(cur[r,q]),[]).append(
                (float(push6[r,q]),bool(hi[r]),bool(isnl_next[r,q])))
    stats=[]
    for tid,v in bytok.items():
        if len(v)<20: continue
        mu=sum(x[0] for x in v)/len(v)
        stats.append({'tok':tid,'repr':repr(cl.d1(tid)),'n':len(v),
                      'mean_push':round(mu,4),
                      'nl_rate':round(sum(1 for x in v if x[2])
                                      /len(v),3)})
    stats.sort(key=lambda s:-s['mean_push'])
    ntypes=len(stats)
    dec=max(1,ntypes//10)
    top_dec=sum(s['mean_push'] for s in stats[:dec])/dec
    med=stats[ntypes//2]['mean_push']
    pa=(med>0.002) and (top_dec>=3*med)

    # (b) document gating, holding the trigger token fixed
    TOPTOK={s['tok'] for s in stats[:dec]}
    ghi=[x[0] for tid in TOPTOK for x in bytok[tid] if x[1]]
    glo=[x[0] for tid in TOPTOK for x in bytok[tid] if not x[1]]
    mhi=sum(ghi)/max(len(ghi),1); mlo=sum(glo)/max(len(glo),1)
    pb=(mlo>0.002) and (mhi>=1.5*mlo)

    # (c)/(null) detector quality
    P=[float(push6[r,q]) for r in range(NFRESH) for q in range(T)
       if isnl_next[r,q]]
    N=[float(push6[r,q]) for r in range(NFRESH) for q in range(T)
       if not isnl_next[r,q]]
    P2=[float(push2[r,q]) for r in range(NFRESH) for q in range(T)
        if isnl_next[r,q]]
    N2=[float(push2[r,q]) for r in range(NFRESH) for q in range(T)
        if not isnl_next[r,q]]
    a6=auc(P,N); a2=auc(P2,N2)
    pc=a6>=0.75; null_ok=a2<0.65
    out={'n_token_types':ntypes,'top_decile_mean_push':round(top_dec,4),
         'median_token_mean_push':round(med,4),
         'top_triggers':stats[:20],'bottom_triggers':stats[-10:],
         'doc_gating':{'high_density_mean':round(mhi,4),
                       'low_density_mean':round(mlo,4),
                       'n_hi':len(ghi),'n_lo':len(glo),
                       'ratio':round(mhi/max(mlo,1e-4),2)},
         'auc_12_6':round(a6,4),'auc_12_2':round(a2,4),
         'mean_push_at_newline':round(sum(P)/max(len(P),1),4),
         'mean_push_elsewhere':round(sum(N)/max(len(N),1),4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'null_ok':bool(null_ok),'runtime_s':time.time()-t0}
    print('\ntop trigger tokens (mean push on "\\n"):')
    for s in stats[:15]:
        print(f"  {s['repr']:>14} n={s['n']:<5} push "
              f"{s['mean_push']:+.4f}  next-is-newline rate "
              f"{s['nl_rate']}")
    print(f"\ntop decile {top_dec:+.4f} vs median token {med:+.4f}")
    print(f"doc gating (top trigger tokens): high-density "
          f"{mhi:+.4f} (n={len(ghi)}) vs low-density {mlo:+.4f} "
          f"(n={len(glo)})")
    print(f"push at newline targets {out['mean_push_at_newline']:+.4f}"
          f" vs elsewhere {out['mean_push_elsewhere']:+.4f}")
    print(f"AUC 12.6 {a6:.4f} | control head 12.2 {a2:.4f}")
    for nm,v in (('a','top-decile token push >= 3x median token'),
                 ('b','same tokens pushed >=1.5x more in '
                      'newline-dense documents'),
                 ('c','12.6 push predicts next-is-newline, AUC>=0.75')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    print(f"NULL (control head 12.2 AUC {a2:.3f} < 0.65): "
          f"{'ok' if null_ok else 'VIOLATED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
