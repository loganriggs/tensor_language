"""BRACKET QUERY GENERIC -- does the query CARRY the distance, or
does it emerge from geometry? (completes the bracket mechanism)
The bracket head reads a 16-dim selection subspace (555) that is
written diffusely and generically (558), so there is no sparse
upstream source. Rotary-off collapses the selection (529), which
says position matters -- but that leaves two possibilities the
program has not separated:
  QUERY-CARRIED: the query encodes a specific look-back DISTANCE
    (a rotary-phase preference), computed per position from
    context. Replacing it with a generic query destroys selection.
  EMERGENT: the query is generic and the selection comes from the
    keys (opener embeddings) under rotary, so a generic query at a
    bracket position still finds the matching opener.
Decisive test: at close-bracket targets, replace head 13.8's query
with a POSITION-INDEPENDENT generic query and re-price the head's
bracket effect and its score mass on the matching opener.
  mean_all    query = mean query over all positions
  mean_brk    query = mean query over bracket-target positions
  mean_in_S   query = target query with its component OUTSIDE the
              selection subspace S removed (keeps only the 16-dim
              part) -- tests whether S is sufficient
  real        untouched (reference)
If mean_brk still selects the matching opener, the distance is not
in the per-position query content and the circuit is largely
geometric. If it fails, the query genuinely carries the distance
and 555's subspace is where that distance lives.
REGISTERED PREDICTIONS:
  (0) SANITY: the real arm reproduces the head's score mass on the
      matching opener (>= 0.30, from 523's 0.367). VOIDS otherwise;
  (a) THE QUERY CARRIES IT: mean_all drops the matching-opener
      share below 0.15 (from ~0.37) and raises the head's bracket
      CE cost by >= 0.30 nats. A generic query should NOT select;
  (b) S IS SUFFICIENT: mean_in_S keeps the matching-opener share
      within 0.10 of the real query. The 16-dim subspace carries
      the selection and the other 1136 dimensions do not;
  (c) NOT A CONSTANT-QUERY ARTIFACT: mean_brk (the bracket-average
      query) does NOT recover selection -- its share stays below
      0.20. If averaging over bracket positions preserved
      selection, the distance would be a fixed offset, which 531
      already refuted, so this is a consistency check;
  NULL: at NON-bracket control positions, swapping in the generic
      query changes CE by < 0.02 nats -- the intervention only
      matters where the head is doing its job."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_query_generic_results.json'
NFRESH=128
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); match={}
    for r in range(NFRESH):
        stack=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s))
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                if mt is not None: tgt[r,q]=True; match[(r,q)]=mt
    at=m.transformer.h[LJ].attn
    # selection subspace S (as in 555)
    Wq=at.c_q.weight.float()[HD*128:(HD+1)*128]
    G=torch.zeros(D,D,device=DEV); ng=0; cap={}
    # also gather mean pre-rotary query (all positions and bracket)
    qsum=torch.zeros(128,device=DEV); qn=0
    qbsum=torch.zeros(128,device=DEV); qbn=0
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hc=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hc.remove()
        X=cap['X']
        qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),(128,))[:,:,HD] \
            .float()                                # pre-rotary
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        kr=are(F.rms_norm(at.c_k(X).view(B,T,NH,128),(128,)),
               cq,sq)[:,:,HD].float()
        qsum+=qpre.reshape(-1,128).sum(0); qn+=qpre.reshape(-1,128).shape[0]
        for r in range(i,min(i+4,NFRESH)):
            if not tgt[r].any(): continue
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                mt=match[(r,q)]
                dk=kr[b,mt]-kr[b,:q+1].mean(0)
                grad=Wq.T@dk; G+=torch.outer(grad,grad); ng+=1
                qbsum+=qpre[b,q]; qbn+=1
    G/=max(ng,1)
    evals,evecs=torch.linalg.eigh(G)
    S=evecs[:,evals.argsort(descending=True)[:16]]
    q_mean_all=qsum/max(qn,1); q_mean_brk=qbsum/max(qbn,1)
    print(f'{int(tgt.sum())} targets | subspace + generic queries '
          f'built',flush=True)

    def run(mode):
        ce=torch.zeros(NFRESH,T); share=[0.0,0]
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            def fh(mo,args,o_):
                y,v1r=o_; X=args[0]
                v1b=args[1] if args[1] is not None else v1r
                z,vm=cl.head_parts(LJ,X,v1b); z=z.clone()
                cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
                qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),
                                (128,))[:,:,HD].float()
                if mode=='mean_all':
                    qp=q_mean_all[None,None].expand(B,T,128).clone()
                elif mode=='mean_brk':
                    qp=q_mean_brk[None,None].expand(B,T,128).clone()
                elif mode=='mean_in_S':
                    # project X onto S for the query only
                    Xs=(X.float()@S)@S.T
                    qp=F.rms_norm(at.c_q(Xs.to(X.dtype))
                        .view(B,T,NH,128),(128,))[:,:,HD].float()
                else:
                    qp=qpre
                # apply rotary to the (possibly generic) query, but
                # only at target positions; elsewhere keep real
                qrot=are(qp[:,:,None].expand(B,T,NH,128).contiguous(),
                         cq,sq)[:,:,HD]
                qreal=are(qpre[:,:,None].expand(B,T,NH,128).contiguous(),
                          cq,sq)[:,:,HD]
                use=torch.zeros(B,T,1,device=DEV)
                for r in range(i,min(i+4,NFRESH)):
                    if not tgt[r].any(): continue
                    for q in tgt[r].nonzero().squeeze(1).tolist():
                        use[r-i,q,0]=1.0
                qfin=torch.where(use.bool(),qrot,qreal)
                def rk(W):
                    return are(F.rms_norm(W(X).view(B,T,NH,128),
                               (128,)),cq,sq)[:,:,HD].float()
                q2=are(F.rms_norm(at.c_q2(X).view(B,T,NH,128),
                       (128,))[:,:,HD][:,:,None].expand(B,T,NH,128)
                       .contiguous(),cq,sq)[:,:,HD] \
                   if False else rk(at.c_q2)
                s1=torch.einsum('bqd,bkd->bqk',qfin.float(),
                                rk(at.c_k))/128
                s2=torch.einsum('bqd,bkd->bqk',q2.float(),
                                rk(at.c_k2))/128
                sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
                den=sc.abs().sum(-1).clamp_min(1e-6)
                for r in range(i,min(i+4,NFRESH)):
                    if not tgt[r].any(): continue
                    b=r-i
                    for q in tgt[r].nonzero().squeeze(1).tolist():
                        mt=match[(r,q)]
                        share[0]+=abs(float(sc[b,q,mt]/den[b,q]))
                        share[1]+=1
                z[:,HD]=torch.einsum('bqk,bkd->bqd',sc,
                                     vm[:,:,HD].float())
                return (at.c_proj(z.transpose(1,2).contiguous()
                        .view(B,T,-1).to(X.dtype)),v1r)
            h=at.register_forward_hook(fh)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            h.remove()
        return (float(ce[tgt].mean()),float(ce[~tgt].mean()),
                share[0]/max(share[1],1))

    # baseline with no head hook
    ce=torch.zeros(NFRESH,T)
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                  reduction='none').view(B,T).cpu()
    base_t=float(ce[tgt].mean()); base_o=float(ce[~tgt].mean())
    res={}
    for mode in ('real','mean_all','mean_brk','mean_in_S'):
        mt,ot,sh=run(mode)
        res[mode]={'target_cost':round(mt-base_t,4),
                   'other_cost':round(ot-base_o,4),
                   'match_share':round(sh,4)}
        print(f"{mode:>10}: match share {sh:.4f} | target dCE "
              f"{mt-base_t:+.4f} | other dCE {ot-base_o:+.4f}",
              flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    p0=res['real']['match_share']>=0.30
    pa=(res['mean_all']['match_share']<0.15
        and res['mean_all']['target_cost']>=0.30)
    pb=abs(res['mean_in_S']['match_share']
           -res['real']['match_share'])<0.10
    pc=res['mean_brk']['match_share']<0.20
    nul=abs(res['mean_all']['other_cost'])<0.02
    print(f"\n(0) real match share {res['real']['match_share']:.3f} "
          f">=0.30: {'HELD' if p0 else 'FAILED -- VOID'}")
    print(f"(a) generic query destroys selection: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) S-only query preserves selection "
          f"({res['mean_in_S']['match_share']:.3f} vs "
          f"{res['real']['match_share']:.3f}): "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) bracket-avg query does not recover selection: "
          f"{'HELD' if pc else 'FAILED'}")
    print(f"NULL (generic query inert off-brackets "
          f"{res['mean_all']['other_cost']:+.4f}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_target':round(base_t,4),'results':res,
         'pred_0':bool(p0),'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
