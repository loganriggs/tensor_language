"""NEWLINE QUERY RANK -- does the bracket circuit's shape
generalize, and do the two structural heads share a subspace?
555 showed the bracket head's distance selection is diffuse over
writers but a compact ~16-dim bracket-specific subspace over
directions. 556 found that subspace largely private, with one weak
lead: the newline head 12.6 was its second-ranked geometric
co-reader. This applies the exact selection-subspace and
causal-rank measurement to head 12.6 on newline targets, and
compares the two selection subspaces directly.
For the newline head the analog of the bracket's matching opener
is the most recent PRECEDING newline key (from writeup 497, where
12.6 puts 3.7x more score mass there at newline targets). The
selection directions are the top eigenvectors of the query-contrast
gradient W_q^T (k_recent_newline - mean_k) over newline targets;
projecting them out of the layer-12 query input and pricing the
model gives the causal rank, exactly as for brackets.
REGISTERED PREDICTIONS:
  (0) SANITY: projecting out all directions reproduces the newline
      head's own effect (>= 0.04 nats at newline targets; 497
      measured 0.068). VOIDS otherwise;
  (a) SAME SHAPE: 16 selection directions remove >= 60% of the
      newline head's effect and beat 16 random directions by >= 3x
      -- i.e. the newline head ALSO reads a compact
      behaviour-specific subspace, making "diffuse writers, low-
      rank subspace" a general property rather than a bracket
      quirk;
  (b) SPECIFICITY: removing the newline selection subspace costs
      far less at non-newline positions than at newline targets;
  (c) THE OVERLAP: report the principal angles between the newline
      selection subspace and the bracket selection subspace
      (recomputed here). Mean cosine above 2x the random floor
      (0.118) is a shared structural signal; near the floor means
      the two heads compute unrelated selections. No bar -- the
      number settles 556's lead;
  NULL: the newline selection directions removed at NON-newline
      positions cost < 1/3 of their cost at newline targets."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; NH=9; NLID=198
BLJ=13; BHD=8  # bracket head, for the subspace compare
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_query_rank_results.json'
NFRESH=128
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}
KS=[1,2,4,8,16,32,64]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); match={}
    isnl=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            if chr(10) in cl.d1(int(cur[r,q])): isnl[r,q]=True
    for r in range(NFRESH):
        keys=isnl[r].nonzero().squeeze(1).tolist()
        for q in range(T):
            if chr(10) in cl.d1(int(nxt[r,q])):
                prev=[k for k in keys if k<q]
                if prev:
                    tgt[r,q]=True; match[(r,q)]=prev[-1]
    at=m.transformer.h[LJ].attn
    # ---- selection Jacobian directions ----
    # for each target, the query-contrast gradient wrt the layer-13
    # input is W_q[head8]^T (rotated) applied to (k_match - mean_k);
    # accumulate the outer structure and take its top singular dirs
    Wq=at.c_q.weight.float()[HD*128:(HD+1)*128]      # (128, D)
    G=torch.zeros(D,D,device=DEV); ng=0
    cap={}
    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH))
              if tgt[r].any()]
        if not rows: continue
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hcap=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hcap.remove()
        X=cap['X']
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        kr=are(F.rms_norm(at.c_k(X).view(B,T,NH,128),(128,)),
               cq,sq)[:,:,HD].float()
        for r in rows:
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                mt=match[(r,q)]
                # dk is the (rotated) key direction the query
                # must align with; the input-gradient of the query's
                # alignment is W_q^T dk. Rotary is a per-position
                # orthogonal map on the query, so it leaves the
                # gradient's SVD structure invariant and folds out.
                dk=kr[b,mt]-kr[b,:q+1].mean(0)     # (128,)
                grad=Wq.T@dk                        # (D,)
                G+=torch.outer(grad,grad); ng+=1
    G/=max(ng,1)
    evals,evecs=torch.linalg.eigh(G)
    idxr=evals.argsort(descending=True)
    B_dir=evecs[:,idxr]                              # (D, D) columns
    S_nl=B_dir[:,:16]
    # bracket selection subspace, same construction on close-brackets
    OPENS2={'(':')','[':']','{':'}'}; CLOSES2={v:k for k,v in OPENS2.items()}
    batt=m.transformer.h[BLJ].attn
    Wqb=batt.c_q.weight.float()[BHD*128:(BHD+1)*128]
    Gb=torch.zeros(D,D,device=DEV); ngb=0; capb={}
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        bcells={}
        for r in range(i,min(i+4,NFRESH)):
            stack=[]
            for q in range(T):
                cc=cl.d1(int(cur[r,q])).strip()
                if cc in OPENS2: stack.append((q,cc))
                elif cc in CLOSES2 and stack: stack.pop()
                nn=cl.d1(int(nxt[r,q])).strip()
                if nn in CLOSES2:
                    mtb=None
                    for p,ch in reversed(stack):
                        if OPENS2[ch]==nn: mtb=p; break
                    if mtb is not None: bcells.setdefault(r,[]).append((q,mtb))
        if not bcells: continue
        hcb=batt.register_forward_pre_hook(
            lambda mo_,a_: capb.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hcb.remove()
        Xb=capb['X']
        cqb,sqb=batt.rotary(batt.c_q(Xb).view(B,T,NH,128))
        krb=are(F.rms_norm(batt.c_k(Xb).view(B,T,NH,128),(128,)),
                cqb,sqb)[:,:,BHD].float()
        for r in bcells:
            b=r-i
            for (q,mtb) in bcells[r]:
                dkb=krb[b,mtb]-krb[b,:q+1].mean(0)
                gb=Wqb.T@dkb; Gb+=torch.outer(gb,gb); ngb+=1
    Gb/=max(ngb,1)
    evb,evcb=torch.linalg.eigh(Gb)
    S_br=evcb[:,evb.argsort(descending=True)[:16]]
    cs_cross=torch.linalg.svdvals(S_nl.T@S_br)
    print(f'newline vs bracket selection subspace: mean cos '
          f'{float(cs_cross.mean()):.3f}, max {float(cs_cross.max()):.3f}'
          f' (random floor 0.118)',flush=True)
    wq_rank=int((torch.linalg.svdvals(Wq)>1e-4
                 *torch.linalg.svdvals(Wq)[0]).sum())
    print(f'{int(tgt.sum())} targets | W_q[head8] rank {wq_rank} | '
          f'top eigenvalue share '
          f'{float(evals[idxr[0]]/evals.clamp_min(0).sum()):.3f}',
          flush=True)

    def price(k=None,random=False,seed=0,at_targets=True):
        if k is not None and k>0:
            if random:
                g=torch.Generator(device=DEV).manual_seed(seed)
                Q,_=torch.linalg.qr(torch.randn(D,k,generator=g,
                                                device=DEV))
            else:
                Q=B_dir[:,:k]
            P=Q@Q.T
        else:
            P=None
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if k is not None:
                def fh(mo,args,o_):
                    y,v1r=o_; X=args[0]
                    v1b=args[1] if args[1] is not None else v1r
                    z,vm=cl.head_parts(LJ,X,v1b); z=z.clone()
                    Xq=(X.float()-(X.float()@P)).to(X.dtype) \
                       if P is not None else X
                    cq,sq=at.rotary(at.c_q(Xq).view(B,T,NH,128))
                    def rq(W,Z):
                        return are(F.rms_norm(W(Z).view(B,T,NH,128),
                                              (128,)),cq,sq)[:,:,HD] \
                               .float()
                    ck,sk=at.rotary(at.c_q(X).view(B,T,NH,128))
                    def rk(W):
                        return are(F.rms_norm(W(X).view(B,T,NH,128),
                                              (128,)),ck,sk)[:,:,HD] \
                               .float()
                    s1=torch.einsum('bqd,bkd->bqk',rq(at.c_q,Xq),
                                    rk(at.c_k))/128
                    s2=torch.einsum('bqd,bkd->bqk',rq(at.c_q2,Xq),
                                    rk(at.c_k2))/128
                    sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
                    z[:,HD]=torch.einsum('bqk,bkd->bqd',sc,
                                         vm[:,:,HD].float())
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        m_=float(ce[tgt].mean()); o_=float(ce[~tgt].mean())
        return m_,o_

    base_t,base_o=price(None)
    full_t,full_o=price(D)
    print(f'baseline CE at targets {base_t:.4f} | all-dirs-out '
          f'{full_t:.4f} (delta {full_t-base_t:+.4f})',flush=True)
    p0=(full_t-base_t)>=0.04
    # keep-all check: k=0 handled as None already is base
    curve={}
    for k in KS:
        mt,ot=price(k)
        rnd=[]
        if k==8:
            for sdv in (1,2,3):
                rt,_=price(k,True,sdv); rnd.append(round(rt-base_t,4))
        curve[k]={'target':round(mt-base_t,4),
                  'other':round(ot-base_o,4),'random':rnd}
        print(f'k={k:>3}: target {mt-base_t:+.4f} | other '
              f'{ot-base_o:+.4f}'
              +(f' | random {rnd}' if rnd else ''),flush=True)
        json.dump({str(x):y for x,y in curve.items()},
                  open(OUT,'w'),indent=1)
    full=full_t-base_t
    def frac(k): return curve[k]['target']/max(full,1e-6)
    va=frac(16)>=0.60
    r8=min(curve[8]['random']) if curve[8]['random'] else 1
    vb=curve[8]['target']>=3*max(r8,1e-6)
    kill=next((k for k in KS if frac(k)>=0.80),None)
    nul_ratio=(curve[8]['other']/max(curve[8]['target'],1e-6))
    print(f"\n(0) all-dirs-out reproduces the newline head effect "
          f"({full:+.4f} >= 0.04): {'HELD' if p0 else 'FAILED'}")
    print(f"(a) top 16 directions remove {100*frac(16):.0f}% of the "
          f"head effect (>=60%) and 8 beat random 3x: "
          f"{'HELD' if (va and vb) else 'FAILED'}")
    print(f"(c) newline-vs-bracket subspace mean cos "
          f"{float(cs_cross.mean()):.3f}")
    print(f"(b) top 8 beat random 8 by 3x: {'HELD' if vb else 'FAILED'}")
    print(f"(c) smallest k killing 80%: {kill} | W_q rank {wq_rank}")
    print(f"NULL (other-position cost is small vs target: ratio "
          f"{nul_ratio:.2f})")
    out={'baseline_target':round(base_t,4),
         'full_cost':round(full,4),'wq_rank':wq_rank,
         'curve':{str(x):y for x,y in curve.items()},
         'kill80_rank':kill,
         'newline_vs_bracket_meancos':round(float(cs_cross.mean()),3),
         'newline_vs_bracket_maxcos':round(float(cs_cross.max()),3),
         'pred_0':bool(p0),'pred_a':bool(va),'pred_b':bool(vb),
         'null_ratio':round(nul_ratio,3),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
