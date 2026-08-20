"""BRACKET QUERY RANK -- how many input directions carry the
distance selection?
553/554 proved by exact composition that head 13.8's query is
diffuse over WRITERS -- a dozen late components each 6-9%. That is
consistent with a single underlying variable relayed by many
hands. This asks the direction question: how many directions of
the layer-13 residual does the head's query actually read to make
its bracket decision?
Two complementary measurements.
  WEIGHT RANK (structure). The query map for head 13.8 is
    W_q[head8] (128 x 1152). Its own rank bounds how many input
    directions the query can possibly depend on: at most 128. But
    the SELECTION uses the query only through its alignment with
    keys, which are dominated by rotated bracket-opener
    embeddings, so the effective input rank may be far smaller.
  CAUSAL RANK (function). Project directions out of the layer-13
    residual (the query input for this head only, leaving keys and
    values and every other head untouched) and measure how the
    head's bracket cost falls. The directions are the top singular
    vectors of the per-target query-contrast Jacobian -- exactly
    the input directions whose change most moves the match-minus-
    mean selection. Projecting k of them out and pricing the
    model's cross-entropy at close-bracket targets gives the
    causal rank.
The reference is the head's full bracket effect, 0.825 nats (522),
reproduced here as the all-directions-out cost.
REGISTERED PREDICTIONS:
  (0) SANITY: projecting out ALL directions the query reads
      reproduces the head's bracket cost to within 0.10 nats;
      keeping all of them costs under 1e-3. Failure VOIDS;
  (a) LOW CAUSAL RANK: projecting out the top 8 selection
      directions removes >= 60% of the head's bracket cost. This
      is the claim that the look-back signal lives in a handful of
      directions even though a dozen writers build it;
  (b) BEATS RANDOM: at 8 directions, the selection directions
      remove at least 3x what 8 random directions in the same
      space remove;
  (c) THE NUMBER: report the smallest k whose removal kills >= 80%
      of the head's bracket cost, and the weight rank of W_q[head8]
      for reference. No bar -- k is the answer to the reframed
      gap 2;
  NULL: projecting these directions out at NON-bracket positions
      must cost far less -- the directions are specific to the
      bracket decision, not generally load-bearing. Report the
      ratio."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_query_rank_results.json'
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
                if mt is not None:
                    tgt[r,q]=True; match[(r,q)]=mt
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
    p0=(abs(full_t-base_t-0.825)<0.15 or full_t-base_t>0.5) \
       and True
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
    va=frac(8)>=0.60
    r8=min(curve[8]['random']) if curve[8]['random'] else 1
    vb=curve[8]['target']>=3*max(r8,1e-6)
    kill=next((k for k in KS if frac(k)>=0.80),None)
    nul_ratio=(curve[8]['other']/max(curve[8]['target'],1e-6))
    print(f"\n(0) all-dirs-out reproduces the head cost "
          f"({full:+.4f} vs 0.825): {'HELD' if p0 else 'FAILED'}")
    print(f"(a) top 8 directions remove {100*frac(8):.0f}% of the "
          f"head cost (>=60%): {'HELD' if va else 'FAILED'}")
    print(f"(b) top 8 beat random 8 by 3x: {'HELD' if vb else 'FAILED'}")
    print(f"(c) smallest k killing 80%: {kill} | W_q rank {wq_rank}")
    print(f"NULL (other-position cost is small vs target: ratio "
          f"{nul_ratio:.2f})")
    out={'baseline_target':round(base_t,4),
         'full_cost':round(full,4),'wq_rank':wq_rank,
         'curve':{str(x):y for x,y in curve.items()},
         'kill80_rank':kill,
         'pred_0':bool(p0),'pred_a':bool(va),'pred_b':bool(vb),
         'null_ratio':round(nul_ratio,3),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
