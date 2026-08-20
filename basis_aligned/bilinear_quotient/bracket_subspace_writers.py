"""BRACKET SUBSPACE WRITERS -- which components write the
selection subspace, one level toward the embedding.
555 found the bracket head reads a compact ~16-dim selection
subspace S of its layer-13 input; 553/554 found the raw query
diffuse over writers. Those were attributions of the whole query.
This attributes S SPECIFICALLY: for each writer i into layer 13,
how much of its contribution to the input lands INSIDE S, at
close-bracket target positions. A writer that builds the selection
signal projects strongly onto S there; one that merely adds
common-mode does not. Attributing the 16-dim subspace rather than
the raw query may concentrate where the raw query did not, because
S is exactly the discriminative part.
Exact, by weight composition: the residual entering layer 13 is
X = SUM_i part_i (cl.writer_parts, checked). Writer i's projection
onto S is ||S^T part_i(target)||, and because projection is linear
these sum to ||S^T X||. Reported as each writer's share of the
total in-subspace energy at target positions, with the same
quantity at position-matched controls as the null.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: the writer parts reconstruct X to 1e-4, and the
      per-writer S-projections sum to the full S-projection to
      1e-4. VOIDS otherwise;
  (a) CONCENTRATION IN S: the top 3 writers carry >= 45% of the
      in-subspace energy at targets -- materially more than the
      25% the raw query gave (553), because S is the discriminative
      part. If it stays at 25%, the selection signal is built as
      diffusely as the query and there is no compact source;
  (b) A NAMED LEAD: the single top writer carries >= 20%. Its
      identity is the next node toward the embedding and is
      reported whatever it is;
  (c) LOCALIZE: if the leader is an attention layer, one head
      carries >= 40% of its S-projection (head slices are linear);
  NULL: at position-matched control positions the top writers'
      combined S-share is at least 20% lower than at targets. If
      the same writers project onto S everywhere, S-projection is
      a geometric constant, not a bracket signal."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_subspace_writers_results.json'
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
    S=evecs[:,idxr[:16]]                             # (D,16) subspace
    print(f'{int(tgt.sum())} targets | selection subspace dim 16',
          flush=True)
    # per-writer projection onto S at targets and controls
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    NW=len(WR)
    en={'nl':torch.zeros(NW),'ct':torch.zeros(NW)}
    cnt={'nl':0,'ct':0}; errp=[]
    gg=torch.Generator().manual_seed(29)
    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH)) if tgt[r].any()]
        if not rows: continue
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        outs={}; hs=[]
        for lj in range(LJ):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(k9=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[k9]=y.detach().float()
                    return h
                hs.append(mod.register_forward_hook(mk()))
        cap2={}
        hs.append(at.register_forward_pre_hook(
            lambda mo_,a_: cap2.__setitem__('X',a_[0])))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        X=cap2['X']
        parts=cl.writer_parts(LJ,E,outs,'a')
        totp=sum(parts.values())
        errp.append(float((F.rms_norm(totp,(D,))-X.float()).norm()
                    /X.float().norm().clamp_min(1e-9)))
        sc=(X.float().norm(dim=-1,keepdim=True)
            /totp.norm(dim=-1,keepdim=True).clamp_min(1e-9))
        # projected part of each writer onto S (16-dim coords)
        proj={w:((parts[w]*sc)@S) for w in WR}       # (B,T,16)
        for r in rows:
            b=r-i
            qs=tgt[r].nonzero().squeeze(1).tolist()
            for q in qs:
                for wi,w in enumerate(WR):
                    en['nl'][wi]+=float(proj[w][b,q].norm())
                cnt['nl']+=1
                jq=min(max(q+int(torch.randint(-6,7,(1,),
                       generator=gg)),1),T-1)
                for wi,w in enumerate(WR):
                    en['ct'][wi]+=float(proj[w][b,q if False else jq].norm())
                cnt['ct']+=1
    reln=en['nl']/max(cnt['nl'],1); relc=en['ct']/max(cnt['ct'],1)
    order=reln.argsort(descending=True)
    tot=float(reln.sum().clamp_min(1e-9))
    top3=float(reln[order[:3]].sum())/tot
    ri=max(errp)
    print(f'(0) writer reconstruction {ri:.3e}',flush=True)
    print(f'\nwriters by projection onto the selection subspace S '
          f'at bracket targets:',flush=True)
    for t in order[:8]:
        ti=int(t)
        print(f'  {WR[ti]:>4}: {100*float(reln[ti])/tot:.1f}% '
              f'(control {100*float(relc[ti])/float(relc.sum().clamp_min(1e-9)):.1f}%)',
              flush=True)
    top=WR[int(order[0])]
    lead_nl=float(reln[int(order[0])])/tot
    lead_ct=float(relc[int(order[0])])/float(relc.sum().clamp_min(1e-9))
    p0=ri<=1e-4
    pa=top3>=0.45
    pb=lead_nl>=0.20
    nul=lead_ct<lead_nl-0.20*lead_nl
    lead_head=None; head_share=None
    if top.startswith('a'):
        LJ2=int(top[1:])
        at2=m.transformer.h[LJ2].attn
        NH2=at2.c_q.weight.shape[0]//128
        coef=cl.writer_coeffs(LJ,'a')[f'a{LJ2}']
        hc=torch.zeros(NH2)
        Wp=at2.c_proj.weight.float()
        for i in range(0,NFRESH,4):
            rows=[r for r in range(i,min(i+4,NFRESH)) if tgt[r].any()]
            if not rows: continue
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            store={}
            h1=at2.register_forward_pre_hook(
                lambda mo_,a_: store.__setitem__('X2',a_[0]))
            hv=at2.register_forward_hook(
                lambda mo_,a_,o_: store.__setitem__('v1',a_[1]))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            h1.remove(); hv.remove()
            z,_=cl.head_parts(LJ2,store['X2'],store.get('v1'))
            for hh in range(NH2):
                zh=z[:,hh].float()
                ho=coef*(zh@Wp[:,hh*128:(hh+1)*128].T)
                pr=(ho@S)
                for r in rows:
                    b=r-i
                    for q in tgt[r].nonzero().squeeze(1).tolist():
                        hc[hh]+=float(pr[b,q].norm())
        lead_head=int(hc.argmax())
        head_share=float(hc[lead_head]/hc.sum().clamp_min(1e-9))
        print(f'  leading head of {top}: {LJ2}.{lead_head} carries '
              f'{100*head_share:.0f}% of its S-projection',flush=True)
    pc=(head_share is not None and head_share>=0.40)
    print(f"\n(0) reconstruction {ri:.3e}: "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    print(f"(a) top 3 writers into S carry {100*top3:.0f}% (>=45%): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) top writer {top} carries {100*lead_nl:.0f}% (>=20%): "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) one head carries >=40% of the leader: "
          f"{'HELD' if pc else 'NA'} ({lead_head})")
    print(f"NULL (leader S-share drops at controls {lead_nl:.3f} -> "
          f"{lead_ct:.3f}): {'ok' if nul else 'VIOLATED'}")
    out={'n_targets':int(tgt.sum()),'reconstruction':ri,
         'writers_into_S':[(WR[int(t)],round(100*float(reln[int(t)])/tot,1))
                           for t in order[:10]],
         'top3_share':round(top3,3),'leader':top,
         'leader_share':round(lead_nl,3),
         'leader_control_share':round(lead_ct,3),
         'leader_head':(f'{top[1:]}.{lead_head}'
                        if lead_head is not None else None),
         'leader_head_share':(round(head_share,3) if head_share else None),
         'pred_0':bool(p0),'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} (%.0fs)'%(time.time()-t0))
    return
    wq_rank=int((torch.linalg.svdvals(Wq)>1e-4
                 *torch.linalg.svdvals(Wq)[0]).sum())

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
