"""BRACKET QUERY SOURCE -- what upstream component sets the
distance? (gap 2 of the bracket circuit, done exactly)
The bracket head 13.8 selects a key at a query-dependent distance
(529, 531). Deleting one score-matrix cell -- its score on the
matching opener -- costs 83% of the head (522). What has never
been located is the QUERY computation: what makes the query point
the right distance back. This closes that gap exactly, by weight
composition rather than ablation.
The key fact, from the user: the query side is LINEAR. For head
13.8 the query is
    q = rotary( rms_norm(W_q X)[head8] )
and rms_norm is a per-position scalar s, rotary is a rotation, so
with X = SUM_i part_i (cl.writer_parts, exact to 1e-7),
    q = SUM_i rotary( s * (W_q part_i)[head8] ) = SUM_i q_i
EXACTLY. The score's first factor at a fixed key k is then
    f1(k) = q . k / 128 = SUM_i (q_i . k)/128
an exact additive decomposition of the selection over the writers
that feed the query -- no ablation, no first-order approximation.
The same holds for the second factor f2 via W_q2. The full score
is f1 * f2, so each factor is attributed to its query-side writers
with the OTHER factor carried as its measured value.
This ranks the query-side writers by their exact contribution to
the score at the matching opener, at close-bracket target
positions, and then LOCALIZES the leading writer: if it is an
attention layer, into which head; the recursion toward the
embedding continues from there next.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: SUM_i q_i reconstructs the real query to 1e-5,
      and SUM_i (q_i.k)/128 reconstructs f1 to 1e-4. Weight
      composition that does not reproduce the model VOIDS the run;
  (a) CONCENTRATION: the top 3 query-side writers carry >= 60% of
      the total absolute contribution to the match-cell f1 score;
  (b) CONTEXT-CARRYING: the single leading query-side writer is an
      ATTENTION layer, not wte and not an MLP. The look-back
      distance depends on what lies between the query and the
      opener, which attention aggregates and a per-token map
      cannot. Reported either way -- the identity is the result;
  (c) LOCALIZE: if the leader is attention layer j, one head of it
      carries >= 40% of that layer's query contribution, decomposed
      by head exactly the same way (the head slices of W_q are
      independent);
  NULL: at POSITION-MATCHED control queries (a nearby position
      where no bracket is closing), the writer that leads at the
      real targets does NOT lead, or its share drops by >= 30%.
      If the same writers dominate everywhere, the ranking is a
      property of the head's query map and not of the closing
      decision.
Absolute contributions reported alongside every share; the score
factors themselves are signed, so contributions are summed in
absolute value for ranking and reported signed for the leaders."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_query_source_results.json'
NFRESH=192
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    cells={}
    for r in range(NFRESH):
        stack=[]; opos=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s)); opos.append(q)
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                if mt is not None:
                    cells.setdefault(r,[]).append((q,mt))
    ncell=sum(len(v) for v in cells.items().__iter__().__next__()[1:]
              ) if False else sum(len(v) for v in cells.values())
    print(f'{ncell} close-bracket target cells with a match',
          flush=True)
    at=m.transformer.h[LJ].attn
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    NW=len(WR)
    # accumulators: signed contribution to f1 and f2 at the match
    # cell, per query-side writer, at targets and at controls
    acc={'nl':{'f1':torch.zeros(NW),'f2':torch.zeros(NW),
               'abs1':torch.zeros(NW),'n':0},
         'ct':{'f1':torch.zeros(NW),'f2':torch.zeros(NW),
               'abs1':torch.zeros(NW),'n':0}}
    err={'q':[],'f1':[]}
    g=torch.Generator().manual_seed(29)

    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
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
        cap={}
        hs.append(at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0])))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        X=cap['X']
        parts=cl.writer_parts(LJ,E,outs,'a')
        tot=sum(parts.values())
        err['q'].append(float((F.rms_norm(tot,(D,))-X.float())
                        .norm()/X.float().norm().clamp_min(1e-9)))
        s=(X.float().norm(dim=-1,keepdim=True)
           /tot.norm(dim=-1,keepdim=True).clamp_min(1e-9))
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        # real rotated q, k, q2, k2 for head HD
        def realrot(W):
            return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                       cq,sq)[:,:,HD].float()
        qr=realrot(at.c_q); kr=realrot(at.c_k)
        q2r=realrot(at.c_q2); k2r=realrot(at.c_k2)
        # per-writer rotated query pieces: q_i = rotary(s_h * (W_q part_i)[HD])
        # the per-head rms scalar s_h from the REAL input
        aq=(at.c_q(X).view(B,T,NH,128)[:,:,HD].float()
            .pow(2).mean(-1,keepdim=True)+1e-12).sqrt()
        aq2=(at.c_q2(X).view(B,T,NH,128)[:,:,HD].float()
             .pow(2).mean(-1,keepdim=True)+1e-12).sqrt()
        def pieces(W,a):
            per=torch.stack([
                W((parts[w]*s).to(X.dtype)).view(B,T,NH,128)[:,:,HD]
                .float() for w in WR],0)          # (NW,B,T,128)
            per=are(per.permute(1,2,0,3),cq,sq).permute(2,0,1,3)
            return per/a[None]                     # (NW,B,T,128)
        qi=pieces(at.c_q,aq); q2i=pieces(at.c_q2,aq2)
        # exactness of q reconstruction
        err['q'].append(float((qi.sum(0)-qr).norm()
                        /qr.norm().clamp_min(1e-9)))
        # controls: a jittered query per target
        for r in rows:
            b=r-i
            for (qpos,mt) in cells[r]:
                # f1 contribution of writer i = q_i(qpos) . k(mt) /128
                c1=torch.einsum('id,d->i',qi[:,b,qpos],kr[b,mt])/128
                c2=torch.einsum('id,d->i',q2i[:,b,qpos],k2r[b,mt])/128
                f1r=float((qr[b,qpos]*kr[b,mt]).sum()/128)
                err['f1'].append(abs(float(c1.sum())-f1r)
                                 /max(abs(f1r),1e-6))
                acc['nl']['f1']+=c1.cpu(); acc['nl']['f2']+=c2.cpu()
                acc['nl']['abs1']+=c1.abs().cpu(); acc['nl']['n']+=1
                # control query: jitter the query position, same key
                jq=min(max(qpos+int(torch.randint(-6,7,(1,),
                       generator=g)),mt+1),T-1)
                cc1=torch.einsum('id,d->i',qi[:,b,jq],kr[b,mt])/128
                acc['ct']['abs1']+=cc1.abs().cpu(); acc['ct']['n']+=1
                acc['ct']['f1']+=cc1.cpu()
    ri=max(err['q']); rf=max(err['f1'])
    print(f'(0) q reconstruction {ri:.3e} | f1 reconstruction '
          f'{rf:.3e}',flush=True)
    p0=(ri<=1e-5 and rf<=1e-4)
    print(f"(0) EXACTNESS: {'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'rel_q':ri,'rel_f1':rf},
                  open(OUT,'w'),indent=1); return
    absn=acc['nl']['abs1']/max(acc['nl']['n'],1)
    order=absn.argsort(descending=True)
    tot_abs=float(absn.sum().clamp_min(1e-9))
    top3=float(absn[order[:3]].sum())/tot_abs
    print(f'\nquery-side writers into head 13.8, by |contribution| '
          f'to the match-cell f1 score:',flush=True)
    signed=acc['nl']['f1']/max(acc['nl']['n'],1)
    for t in order[:8]:
        ti=int(t)
        print(f"  {WR[ti]:>4}: |contrib| {float(absn[ti]):.4f} "
              f"signed {float(signed[ti]):+.4f} "
              f"({100*float(absn[ti])/tot_abs:.1f}%)",flush=True)
    top=WR[int(order[0])]
    # control share for the leader
    absc=acc['ct']['abs1']/max(acc['ct']['n'],1)
    lead_share_nl=float(absn[int(order[0])])/tot_abs
    lead_share_ct=float(absc[int(order[0])]
                        /absc.sum().clamp_min(1e-9))
    pa=top3>=0.60
    pb=top.startswith('a')
    nul=(lead_share_ct<lead_share_nl-0.30*lead_share_nl)
    # (c) localize the leader if attention
    head_share=None; lead_head=None
    if top.startswith('a'):
        LJ2=int(top[1:])
        # decompose the leading attention writer's contribution to
        # the query by its own heads: part 'a{LJ2}' output is
        # sum over heads of c_proj(head z); re-run to split it
        print(f'\nlocalizing {top} into heads (its contribution to '
              f"head 13.8's query)...",flush=True)
        at2=m.transformer.h[LJ2].attn
        hc=torch.zeros(at2.c_q.weight.shape[0]//128)
        n2=0
        for i in range(0,NFRESH,4):
            rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
            if not rows: continue
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            store={}
            h1=at2.register_forward_pre_hook(
                lambda mo_,a_: store.__setitem__('X2',a_[0]))
            hv=at2.register_forward_hook(
                lambda mo_,a_,o_: store.__setitem__('v1',a_[1]))
            cap={}
            hc0=at.register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('X',a_[0]))
            E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            h1.remove(); hv.remove(); hc0.remove()
            X2=store['X2']
            z,_=cl.head_parts(LJ2,X2,store.get('v1'))
            Wp=at2.c_proj.weight.float()
            X=cap['X']
            s=1.0  # per-writer coeff for a{LJ2} into layer 13 is
                   # writer_coeffs(13)['a{LJ2}']
            coef=cl.writer_coeffs(LJ,'a')[f'a{LJ2}']
            aq=(at.c_q(X).view(B,T,NH,128)[:,:,HD].float()
                .pow(2).mean(-1,keepdim=True)+1e-12).sqrt()
            cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
            kr=are(F.rms_norm(at.c_k(X).view(B,T,NH,128),(128,)),
                   cq,sq)[:,:,HD].float()
            NH2=z.shape[1]
            # exact linear split: a{LJ2} output = coef * sum_h
            # c_proj_h(z_h); the query contribution of head hh is
            # rotary( (W_q of that head-output)[HD] / aq ), and the
            # per-head rms scalar aq is a constant from the real
            # input, so this is additive across hh with no
            # approximation.
            for hh in range(NH2):
                zh=z[:,hh].float()
                head_out=coef*(zh@Wp[:,hh*128:(hh+1)*128].T)
                qh_lin=(at.c_q(head_out.to(X.dtype))
                        .view(B,T,NH,128)[:,:,HD].float())/aq
                qh=are(qh_lin,cq,sq)
                for r in rows:
                    bd=r-i
                    for (qpos,mt) in cells[r]:
                        hc[hh]+=abs(float(
                            (qh[bd,qpos]*kr[bd,mt]).sum()/128))
            n2+=1
        hc=hc/max(n2,1)
        lead_head=int(hc.argmax())
        head_share=float(hc[lead_head]/hc.sum().clamp_min(1e-9))
        print(f'  leading head of {top}: {LJ2}.{lead_head} carries '
              f'{100*head_share:.0f}% of its query contribution',
              flush=True)
    pc=(head_share is not None and head_share>=0.40)
    print(f"\n(a) top 3 query writers carry {100*top3:.0f}% "
          f"(>=60%): {'HELD' if pa else 'FAILED'}")
    print(f"(b) leader {top} is an attention layer: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) one head carries >=40% of the leader: "
          f"{'HELD' if pc else 'FAILED/NA'} "
          f"({lead_head if lead_head is not None else '-'})")
    print(f"NULL (leader share drops at controls: {lead_share_nl:.3f}"
          f" -> {lead_share_ct:.3f}): {'ok' if nul else 'VIOLATED'}")
    out={'n_cells':ncell,'rel_q':ri,'rel_f1':rf,
         'writers_by_abs':[(WR[int(t)],round(float(absn[int(t)]),4),
                            round(float(signed[int(t)]),4))
                           for t in order[:10]],
         'top3_share':round(top3,3),'leader':top,
         'leader_share_target':round(lead_share_nl,3),
         'leader_share_control':round(lead_share_ct,3),
         'leader_head':(f'{top[1:]}.{lead_head}'
                        if lead_head is not None else None),
         'leader_head_share':(round(head_share,3)
                              if head_share else None),
         'pred_0':True,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
