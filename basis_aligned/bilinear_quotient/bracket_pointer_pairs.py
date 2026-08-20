"""BRACKET POINTER PAIRS -- what computes the pointer?
523 established what head 13.8 does and left the important
question open. It is a POINTER: at a position where a bracket is
about to close it puts 0.367 of its score mass on the one opener
that bracket closes, and 0.020-0.023 on every other bracket in
the context whether open or closed -- the same share it gives
ordinary text. And 522 showed that single cell is the mechanism:
zeroing it costs 0.689 of the head's 0.825 nats.
So the head's entire contribution passes through ONE NUMBER: the
score at (query = the position before the closer, key = the
matching opener). That is the cleanest object this model offers
for a compositional account, because the tier-4 decomposition of
one scalar is directly interpretable, with no averaging over keys
to blur it.
The algebra is the one verified exactly in 519 (5.14e-7): with
writer parts p_i, rms scalars absorbed into each writer's share
and rotary a rotation,
    score(q,k) = SUM_ij [ (1/128) Q_i(q).K_j(k) ] * factor2(q,k)
an exact 625-term sum. Here it is evaluated at exactly the cells
that matter.
The contrast is what makes it informative. The same decomposition
is computed at the MATCH cell and at a DISTRACTOR cell -- the
nearest non-matching opener, a lexically identical token at a
similar distance that the head gives 15.8x less mass. If the same
writer pairs dominate both, the pointer is not built by the pair
structure and the selection happens elsewhere; if different pairs
dominate, those pairs ARE the pointer.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: writer parts reproduce the layer-13 input to
      1e-4, and the 625-term sum reproduces the head's real score
      at the measured cells to 1e-4 relative. Failure VOIDS;
  (a) SPARSE AT THE MATCH CELL: the top 10 of 625 pairs carry
      >= 50% of the absolute pair mass at the match cell. 519
      measured 12.2% for the newline head averaged over all keys;
      a single well-defined cell is the case where sparsity has
      the best chance;
  (b) THE PAIRS ARE THE POINTER: the top-10 pair set at the match
      cell differs from the top-10 set at the distractor cell by
      >= 4 pairs. This is the falsifiable form of "the pair
      structure selects the match";
  (c) SIGNED SEPARATION: the summed top-10 contribution is larger
      at the match cell than at the distractor cell by >= 2x, in
      absolute terms with both numbers reported.
  NULL: 10 pairs drawn at random carry < 10% of the match-cell
      mass, three draws.
If (a) fails and (b) holds, the pointer is built by a diffuse but
behaviour-specific set of pairs -- worth knowing. If both fail,
the selection is not visible in the pair structure at all, which
would say the matching is computed upstream and merely READ OUT
here, and would make the upstream MLPs the next target."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_pointer_pairs_results.json'
NFRESH=192; TOPK=10
OPENS={'(':')','[':']','{':'}'}
CLOSES={v:k for k,v in OPENS.items()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    cells=[]           # (row, q, match, distractor)
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
                ds=[p for p in opos if p<=q and p!=mt]
                if mt is not None and ds:
                    cells.append((r,q,mt,ds[-1]))
    print(f'{len(cells)} target cells with both a match and a '
          f'distractor',flush=True)
    if len(cells)<25:
        print('*** comparison class unpopulated -- VOID ***')
        json.dump({'void':'too few match/distractor pairs',
                   'n':len(cells)},open(OUT,'w'),indent=1); return
    at=m.transformer.h[LJ].attn
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    NW=len(WR)
    mass={'match':torch.zeros(NW,NW),'dist':torch.zeros(NW,NW)}
    tot_sc={'match':0.0,'dist':0.0}
    err={'input':[],'score':[]}
    by_row={}
    for (r,q,mt,ds) in cells: by_row.setdefault(r,[]).append((q,mt,ds))

    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH)) if r in by_row]
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
        missing=[w for w in WR if w not in parts]
        if missing:
            print(f'*** writer_parts missing {missing} -- VOID ***')
            json.dump({'void':f'missing {missing}'},
                      open(OUT,'w'),indent=1); return
        tot=sum(parts.values())
        err['input'].append(float((F.rms_norm(tot,(D,))-X.float())
                            .norm()/X.float().norm().clamp_min(1e-9)))
        s=(X.float().norm(dim=-1,keepdim=True)
           /tot.norm(dim=-1,keepdim=True).clamp_min(1e-9))
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        P={}
        for nm,W in (('q',at.c_q),('k',at.c_k),
                     ('q2',at.c_q2),('k2',at.c_k2)):
            full=W(X).view(B,T,NH,128)[:,:,HD].float()
            a=full.pow(2).mean(-1,keepdim=True).sqrt().clamp_min(1e-9)
            per=torch.stack([
                W((parts[w]*s).to(X.dtype)).view(B,T,NH,128)[:,:,HD]
                .float() for w in WR],0)
            per=are(per.permute(1,2,0,3),cq,sq).permute(2,0,1,3)
            P[nm]=per/a[None]
        def realf(W1,W2):
            def r_(W):
                return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                           cq,sq)[:,:,HD].float()
            return torch.einsum('bqd,bkd->bqk',r_(W1),r_(W2))/128
        f1r=realf(at.c_q,at.c_k); f2r=realf(at.c_q2,at.c_k2)
        f1=torch.einsum('bqd,bkd->bqk',P['q'].sum(0),
                        P['k'].sum(0))/128
        err['score'].append(float((f1-f1r).norm()
                            /f1r.norm().clamp_min(1e-9)))
        for r in rows:
            b=r-i
            for (q,mt,ds) in by_row[r]:
                for nm,k in (('match',mt),('dist',ds)):
                    pr=torch.einsum('id,jd->ij',P['q'][:,b,q],
                                    P['k'][:,b,k])
                    term=(pr/128)*float(f2r[b,q,k])
                    mass[nm]+=term.abs().cpu()
                    tot_sc[nm]+=abs(float(f1r[b,q,k]*f2r[b,q,k]))
    ri=max(err['input']); rs=max(err['score'])
    print(f'(0) input rel err {ri:.3e} | score rel err {rs:.3e}')
    p0=(ri<=1e-4 and rs<=1e-4)
    print(f"(0) EXACTNESS: {'HELD' if p0 else 'FAILED -- RUN VOID'}")
    if not p0:
        json.dump({'pred_0':False,'rel_input':ri,'rel_score':rs},
                  open(OUT,'w'),indent=1); return
    def top(mat,k=TOPK):
        fl=mat.flatten(); idx=fl.argsort(descending=True)[:k]
        return ([(WR[int(t)//NW],WR[int(t)%NW],round(float(fl[t]),4))
                 for t in idx],
                {(int(t)//NW,int(t)%NW) for t in idx},
                float(fl[idx].sum()))
    tm,sm,summ=top(mass['match']); td,sd,sumd=top(mass['dist'])
    shm=summ/float(mass['match'].sum().clamp_min(1e-9))
    shd=sumd/float(mass['dist'].sum().clamp_min(1e-9))
    print(f'\ntop {TOPK} pairs at the MATCH cell '
          f'({shm*100:.1f}% of pair mass):')
    for a,b,v in tm: print(f'   {a:>4} x {b:<4} {v}')
    print(f'top {TOPK} at the DISTRACTOR cell ({shd*100:.1f}%):')
    for a,b,v in td: print(f'   {a:>4} x {b:<4} {v}')
    diff=len(sm-sd)
    gg=torch.Generator().manual_seed(7); rnd=[]
    fl=mass['match'].flatten()
    for _ in range(3):
        pick=torch.randperm(NW*NW,generator=gg)[:TOPK]
        rnd.append(round(float(fl[pick].sum()
                   /mass['match'].sum().clamp_min(1e-9)),4))
    va,_=cl.score_bar('a',shm,0.50)
    print(f'(b) top-10 sets differ by {diff} pairs (bar 4): '
          f"{'HELD' if diff>=4 else 'FAILED'}")
    vc,_=cl.score_bar('c',summ-2*sumd,1e-9)
    print(f'   summed top-10 contribution: match {summ:.4g} vs '
          f'distractor {sumd:.4g}')
    nul=max(rnd)<0.10
    print(f"NULL (random 10 pairs carry < 10%: {rnd}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'pred_0':True,'rel_input':ri,'rel_score':rs,
         'n_cells':len(cells),
         'top_match':tm,'top_distractor':td,
         'match_top10_share':round(shm,4),
         'dist_top10_share':round(shd,4),
         'pairs_differing':diff,
         'summed_top10_match':round(summ,4),
         'summed_top10_dist':round(sumd,4),
         'random_shares':rnd,
         'pred_a':va=='HELD','pred_b':bool(diff>=4),
         'pred_c':vc=='HELD','null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
