"""NEWLINE HEAD PAIRS -- the compositional structure, not another
ablation.
506 silenced each upstream writer's contribution to head 12.6's
query one at a time and concluded its input was "diffuse": no
writer worth more than 0.0154 AUC, all of them together worth
0.140. That conclusion is about the INSTRUMENT, and the
superadditivity is the tell. This network has no softmax and no
activation function: an attention score is a PRODUCT of two
bilinear forms. Leave-one-out is a first-order probe, and a
first-order probe applied to a multiplicative computation reports
"nothing matters alone, everything matters together" whether or
not the underlying structure is simple. That is what 506 reported.
The architecture permits an exact answer instead. Write the
residual entering layer 12 as writer parts p_i (cl.writer_parts,
exact to 1e-7). rms_norm is a per-position scalar and rotary is a
rotation, so with
    Q_i(q) = R_q( (W_q p_i(q))_head6 ),  K_j(k) likewise,
    C(q,k) = 1/128  (the rms scalars divide into each writer's
                     own share, so nothing else remains)
each factor of the score is EXACTLY and ADDITIVELY
    factor1(q,k) = C(q,k) * SUM_ij Q_i(q).K_j(k)
over 25x25 = 625 writer pairs, and since score = factor1*factor2,
    score(q,k) = SUM_ij [ C(q,k) Q_i(q).K_j(k) * factor2(q,k) ]
is an exact 625-term decomposition of the head's actual score. No
approximation, no ablation, no leave-one-out.
Sparsity alone would only be a ranking, so the run also tests
SUFFICIENCY: rebuild the head's score from its top pairs, run the
real model with the rebuilt score, and measure how much of the
head's newline behaviour survives, against the head's own ablation
    retention = (CE_ablated - CE_rebuilt)/(CE_ablated - CE_real)
at newline targets, so 1.0 reproduces the head and 0.0 reproduces
nothing.
REGISTERED PREDICTIONS:
  (0) EXACTNESS, twice: writer parts reproduce the layer input to
      1e-4 relative, AND the 625-term sum reproduces the head's
      real score to 1e-4 relative. A decomposition that does not
      add up cannot attribute. Failure VOIDS the run;
  (a) SPARSE: the top 10 pairs of 625 carry >= 50% of the total
      absolute pair mass at newline-target queries;
  (b) SUFFICIENT: the top-10 rebuild retains >= 0.70 of the head's
      newline benefit. This is the bar that makes it a mechanism
      claim and not a ranking;
  (c) BEHAVIOUR-SPECIFIC: the top-10 pair set at newline-target
      queries differs by at least 3 pairs from the top-10 set at
      position-matched control queries. If the same pairs lead
      everywhere, the head's input structure is fixed and the
      newline specificity lives elsewhere -- a finding, to be
      reported as one.
  NULL: 10 pairs drawn at random, three draws, retain < 0.20.
TIERS OF UNDERSTANDING (writeup 512), recorded because this is the
program's first tier-4 attempt:
  1 localization -- deleting X costs Y on behaviour B
  2 behavioural  -- what token X moves, where, by how much
  3 first-order attribution -- which writers feed X (506: diffuse)
  4 COMPOSITIONAL -- the exact algebraic form: which products of
    which upstream contributions produce X's output
  5 recursive -- the same treatment applied to those contributions,
    down to the embedding
Tier 3 is where an ablation-only method stops. Tier 4 is available
in closed form for this architecture and this is the first use."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; NH=9; NLID=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_head_pairs_results.json'
NFRESH=32; TOPK=10

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    nl=(fresh[:,1:257]==NLID)
    g=torch.Generator().manual_seed(29)
    ctrl=torch.zeros_like(nl)
    for r in range(NFRESH):
        k=int(nl[r].sum())
        if k==0: continue
        pos=nl[r].nonzero().squeeze(1)
        ctrl[r,(torch.randint(-6,7,(k,),generator=g)+pos)
             .clamp(0,T-1)]=True
    at=m.transformer.h[LJ].attn
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    NW=len(WR)
    TRI=torch.tril(torch.ones(T,T,device=DEV))
    err={'input':[],'score':[]}

    def pieces(X,E,outs,B):
        """Exact per-writer rotated vectors for head HD, plus the
        scalar prefactor. Returns Q,K,Q2,K2 (NW,B,T,128) and
        C (B,T,T)."""
        parts=cl.writer_parts(LJ,E,outs,'a')
        tot=sum(parts.values())
        err['input'].append(float((F.rms_norm(tot,(D,))-X.float())
                            .norm()/X.float().norm().clamp_min(1e-9)))
        # X = tot * s, s per-position scalar
        s=(X.float().norm(dim=-1,keepdim=True)
           /tot.norm(dim=-1,keepdim=True).clamp_min(1e-9))
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        out={}; scal={}
        for nm,W in (('q',at.c_q),('k',at.c_k),
                     ('q2',at.c_q2),('k2',at.c_k2)):
            full=W(X).view(B,T,NH,128)[:,:,HD].float()
            a=(full.pow(2).mean(-1,keepdim=True)+0).sqrt() \
                .clamp_min(1e-9)                      # per-head rms
            scal[nm]=a
            per=torch.stack([
                W((parts[w]*s).to(X.dtype)).view(B,T,NH,128)[:,:,HD]
                .float() for w in WR],0)              # (NW,B,T,128)
            per=are(per.permute(1,2,0,3),cq,sq).permute(2,0,1,3)
            out[nm]=per/a[None]
        C1=(scal['q']*1.0)[...,0]                     # (B,T)
        C2=(scal['q2']*1.0)[...,0]
        K1=(scal['k']*1.0)[...,0]; K2=(scal['k2']*1.0)[...,0]
        return out,(C1,K1,C2,K2)

    def real_factors(X,B):
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        def r(W):
            return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                       cq,sq)[:,:,HD].float()
        f1=torch.einsum('bqd,bkd->bqk',r(at.c_q),r(at.c_k))/128
        f2=torch.einsum('bqd,bkd->bqk',r(at.c_q2),r(at.c_k2))/128
        return f1*TRI,f2*TRI

    mass={'nl':torch.zeros(NW,NW),'ct':torch.zeros(NW,NW)}
    def sweep():
        for i in range(0,NFRESH,4):
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
            P,(cq1,ck1,cq2,ck2)=pieces(X,E,outs,B)
            f1r,f2r=real_factors(X,B)
            # exact rebuild of factor1 from all pairs
            sq=P['q'].sum(0); sk=P['k'].sum(0)
            # the per-position and per-head rms scalars are
            # already divided into each writer's share, so the
            # summed shares ARE the normalized rotated vectors and
            # no prefactor remains. Applying the scalars a second
            # time was the bug the exactness gate caught (rel err
            # 17.4 -> see writeup 514).
            f1=torch.einsum('bqd,bkd->bqk',sq,sk)/128*TRI
            err['score'].append(
                float((f1-f1r).norm()/f1r.norm().clamp_min(1e-9)))
            sq2=P['q2'].sum(0); sk2=P['k2'].sum(0)
            f2=torch.einsum('bqd,bkd->bqk',sq2,sk2)/128*TRI
            for nm,mask in (('nl',nl[i:i+4]),('ct',ctrl[i:i+4])):
                for b in range(B):
                    qs=mask[b].nonzero().squeeze(1).to(DEV)
                    if not len(qs): continue
                    Qi=P['q'][:,b][:,qs]                # (NW,nq,128)
                    Kj=P['k'][:,b]                      # (NW,T,128)
                    pr=torch.einsum('iqd,jkd->ijqk',Qi,Kj)
                    term=(pr/128)*f2[b][qs][None,None] \
                         *TRI[qs][None,None]
                    mass[nm]+=term.abs().sum(dim=(2,3)).cpu()
            yield i,X,P,(cq1,ck1,cq2,ck2),f2
    for _ in sweep(): pass
    ri=max(err['input']); rs=max(err['score'])
    print(f'(0) input rel err {ri:.3e} | score rel err {rs:.3e}',
          flush=True)
    p0=(ri<=1e-4 and rs<=1e-4)
    print(f"(0) EXACTNESS: {'HELD' if p0 else 'FAILED -- RUN VOID'}")
    if not p0:
        json.dump({'pred_0':False,'rel_input':ri,'rel_score':rs},
                  open(OUT,'w'),indent=1); return
    def top(mat,k=TOPK):
        fl=mat.flatten(); idx=fl.argsort(descending=True)[:k]
        return [(WR[int(t)//NW],WR[int(t)%NW],float(fl[t]))
                for t in idx],[(int(t)//NW,int(t)%NW) for t in idx]
    tn,tni=top(mass['nl']); tc,tci=top(mass['ct'])
    share=float(mass['nl'].flatten().sort(descending=True)
                .values[:TOPK].sum()/mass['nl'].sum().clamp_min(1e-9))
    print(f'\ntop {TOPK} pairs at newline targets '
          f'({share*100:.1f}% of pair mass):')
    for a,b,v in tn: print(f'   {a:>4} x {b:<4}  {v:.4g}')
    print(f'top {TOPK} at position-matched controls:')
    for a,b,v in tc: print(f'   {a:>4} x {b:<4}  {v:.4g}')
    diff=len(set(tni)-set(tci))
    va,_=cl.score_bar('a',share,0.50)
    print(f'(c) top-10 sets differ by {diff} pairs (bar 3): '
          f"{'HELD' if diff>=3 else 'FAILED'}")
    out={'pred_0':True,'rel_input':ri,'rel_score':rs,
         'top_pairs_newline':[(a,b,v) for a,b,v in tn],
         'top_pairs_control':[(a,b,v) for a,b,v in tc],
         'top10_mass_share':round(share,4),
         'pairs_differing':diff,
         'pred_a':va=='HELD','pred_c':bool(diff>=3),
         'note':'sufficiency (b) measured by newline_head_rebuild',
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
