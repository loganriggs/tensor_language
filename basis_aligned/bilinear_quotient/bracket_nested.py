"""BRACKET NESTED -- does head 13.8 track depth, or just find the
most recent opener?
522 is the tightest causal result in this program: leaving head
13.8 intact and zeroing ONE cell of its score matrix -- its score
on the opener that the upcoming bracket closes -- costs 0.689 of
the head's full 0.825 nats, while the same operation on the
nearest NON-matching opener costs 0.0136.
It has one gap, stated when it was recorded. Only 1 of 84 natural
targets had a matching opener that was not also the most recent
opener, because FineWeb prose almost never nests brackets. So
"attends to the match" and "attends to the most recent opener"
are the same hypothesis on that data, and a depth-free rule would
have produced the same numbers.
Constructed text separates them. Two context types, built from the
same filler vocabulary so they differ only in structure:
  OUTER  "... ( aaa ( bbb ) ccc" -> the next ")" closes the FIRST
         opener, and the most recent opener is the second one,
         which is already closed. A depth-free "most recent
         opener" rule points at the wrong token here.
  INNER  "... ( aaa ( bbb" -> the next ")" closes the SECOND
         opener, which IS the most recent. Both hypotheses agree,
         so this is the sanity condition.
Measured for each: the head's signed score-mass share on the
correct match and on the distractor, and the causal cost of
zeroing each of those single cells.
Constructed text is out of distribution, which is a real risk and
gets its own null rather than a footnote: if the model does not
actually predict the closing bracket in these contexts, nothing
measured in them means anything.
REGISTERED PREDICTIONS:
  (0) THE CONTEXTS WORK: base cross-entropy at the target position
      is below 2.0 nats in both conditions, i.e. the model does
      expect a closing bracket there. Failure VOIDS the run;
  (a) DEPTH TRACKING, mass: in OUTER contexts the share on the
      correct (first) opener is >= 2x the share on the closed
      distractor. A most-recent-opener rule predicts the
      opposite ordering;
  (b) DEPTH TRACKING, causally: in OUTER contexts, zeroing the
      correct match costs >= 3x zeroing the distractor;
  (c) SANITY: in INNER contexts, zeroing the match costs >= 0.20
      nats and reproduces the natural-text pattern.
  NULL: zeroing a random earlier non-bracket cell costs < 0.05 in
      both conditions.
If (a) and (b) fail while (c) holds, the head implements "most
recent opener" and does NOT track depth -- a weaker mechanism
that is right in ordinary prose and wrong in nested text, which
would be a precise and reportable description rather than a
failure. Absolute pairs reported throughout; no quotient scored
with a denominator that can approach zero (the rule from 520)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_nested_results.json'

FILL=[' the report',' this study',' our analysis',' the survey',
      ' the paper',' that review',' the dataset',' his account',
      ' the article',' their model',' the figure',' the sample']
MID=[' see figure',' see table',' page 12',' section 3',
     ' as noted',' cf. above',' n = 40',' ref. 7',
     ' see note',' line 5',' vol. 2',' fig. 4']
TAIL=[' for details',' in full',' below',' for context',
      ' as well',' throughout',' here',' in part',
      ' overall',' in turn',' as shown',' at length']
LEAD=('The committee reviewed the evidence carefully and then '
      'concluded that')

@torch.no_grad()
def build(kind,enc):
    """Returns (ids, target_pos, match_pos, distractor_pos)."""
    rows=[]
    for i in range(len(FILL)):
        if kind=='outer':
            txt=(LEAD+FILL[i]+' ('+MID[i]+' ('+TAIL[i]+' )'
                 +FILL[(i+3)%len(FILL)])
        else:
            txt=(LEAD+FILL[i]+' ('+MID[i]+' ('+TAIL[i])
        ids=enc.encode(txt)
        # locate bracket tokens
        op=[j for j,t in enumerate(ids)
            if cl.d1(int(t)).strip()=='(']
        cl_=[j for j,t in enumerate(ids)
             if cl.d1(int(t)).strip()==')']
        if len(op)<2: continue
        if kind=='outer':
            if not cl_: continue
            mt,ds=op[0],op[1]
        else:
            mt,ds=op[1],op[0]
        tp=len(ids)-1          # predict the closer AFTER the last token
        rows.append((ids,tp,mt,ds))
    return rows

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    enc=cl.enc()
    CLOSE=enc.encode(' )')[0] if enc.encode(' )') else None
    CLOSE2=enc.encode(')')[0]
    at=m.transformer.h[LJ].attn
    res={}
    for kind in ('outer','inner'):
        rows=build(kind,enc)
        if len(rows)<8:
            print(f'*** {kind}: only {len(rows)} contexts built -- '
                  f'unpopulated, VOID ***')
            json.dump({'void':f'{kind} unpopulated'},
                      open(OUT,'w'),indent=1); return
        L=max(len(r[0]) for r in rows)
        B=len(rows)
        ids=torch.zeros(B,L,dtype=torch.long)
        for b,(x,_,_,_) in enumerate(rows):
            ids[b,:len(x)]=torch.tensor(x)
        ids=ids.to(DEV)
        TP=[r[1] for r in rows]; MT=[r[2] for r in rows]
        DS=[r[3] for r in rows]
        gg=torch.Generator().manual_seed(5)
        RD=[int(torch.randint(1,max(r[1],2),(1,),generator=gg))
            for r in rows]

        def fwd(arm):
            def fh(mo,args,o_):
                y,v1r=o_; X=args[0]; Bb=X.shape[0]; Tq=X.shape[1]
                v1b=args[1] if args[1] is not None else v1r
                z,vm=cl.head_parts(LJ,X,v1b); z=z.clone()
                if arm=='delete':
                    z[:,HD]=z[:,HD].mean(dim=(0,1),keepdim=True)
                elif arm!='real':
                    cq,sq=at.rotary(at.c_q(X).view(Bb,Tq,NH,128))
                    def r2(W):
                        return are(F.rms_norm(
                            W(X).view(Bb,Tq,NH,128),(128,)),
                            cq,sq)[:,:,HD].float()
                    s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),
                                    r2(at.c_k))/128
                    s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),
                                    r2(at.c_k2))/128
                    sc=(s1*s2)*torch.tril(
                        torch.ones(Tq,Tq,device=DEV))
                    key={'kill_match':MT,'kill_distractor':DS,
                         'kill_random':RD}[arm]
                    for b in range(Bb):
                        sc[b,TP[b],key[b]]=0.0
                    z[:,HD]=torch.einsum('bqk,bkd->bqd',sc,
                                         vm[:,:,HD].float())
                return (at.c_proj(z.transpose(1,2).contiguous()
                        .view(Bb,Tq,-1).to(X.dtype)),v1r)
            hs=[] if arm=='real' else [at.register_forward_hook(fh)]
            x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            for h in hs: h.remove()
            out=[]
            for b in range(B):
                lp=F.log_softmax(lg[b,TP[b]],dim=-1)
                out.append(-float(max(lp[CLOSE2],
                                      lp[CLOSE] if CLOSE else lp[CLOSE2])))
            return sum(out)/len(out)

        base=fwd('real')
        arms={a:fwd(a) for a in ('delete','kill_match',
                                 'kill_distractor','kill_random')}
        # score-mass shares
        cap={}
        hh=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hh.remove()
        X=cap['X']; Bb,Tq=X.shape[0],X.shape[1]
        cq,sq=at.rotary(at.c_q(X).view(Bb,Tq,NH,128))
        def r2(W):
            return are(F.rms_norm(W(X).view(Bb,Tq,NH,128),(128,)),
                       cq,sq)[:,:,HD].float()
        s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),r2(at.c_k))/128
        s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),r2(at.c_k2))/128
        p2=((s1*s2)*torch.tril(torch.ones(Tq,Tq,device=DEV))).cpu()
        den=p2.abs().sum(-1).clamp_min(1e-6)
        sm=lambda K:sum(abs(float(p2[b,TP[b],K[b]]/den[b,TP[b]]))
                        for b in range(B))/B
        res[kind]={'n':B,'base_nll_closer':round(base,4),
                   'delete':round(arms['delete']-base,4),
                   'kill_match':round(arms['kill_match']-base,4),
                   'kill_distractor':round(
                       arms['kill_distractor']-base,4),
                   'kill_random':round(arms['kill_random']-base,4),
                   'share_match':round(sm(MT),4),
                   'share_distractor':round(sm(DS),4),
                   'share_random':round(sm(RD),4)}
        r=res[kind]
        print(f"{kind}: n={B} base NLL of the closer "
              f"{r['base_nll_closer']:.3f}",flush=True)
        print(f"   delete {r['delete']:+.4f} | kill match "
              f"{r['kill_match']:+.4f} | kill distractor "
              f"{r['kill_distractor']:+.4f} | kill random "
              f"{r['kill_random']:+.4f}")
        print(f"   share  match {r['share_match']:.4f} | "
              f"distractor {r['share_distractor']:.4f} | random "
              f"{r['share_random']:.4f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    O=res['outer']; I=res['inner']
    p0=(O['base_nll_closer']<2.0 and I['base_nll_closer']<2.0)
    print(f"\n(0) the model expects a closer in both conditions "
          f"(outer {O['base_nll_closer']:.3f}, inner "
          f"{I['base_nll_closer']:.3f} < 2.0): "
          f"{'HELD' if p0 else 'FAILED -- RUN VOID'}")
    if not p0:
        json.dump({'per_condition':res,'pred_0':False},
                  open(OUT,'w'),indent=1); return
    va,_=cl.score_bar('a',O['share_match']-2*O['share_distractor'],
                      1e-9)
    vb,_=cl.score_bar('b',O['kill_match']-3*max(
        O['kill_distractor'],0),1e-9)
    vc,_=cl.score_bar('c',I['kill_match'],0.20)
    nul=(O['kill_random']<0.05 and I['kill_random']<0.05)
    print(f"NULL (random cell costs < 0.05 in both: outer "
          f"{O['kill_random']:+.4f}, inner {I['kill_random']:+.4f}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'per_condition':res,'pred_0':True,'pred_a':va=='HELD',
         'pred_b':vb=='HELD','pred_c':vc=='HELD',
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
