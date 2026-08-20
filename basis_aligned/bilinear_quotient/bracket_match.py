"""BRACKET MATCH -- is the matching actually the mechanism?
520 localized the largest concentrated effect in this program to
ONE head. Deleting head 13.8 costs +0.825 nats at positions where
the next token is a closing bracket, +0.006 six tokens away,
+0.007 on random targets, and it drops the closing bracket's own
logit by 1.542 against 0.473 for the best competitor. Its score
mass at those positions sits 14.6x more heavily on the SPECIFIC
matching opener (0.381) than on other earlier openers (0.026).
That last number is a correlation. The head looks at the matching
opener; it has not been shown that looking there is what makes it
work. The intervention that settles it is to leave the head intact
and delete individual entries of its score matrix.
Arms, all at close-bracket target queries only, with the head
otherwise untouched:
  kill_match      zero the score on the matching opener
  kill_random     zero the score on a random earlier position
  kill_nearest    zero the score on the nearest NON-matching
                  earlier opener -- the hardest control, because
                  it is the same kind of token at a similar
                  distance, differing only in whether it is the
                  one this bracket closes
  kill_prev       zero the score on the previous token
  delete          the whole head mean-ablated (reference, 0.825)
Each arm zeroes exactly ONE (query, key) entry per target, so the
arms are matched in how much of the score matrix they remove; only
WHICH entry differs. Cost is priced at the close-bracket targets.
REGISTERED PREDICTIONS:
  (a) MATCHING IS THE MECHANISM: kill_match costs >= 0.30 nats at
      close-bracket targets, i.e. at least 36% of the full
      deletion, from removing a single matrix entry per position;
  (b) IT IS THE MATCH AND NOT PROXIMITY: kill_match costs at
      least 3x kill_nearest, the same-token-type control at a
      similar distance. This is the arm that separates a matcher
      from a head that attends to whatever opener is closest;
  (c) SANITY: kill_random and kill_prev each cost < 0.10, so the
      effect is not that removing any entry hurts.
  NULL: on the 7 of 84 targets that have NO matching opener in
      the window, kill_random must cost about what it costs
      elsewhere; and the arms are reported as absolute pairs, not
      quotients, because three degenerate-ratio nulls in one
      session is enough (520).
If (a) holds and (b) fails, the head attends to the nearest opener
rather than the matching one, which is a weaker but still real
mechanism and must be reported as such -- most brackets in natural
text are not nested, so nearest and matching coincide most of the
time, and the nested cases are where they separate. The run
therefore also reports both arms restricted to the subset of
targets whose match is NOT the nearest opener."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_match_results.json'
NFRESH=128
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
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool)
    match={}; nearest={}; opens_at={}
    g=torch.Generator().manual_seed(29)
    for r in range(NFRESH):
        stack=[]; opos=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s)); opos.append(q)
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                tgt[r,q]=True
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                match[(r,q)]=mt
                nm=[p for p in opos if p<=q and p!=mt]
                nearest[(r,q)]=nm[-1] if nm else None
                opens_at[(r,q)]=list(opos)
    N=int(tgt.sum())
    nested=[(r,q) for (r,q),mt in match.items()
            if mt is not None and nearest.get((r,q)) is not None
            and nearest[(r,q)]>mt]
    print(f'{N} targets | {sum(1 for v in match.values() if v is not None)} '
          f'matched | {len(nested)} where the match is NOT the '
          f'nearest opener',flush=True)
    at=m.transformer.h[LJ].attn
    ARMS=['delete','kill_match','kill_nearest','kill_random',
          'kill_prev']

    def run(arm):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            def fh(mo,args,o_):
                y,v1r=o_; X=args[0]
                v1b=args[1] if args[1] is not None else v1r
                z,vm=cl.head_parts(LJ,X,v1b)
                z=z.clone()
                if arm=='delete':
                    z[:,HD]=z[:,HD].mean(dim=(0,1),keepdim=True)
                else:
                    cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
                    def r2(W):
                        return are(F.rms_norm(
                            W(X).view(B,T,NH,128),(128,)),
                            cq,sq)[:,:,HD].float()
                    s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),
                                    r2(at.c_k))/128
                    s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),
                                    r2(at.c_k2))/128
                    sc=(s1*s2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    for b in range(B):
                        r=i+b
                        for q in tgt[r].nonzero().squeeze(1).tolist():
                            if arm=='kill_match':
                                k=match.get((r,q))
                            elif arm=='kill_nearest':
                                k=nearest.get((r,q))
                            elif arm=='kill_prev':
                                k=q-1 if q>0 else None
                            else:
                                k=(int(torch.randint(0,q+1,(1,),
                                   generator=g)) if q>0 else None)
                            if k is None: continue
                            sc[b,q,k]=0.0
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
        return ce

    base=run('none_intact') if False else None
    # intact baseline: no hook at all
    def plain():
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
        return ce
    base=plain()
    nestmask=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r,q in nested: nestmask[r,q]=True
    res={}
    for arm in ARMS:
        ce=run(arm); d=ce-base
        res[arm]={'target':round(float(d[tgt].mean()),4),
                  'nested':round(float(d[nestmask].mean()),4)
                            if len(nested) else None,
                  'global':round(float(d.mean()),5)}
        print(f"{arm:>13}: target {res[arm]['target']:+.4f} | "
              f"nested-only {res[arm]['nested']} | global "
              f"{res[arm]['global']:+.5f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    km=res['kill_match']['target']; kn=res['kill_nearest']['target']
    kr=res['kill_random']['target']; kp=res['kill_prev']['target']
    dl=res['delete']['target']
    va,_=cl.score_bar('a',km,0.30)
    vb,_=cl.score_bar('b',km-3*max(kn,0),1e-9)
    vc='HELD' if (kr<0.10 and kp<0.10) else 'FAILED'
    print(f'(c) kill_random {kr:+.4f} and kill_prev {kp:+.4f} both '
          f'< 0.10: {vc}')
    print(f'  one entry of the score matrix removed: match '
          f'{km:+.4f} of the head\'s full {dl:+.4f}')
    if len(nested):
        print(f"  nested subset (n={len(nested)}): match "
              f"{res['kill_match']['nested']} vs nearest "
              f"{res['kill_nearest']['nested']}")
    out={'arms':res,'n_targets':N,'n_nested':len(nested),
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
