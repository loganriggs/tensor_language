"""BRACKET STATE -- does head 13.8 track which brackets are still
open, or just which tokens are brackets?
520 showed the head puts 14.6x more score mass on the specific
matching opener than on other earlier openers. "Other earlier
openers" lumped together two very different things, and the
distinction is the whole question of whether this head holds
STATE:
  STILL-OPEN openers -- brackets that have not been closed yet at
    the query position. The matching one is a member of this set,
    normally its most recent member.
  ALREADY-CLOSED openers -- brackets whose closer has already been
    emitted. These are lexically identical "(" tokens sitting at
    similar distances, and a head that merely recognizes bracket
    tokens has no way to tell them apart.
If the head discounts already-closed openers, it is reading a
running state variable -- the open-bracket stack -- and not a
token property. That is the difference between a semantically
meaningful feature and pattern-matching on token identity.
Measured at close-bracket target queries as signed shares of the
head's score mass (score/sum|score| over the causal window, the
same statistic as 497 and 520), for five key groups:
  the matching opener
  other still-open openers
  already-closed openers
  a random earlier NON-bracket position (lexical baseline)
  the previous token
and, as the null, the same five groups measured at
position-matched control queries where no closing bracket is due.
REGISTERED PREDICTIONS:
  (a) STATE, NOT TOKENS: the absolute share on already-closed
      openers is at most half the share on still-open ones. This
      is the claim, and it can only be true if the head knows
      which brackets are outstanding;
  (b) LEXICAL BASELINE: already-closed openers get a share within
      2x of a random earlier non-bracket position, i.e. once a
      bracket is closed the head treats it roughly like ordinary
      text;
  (c) THE MATCH IS STILL SPECIAL: the matching opener beats even
      the other STILL-OPEN openers by >= 2x, so the head is not
      simply attending to the whole open stack.
  NULL: at position-matched control queries the ratio of
      still-open to already-closed share falls below 2.0. If the
      head separates them everywhere, the separation is a fixed
      property of its key side rather than something it computes
      for the closing decision -- still interesting, but a
      different claim, and it must be reported that way.
All bars on ABSOLUTE shares with the pair printed; no quotient is
scored whose denominator can approach zero (the rule from 520,
after three degenerate-ratio nulls in one session)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_state_results.json'
NFRESH=192
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
    info={}
    g=torch.Generator().manual_seed(29)
    for r in range(NFRESH):
        stack=[]; allopen=[]; closed=[]
        isbr=[False]*T
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS:
                stack.append((q,s)); allopen.append(q); isbr[q]=True
            elif s in CLOSES:
                isbr[q]=True
                if stack:
                    p,_=stack.pop(); closed.append(p)
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                stillopen=[p for p,_ in stack if p!=mt]
                nonbr=[p for p in range(q+1) if not isbr[p]]
                rnd=(int(nonbr[int(torch.randint(0,len(nonbr),(1,),
                     generator=g))]) if nonbr else None)
                if mt is not None or stillopen or closed:
                    tgt[r,q]=True
                    info[(r,q)]={'match':mt,
                                 'open':list(stillopen),
                                 'closed':[p for p in closed if p<=q],
                                 'rand':rnd}
    # position-matched controls, with the same group definitions
    ctrl=torch.zeros_like(tgt); cinfo={}
    for r in range(NFRESH):
        k=int(tgt[r].sum())
        if k==0: continue
        pos=tgt[r].nonzero().squeeze(1)
        j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(1,T-1)
        for a,q in zip(pos.tolist(),j.tolist()):
            if tgt[r,q]: continue
            d=info[(r,a)]
            cinfo[(r,q)]={'match':d['match'] if (d['match'] is not None
                          and d['match']<q) else None,
                          'open':[p for p in d['open'] if p<q],
                          'closed':[p for p in d['closed'] if p<q],
                          'rand':d['rand'] if (d['rand'] is not None
                                 and d['rand']<q) else None}
            ctrl[r,q]=True
    nmatch=sum(1 for d in info.values() if d['match'] is not None)
    nclosed=sum(1 for d in info.values() if d['closed'])
    print(f'{int(tgt.sum())} close-bracket targets over {NFRESH} '
          f'rows | {nmatch} with a matching opener | {nclosed} with '
          f'at least one already-closed opener | {int(ctrl.sum())} '
          f'controls',flush=True)
    if nclosed<20:
        print('*** too few contexts with a closed opener -- the '
              'comparison class is unpopulated, run VOID ***')
        json.dump({'void':'unpopulated closed-opener class',
                   'n_closed':nclosed},open(OUT,'w'),indent=1); return
    at=m.transformer.h[LJ].attn
    GROUPS=['match','open','closed','rand','prev']
    acc={'tgt':{k:[0.0,0] for k in GROUPS},
         'ctrl':{k:[0.0,0] for k in GROUPS}}
    cap={}
    hh=at.register_forward_pre_hook(
        lambda mo_,a_: cap.__setitem__('X',a_[0]))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        X=cap['X']
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        def r2(W):
            return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                       cq,sq)[:,:,HD].float()
        s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),r2(at.c_k))/128
        s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),r2(at.c_k2))/128
        p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
        den=p2.abs().sum(-1).clamp_min(1e-6)
        for b in range(B):
            r=i+b
            for nm,mask,src in (('tgt',tgt,info),('ctrl',ctrl,cinfo)):
                for q in mask[r].nonzero().squeeze(1).tolist():
                    d=src.get((r,q))
                    if d is None: continue
                    sh=lambda k: float(p2[b,q,k]/den[b,q])
                    if d['match'] is not None:
                        acc[nm]['match'][0]+=abs(sh(d['match']))
                        acc[nm]['match'][1]+=1
                    if d['open']:
                        acc[nm]['open'][0]+=sum(abs(sh(p))
                            for p in d['open'])/len(d['open'])
                        acc[nm]['open'][1]+=1
                    if d['closed']:
                        acc[nm]['closed'][0]+=sum(abs(sh(p))
                            for p in d['closed'])/len(d['closed'])
                        acc[nm]['closed'][1]+=1
                    if d['rand'] is not None:
                        acc[nm]['rand'][0]+=abs(sh(d['rand']))
                        acc[nm]['rand'][1]+=1
                    if q>0:
                        acc[nm]['prev'][0]+=abs(sh(q-1))
                        acc[nm]['prev'][1]+=1
    hh.remove()
    S={nm:{k:(round(v[k][0]/max(v[k][1],1),4),v[k][1])
           for k in GROUPS} for nm,v in acc.items()}
    print('\nabsolute score-mass share (mean, n):')
    for nm in ('tgt','ctrl'):
        print(f'  {nm}: '+'  '.join(
            f'{k} {S[nm][k][0]:.4f} (n={S[nm][k][1]})'
            for k in GROUPS))
    to=S['tgt']['open'][0]; tc=S['tgt']['closed'][0]
    tm=S['tgt']['match'][0]; tr=S['tgt']['rand'][0]
    co=S['ctrl']['open'][0]; cc=S['ctrl']['closed'][0]
    va,_=cl.score_bar('a',0.5*to-tc,1e-9)
    vb,_=cl.score_bar('b',2*tr-tc,1e-9)
    vc,_=cl.score_bar('c',tm-2*to,1e-9)
    nul=(co/max(cc,1e-9))<2.0 if cc>1e-9 else False
    print(f'  (a) closed {tc:.4f} <= half of still-open '
          f'{to:.4f}')
    print(f'  (b) closed {tc:.4f} within 2x of random non-bracket '
          f'{tr:.4f}')
    print(f'  (c) match {tm:.4f} >= 2x still-open {to:.4f}')
    print(f'  NULL at controls: still-open {co:.4f} vs closed '
          f'{cc:.4f}, ratio {co/max(cc,1e-9):.2f} < 2.0: '
          f"{'ok' if nul else 'VIOLATED'}")
    out={'shares':{nm:{k:S[nm][k][0] for k in GROUPS}
                   for nm in S},
         'counts':{nm:{k:S[nm][k][1] for k in GROUPS} for nm in S},
         'n_targets':int(tgt.sum()),'n_with_closed':nclosed,
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
