"""QUOTE DESTINATION -- where does head 10.7 actually attend at
opening-quote targets? (identify the target before the AND test)
563 tested the quote head's discrimination against a match target
(most recent preceding quote) it may not use, and the factors
disagreed. This does the step that should have come first: at
opening-quote-predicting positions, measure where head 10.7 puts
its score mass, over candidate key classes, so any later
mechanism test uses the real target.
Signed score-mass share (score / sum|score| over the causal
window, the statistic of 497/523) at opening-quote targets, for
key groups:
  recent_quote   the most recent preceding quote token
  line_start     the token after the most recent newline
  sent_start     the token after the most recent sentence-ender
  prev           the previous token
  pos0           position 0
  self           the query position
  other          everything else
Reported at opening-quote targets and, as the null, at
position-matched control positions.
REGISTERED PREDICTIONS:
  (0) POPULATED: at least 30 opening-quote targets;
  (a) A DOMINANT DESTINATION: some named class (not 'other') gets
      >= 2x the score-mass share of the previous token. This
      identifies where the head looks;
  (b) SPECIFIC: that class's share at opening-quote targets is
      >= 1.5x its share at position-matched controls;
  (c) report the full profile. No bar -- the profile is the
      result and sets the next experiment;
  NULL: if 'other' dominates all named classes, the head has no
      compact attention target and is a diffuse detector -- report
      that plainly rather than forcing a target."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=10; HD=7; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'quote_destination_results.json'
NFRESH=192

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    def isquote(t):
        z=cl.d1(int(t)).strip(); return z in ('"',"'",'``',"''",'`')
    def issent(t):
        return cl.d1(int(t)).strip() in ('.','!','?')
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool)
    info={}
    for r in range(NFRESH):
        quotes=[]; nlines=[]; sents=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q]))
            if isquote(cur[r,q]): quotes.append(q)
            if chr(10) in s: nlines.append(q)
            if issent(cur[r,q]): sents.append(q)
            if isquote(nxt[r,q]):
                tgt[r,q]=True
                info[(r,q)]={
                    'recent_quote':quotes[-1] if quotes else None,
                    'line_start':(nlines[-1]+1 if nlines
                                  and nlines[-1]+1<=q else None),
                    'sent_start':(sents[-1]+1 if sents
                                  and sents[-1]+1<=q else None)}
    g=torch.Generator().manual_seed(29)
    ctrl=torch.zeros_like(tgt); cinfo={}
    for r in range(NFRESH):
        k=int(tgt[r].sum())
        if k==0: continue
        pos=tgt[r].nonzero().squeeze(1)
        j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(1,T-1)
        for a,q in zip(pos.tolist(),j.tolist()):
            if tgt[r,q]: continue
            d=info[(r,a)]
            cinfo[(r,q)]={kk:(vv if vv is not None and vv<q else None)
                          for kk,vv in d.items()}
            ctrl[r,q]=True
    n=int(tgt.sum())
    print(f'{n} opening-quote targets',flush=True)
    if n<30:
        print('*** too few targets -- VOID ***')
        json.dump({'void':'too few','n':n},open(OUT,'w'),indent=1)
        return
    at=m.transformer.h[LJ].attn
    GROUPS=['recent_quote','line_start','sent_start','prev','pos0',
            'self','other']
    acc={'t':{k:[0.0,0] for k in GROUPS},
         'c':{k:[0.0,0] for k in GROUPS}}
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
        for nm,mask,src in (('t',tgt,info),('c',ctrl,cinfo)):
            for r in range(i,min(i+4,NFRESH)):
                if not mask[r].any(): continue
                b=r-i
                for q in mask[r].nonzero().squeeze(1).tolist():
                    d=src.get((r,q))
                    if d is None: continue
                    sh=lambda k: float(p2[b,q,k]/den[b,q])
                    used=set()
                    for gk in ('recent_quote','line_start',
                               'sent_start'):
                        kk=d.get(gk)
                        if kk is not None:
                            acc[nm][gk][0]+=sh(kk); acc[nm][gk][1]+=1
                            used.add(kk)
                    if q>0:
                        acc[nm]['prev'][0]+=sh(q-1); acc[nm]['prev'][1]+=1
                    acc[nm]['pos0'][0]+=sh(0); acc[nm]['pos0'][1]+=1
                    acc[nm]['self'][0]+=sh(q); acc[nm]['self'][1]+=1
                    oth=[k for k in range(q+1)
                         if k not in used and k not in (q-1,0,q)]
                    if oth:
                        acc[nm]['other'][0]+=sum(sh(k) for k in oth) \
                            /len(oth)
                        acc[nm]['other'][1]+=1
    hh.remove()
    S={nm:{k:round(v[k][0]/max(v[k][1],1),4) for k in GROUPS}
       for nm,v in acc.items()}
    print('\nscore-mass share at opening-quote targets:',flush=True)
    for k in GROUPS:
        print(f'  {k:>13}: {S["t"][k]:+.4f} (control {S["c"][k]:+.4f})',
              flush=True)
    named={k:S['t'][k] for k in GROUPS if k!='other'}
    top=max(named,key=lambda k:abs(named[k]))
    pa=abs(S['t'][top])>=2*abs(S['t']['prev'])
    pb=abs(S['t'][top])>=1.5*abs(S['c'][top])
    diffuse=all(abs(S['t'][k])<=abs(S['t']['other'])
                for k in named)
    print(f"\n(a) dominant destination {top} "
          f"({S['t'][top]:+.4f}) >= 2x prev: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) {top} specific vs control: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"NULL (other dominates -> diffuse detector): "
          f"{'DIFFUSE' if diffuse else 'has a target'}")
    out={'n_targets':n,'shares_target':S['t'],
         'shares_control':S['c'],'top_destination':top,
         'diffuse':bool(diffuse),
         'pred_a':bool(pa),'pred_b':bool(pb),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
