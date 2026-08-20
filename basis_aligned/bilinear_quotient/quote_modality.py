"""QUOTE MODALITY -- the third-head test, done right.
564 confirmed head 10.7 attends to the most recent quote (5.6x
over control) and diagnosed why the earlier AND test was
ambiguous: a poor distractor and a sign-blind absolute-value
factor metric. This redoes it with the sign-aware normalized
score-mass share (the metric that worked in destination), a
MATCHED distractor (the second-most-recent quote, so both keys are
quotes and only position differs), and the rotary probe.
The general account (562) predicts: opening-quote prediction is a
DETECTION task, so head 10.7 should attend to quote tokens
indifferently to position -- given two quotes, no preference, and
rotary removal should not change that. That mirrors the newline
head and contrasts with the bracket head's positional matching.
Measured at opening-quote targets: signed score-mass share on the
most-recent quote (match) and the second-most-recent quote
(distractor), with real rotary and with rotary disabled.
REGISTERED PREDICTIONS:
  (0) POPULATED: >= 30 targets with two prior quotes;
  (a) DETECTION, NOT MATCHING: the match/distractor share ratio is
      below 1.5 -- the head does not strongly prefer one quote
      over the other by position. (Contrast: brackets 5.99.);
  (b) ROTARY-INSENSITIVE: disabling rotary changes the
      match/distractor ratio by < 30% (position is not the
      discriminator);
  (c) report both ratios and the raw shares. No bar;
  NULL: the head's total quote-share (match+distractor) must
      exceed its share on a non-quote key at the same positions,
      confirming token detection is active even if position is
      not. Report the quote-vs-nonquote contrast."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=10; HD=7; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'quote_modality_results.json'
NFRESH=256

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
    isq=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            if isquote(cur[r,q]): isq[r,q]=True
    cells={}
    for r in range(NFRESH):
        keys=isq[r].nonzero().squeeze(1).tolist()
        for q in range(T):
            if isquote(nxt[r,q]):
                prev=[k for k in keys if k<q]
                if len(prev)>=2:
                    # match=recent quote, distractor=2nd recent,
                    # nonq=a nearby non-quote for the null
                    nq=None
                    for k in range(q,-1,-1):
                        if not isq[r,k] and k!=q: nq=k; break
                    cells.setdefault(r,[]).append(
                        (q,prev[-1],prev[-2],nq))
    n=sum(len(v) for v in cells.values())
    print(f'{n} opening-quote targets with two prior quotes',
          flush=True)
    if n<30:
        print('*** too few -- VOID ***')
        json.dump({'void':'too few','n':n},open(OUT,'w'),indent=1)
        return
    at=m.transformer.h[LJ].attn
    acc={'real':{'m':0.0,'d':0.0,'nq':0.0,'n':0},
         'norot':{'m':0.0,'d':0.0,'n':0}}
    cap={}
    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
        if not rows: continue
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hc=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hc.remove()
        X=cap['X']
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        def rot(W): return are(F.rms_norm(W(X).view(B,T,NH,128),
                        (128,)),cq,sq)[:,:,HD].float()
        def raw(W): return F.rms_norm(W(X).view(B,T,NH,128),
                        (128,))[:,:,HD].float()
        for tag,qf,kf,q2,k2 in (
            ('real',rot(at.c_q),rot(at.c_k),rot(at.c_q2),rot(at.c_k2)),
            ('norot',raw(at.c_q),raw(at.c_k),raw(at.c_q2),raw(at.c_k2))):
            s1=torch.einsum('bqd,bkd->bqk',qf,kf)/128
            s2=torch.einsum('bqd,bkd->bqk',q2,k2)/128
            p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
            den=p2.abs().sum(-1).clamp_min(1e-6)
            for r in rows:
                b=r-i
                for (q,mt,ds,nq) in cells[r]:
                    acc[tag]['m']+=abs(float(p2[b,q,mt]/den[b,q]))
                    acc[tag]['d']+=abs(float(p2[b,q,ds]/den[b,q]))
                    acc[tag]['n']+=1
                    if tag=='real' and nq is not None:
                        acc['real']['nq']+=abs(float(
                            p2[b,q,nq]/den[b,q]))
    R=acc['real']; N=acc['norot']
    mr=R['m']/max(R['n'],1); dr=R['d']/max(R['n'],1)
    nqr=R['nq']/max(R['n'],1)
    mn=N['m']/max(N['n'],1); dn=N['d']/max(N['n'],1)
    ratio_real=mr/max(dr,1e-9); ratio_nr=mn/max(dn,1e-9)
    print(f'\nreal: match {mr:.4f} distractor {dr:.4f} '
          f'ratio {ratio_real:.2f} | non-quote {nqr:.4f}',flush=True)
    print(f'norot: match {mn:.4f} distractor {dn:.4f} '
          f'ratio {ratio_nr:.2f}',flush=True)
    pa=ratio_real<1.5
    pb=abs(ratio_real-ratio_nr)/max(ratio_real,1e-9)<0.30
    nul=(mr+dr)>2*nqr
    print(f"\n(a) match/distractor ratio {ratio_real:.2f} < 1.5 "
          f"(detection not matching): {'HELD' if pa else 'FAILED'}")
    print(f"(b) rotary-insensitive ({ratio_real:.2f} -> "
          f"{ratio_nr:.2f}): {'HELD' if pb else 'FAILED'}")
    print(f"NULL (quote-share {mr+dr:.4f} > 2x non-quote "
          f"{nqr:.4f}): {'ok' if nul else 'VIOLATED'}")
    verdict=('token-detection' if (pa and pb) else
             'positional' if not pa else 'mixed')
    print(f"MODALITY: {verdict} (brackets=positional, "
          f"newlines=token-detection)")
    out={'n':n,'match_share':round(mr,4),'distractor_share':round(dr,4),
         'nonquote_share':round(nqr,4),
         'ratio_real':round(ratio_real,3),'ratio_norot':round(ratio_nr,3),
         'modality':verdict,'pred_a':bool(pa),'pred_b':bool(pb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
