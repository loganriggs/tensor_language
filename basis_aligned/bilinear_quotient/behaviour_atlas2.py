"""BEHAVIOUR ATLAS 2 -- the same screen on an unbiased denominator.
The first run (513) answered the generalization question: six
behaviour classes beyond newlines have a concentrated component,
led by four distinct components, with the newline/a12 positive
control recovered at 14.44. But its NULL failed for two classes,
and the cause was the denominator. "Elsewhere" was the complement
of the target, position-control and random masks, so for a class
with many targets or large effects the remainder is a biased,
unusually easy set of positions, and any position set scores above
it -- including a random one. Capitalized covers 1693 of ~8000
positions; its remainder is mostly easy lowercase continuations.
Same screen, same classes, same controls, one change: the
denominator is the GLOBAL mean damage over all positions, which
cannot move when a class is large. Every ratio in 513 is
superseded by this run; the absolute target/elsewhere pairs and
the position-matched comparisons were never affected.
Original framing follows.
--- does the method that found the newline head generalize? ---
510 retired the damage-cluster screen: across sixty leaves, on two
different decompositions, it produced no writer-level mechanism
that survived a causal test. The method that DID work three times
(induction, the position-0 bias, the newline head) starts from a
behaviour and works outward, and 495 reduced its first step to a
single cheap statistic: rank components not by how expensive they
are to delete, but by how CONCENTRATED their damage is on the
target behaviour relative to their own damage elsewhere. That
found attention layer 12 for newlines in 97 seconds, after the
magnitude ranking had spent the whole program pointing at the
front of the model.
The obvious question is whether newlines were special. This runs
the same screen over ten behaviour classes at once. Ablating each
of the 36 components once gives per-position costs, and every
class is a different readout of the SAME forward passes, so ten
behaviours cost what one did.
Classes, defined on the token being predicted: newline, digit,
sentence-final punctuation, comma, colon, opening quote, closing
quote, open bracket, close bracket, capitalized word.
Three masks per class: the targets; a POSITION-MATCHED control
(the same positions jittered by up to six tokens, which catches a
component that merely matters at certain places in a sequence);
and a FULLY RANDOM set of the same size, which catches a screen
that fires on any arbitrary position set at all.
REGISTERED PREDICTIONS:
  (a) POSITIVE CONTROL: newline recovers a12 as its top component
      with a concentration ratio >= 5.0 (495 measured 10.64 on a
      different sample). If the control fails, the screen is not
      reproducing a known result and the whole run is VOID --
      checked and reported before anything else is scored;
  (b) IT GENERALIZES: at least three of the other nine classes
      have a top component with ratio >= 2.0 that also beats its
      own position-matched control by >= 50%;
  (c) NOT ONE COMPONENT FOR EVERYTHING: across the classes that
      qualify under (b), at least three DISTINCT components lead.
      If one component leads every behaviour, the screen has found
      a general-purpose component and not behaviour-specific
      structure, which would be a negative result about the method
      and must be reported as one.
  NULL: for every class, the top component measured against the
      FULLY RANDOM target set must have ratio < 2.0. A screen that
      fires on random position sets is measuring nothing. THIS IS
      THE BAR THE FIRST RUN FAILED for newline (2.41) and
      capitalized (9.11); the denominator is now the global mean
      damage over all positions rather than the complement of the
      masks, which is what caused it.
Reporting rules from 497/500/501: absolute pairs alongside every
ratio, and every bar scored through cl.score_bar so that a
near-zero denominator returns UNEVALUABLE rather than a verdict."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'behaviour_atlas2_results.json'
NFRESH=48

def build_classes(rows):
    nxt=rows[:,1:257]
    R,Tn=nxt.shape
    names=['newline','digit','sentence_end','comma','colon',
           'open_quote','close_quote','open_bracket',
           'close_bracket','capitalized']
    M={n:torch.zeros(R,Tn,dtype=torch.bool) for n in names}
    for r in range(R):
        for q in range(Tn):
            s=cl.d1(int(nxt[r,q])); t=s.strip()
            if '\n' in s: M['newline'][r,q]=True
            if not t: continue
            if t[0].isdigit(): M['digit'][r,q]=True
            if t in ('.','!','?'): M['sentence_end'][r,q]=True
            if t==',': M['comma'][r,q]=True
            if t==':' or t==';': M['colon'][r,q]=True
            if t in ('"',"'",'``',"'"):
                (M['open_quote'] if s.startswith(' ') or s==t
                 else M['close_quote'])[r,q]=True
            if t in ('(','[','{'): M['open_bracket'][r,q]=True
            if t in (')',']','}'): M['close_bracket'][r,q]=True
            if t[0].isupper(): M['capitalized'][r,q]=True
    return M

def matched_controls(M,seed=29):
    g=torch.Generator().manual_seed(seed)
    ctrl={}; rnd={}
    for nm,mask in M.items():
        c=torch.zeros_like(mask); rr=torch.zeros_like(mask)
        R,Tn=mask.shape
        for r in range(R):
            k=int(mask[r].sum())
            if k==0: continue
            pos=mask[r].nonzero().squeeze(1)
            j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(0,Tn-1)
            c[r,j]=True
            rr[r,torch.randint(0,Tn,(k,),generator=g)]=True
        ctrl[nm]=c; rnd[nm]=rr
    return ctrl,rnd

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})
    fresh=cl.fineweb_rows(NFRESH)
    M=build_classes(fresh)
    CT,RN=matched_controls(M)
    for nm in M: print(f'{nm}: {int(M[nm].sum())} targets',flush=True)

    def hooks(key):
        mu=mus[key].to(DEV); mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]

    def run(key):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=hooks(key) if key else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return ce

    base=run(None)
    per={}
    for key in list(MODS):
        d=run(key)-base
        row={}
        for nm,mask in M.items():
            # 2026-08-20 (writeup 513): "elsewhere" was the
            # complement of the three masks, which biases the
            # denominator downward for large or high-damage
            # classes and inflated two nulls. The global mean over
            # ALL positions cannot move when a class is large.
            do=float(d.mean())
            row[nm]={
              'target':round(float(d[mask].mean()),5)
                       if int(mask.sum()) else None,
              'ctrl':round(float(d[CT[nm]].mean()),5)
                     if int(CT[nm].sum()) else None,
              'rand':round(float(d[RN[nm]].mean()),5)
                     if int(RN[nm].sum()) else None,
              'other':round(do,5)}
        per[key]=row
        print(f'{key} done',flush=True)
        json.dump(per,open(OUT,'w'),indent=1)

    summary={}
    for nm in M:
        cands=[]
        for key,row in per.items():
            r=row[nm]
            if r['target'] is None or r['other'] is None: continue
            if abs(r['other'])<1e-4: continue
            cands.append((r['target']/r['other'],key,r))
        cands.sort(key=lambda x:-x[0])
        if not cands: summary[nm]=None; continue
        ratio,key,r=cands[0]
        cr=(r['ctrl']/r['other']) if r['other'] else float('nan')
        rr=(r['rand']/r['other']) if r['other'] else float('nan')
        summary[nm]={'top':key,'ratio':round(ratio,2),
                     'ctrl_ratio':round(cr,2),'rand_ratio':round(rr,2),
                     'target':r['target'],'other':r['other'],
                     'n':int(M[nm].sum()),
                     'qualifies':bool(ratio>=2.0 and
                                      ratio>=1.5*max(cr,1e-6))}
        s=summary[nm]
        print(f"{nm:>14}: {key:>4} ratio {ratio:6.2f} "
              f"(target {r['target']:+.5f} vs other {r['other']:+.5f})"
              f" | pos-ctrl {cr:6.2f} | random {rr:6.2f} | "
              f"{'QUALIFIES' if s['qualifies'] else '-'}",flush=True)
    nl=summary.get('newline')
    pa=bool(nl and nl['top']=='a12' and nl['ratio']>=5.0)
    print(f"\n(a) POSITIVE CONTROL newline -> a12 with ratio >=5: "
          f"{'HELD' if pa else 'FAILED -- RUN VOID'}")
    qual=[nm for nm,s in summary.items()
          if nm!='newline' and s and s['qualifies']]
    pb=len(qual)>=3
    leaders={summary[nm]['top'] for nm in qual}
    pc=len(leaders)>=3
    nullbad=[nm for nm,s in summary.items()
             if s and s['rand_ratio']>=2.0]
    print(f"(b) >=3 other classes qualify: {qual} -> "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) >=3 distinct leaders among them: {sorted(leaders)} -> "
          f"{'HELD' if pc else 'FAILED'}")
    print(f"NULL (no class fires on fully random targets): "
          f"{'ok' if not nullbad else 'VIOLATED for '+str(nullbad)}")
    out={'summary':summary,'per_component':per,
         'qualifying':qual,'leaders':sorted(leaders),
         'pred_a':pa,'pred_b':bool(pb),'pred_c':bool(pc),
         'null_violations':nullbad,'n_rows':NFRESH,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
