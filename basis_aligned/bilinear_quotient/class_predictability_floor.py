"""Floor measurement for the input-only dictionary (the section-242 rung):
how well can site CLASS (currently oracle: defined by the target token) be
predicted from CONTEXT ONLY? Deterministic input-only rules, no model:
  digit:   prev token is/ends with a digit, or prev is digit-punct (.,:/-)
           with a digit before it
  bclose:  an unclosed ( or [ exists in the last 60 tokens
  newline: prev token is/ends '\n'-adjacent punctuation or the line is
           long (pos since last newline > 40) -- crude, reported honestly
  sentend: prev token ends a plausible clause (length>3 word) -- weak,
           expect low precision, reported
  subword: prev token is a word FRAGMENT (no leading space, alphabetic) or
           an alpha token that commonly continues (no trailing space
           structure in BPE -- approximated: prev has no leading space OR
           next-token-in-vocab continuation; approximation stated)
  ind:     the prev token occurred earlier AND its successor then is a
           candidate (classic induction feature)
  rep/name/comma/other: analogous simple rules.
Report the confusion between input-predicted class and oracle class on
window C, per-class precision/recall, and the oracle-class base rates.

REGISTERED PREDICTIONS: (a) bclose and digit are input-predictable at
precision >= 0.5 (their context signatures are strong); (b) subword
input-precision >= 0.4 (mid-word fragments are visible); (c) overall
input-predicted class agrees with oracle class on >= 40% of sites
(weighted); whatever fails bounds the deployable dictionary and is the
honest 'deciding is the hard part' measurement."""
import json, sys, time, torch, collections
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
from circuit_dictionary import classify, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'class_predictability_floor_results.json'
R0,R1=120,300

def input_class(toks,pos):
    pv=enc.decode([toks[pos]]); pvs=pv.strip()
    ctx=toks[max(0,pos-60):pos+1]
    dctx=enc.decode(ctx)
    if pvs and (pvs[-1].isdigit() or (pvs[-1] in '.,:/-' and len(pvs)>1
                and pvs[-2].isdigit())): return 0
    if (dctx.count('(')>dctx.count(')')) or (dctx.count('[')>dctx.count(']')):
        return 1
    # newline: after sentence-final punct + quote, crude
    if pvs in ('.','!','?') : return 3   # sentend-adjacent: predict sentend? actually after '.' comes space-word usually; skip
    if pv==',': return 4
    if pvs and pvs[-1] in ('.','!','?','"',')') and pos>200: return 2
    if pv.strip() and pv[:1]==' ' and pvs[:1].isupper(): return 5
    if (not pv.startswith(' ')) and pvs.isalpha() and len(pvs)<=4: return 7
    if toks[pos] in toks[:pos]: return 8
    return 9

def main():
    t0=time.time()
    oracle=classify(R0,R1).reshape(-1)
    pred=torch.zeros_like(oracle)
    i=0
    for r in range(R0,R1):
        toks=FW[r,:257].tolist()
        for pos in range(256):
            pred[i]=input_class(toks,pos); i+=1
    agree=float((pred==oracle).float().mean())
    per={}
    for k,name in enumerate(CLS):
        o=oracle==k; p=pred==k
        if int(o.sum())<50: continue
        prec=float((o&p).sum())/max(int(p.sum()),1)
        rec=float((o&p).sum())/int(o.sum())
        per[name]={'base':round(float(o.float().mean()),3),
                   'precision':round(prec,2),'recall':round(rec,2)}
        print(f'{name:8s} base {per[name]["base"]:.3f} prec {prec:.2f} '
              f'rec {rec:.2f}',flush=True)
    pa=per.get('bclose',{}).get('precision',0)>=0.5 and \
       per.get('digit',{}).get('precision',0)>=0.5
    pb=per.get('subword',{}).get('precision',0)>=0.4
    pc=agree>=0.40
    out={'agreement':round(agree,3),'per_class':per,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\noverall agreement {agree:.0%}")
    print(f"(a) bclose+digit prec >=0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) subword prec >=0.4: {'HELD' if pb else 'FAILED'}")
    print(f"(c) agreement >=40%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
