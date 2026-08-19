"""Track-1 SEMANTIC PILOT (the user's Bills-style track, with the session LLM as
explanation writer and mechanical causal scoring). Four components, each with an
explanation derived from the program's findings, compiled to a per-token
predictor of the ablation fingerprint:

  mlp16  : "carries content that keeps the model confident where it is right;
            deleting it hurts easy tokens most"        -> predictor: -base_loss
  attn14 : "net-harmful late attention; deleting it relieves confidently-wrong
            (hard) tokens"                             -> predictor: -base_loss
            ... but with NEGATIVE net: predict delta ~ -(hard-relief) => delta
            most negative on hard tokens               -> predictor: -base_loss
  attn1  : "early context assembly; damage grows with how much context the
            position depends on"                       -> predictor: +position
  mlp9   : "regularizer span; deletion helps hard tokens, hurts easy"
                                                       -> predictor: -base_loss
Scoring: Spearman(predictor, fingerprint). The measured confound floor is 0.13
(section 162) -- an explanation earns credit above it. THE BILLS CRITERION:
correct explanation-component matching must beat shuffled matching.

REGISTERED PREDICTIONS: (a) >= 2/4 correct-assignment scores >= 0.18 (floor
+0.05); (b) matching matters: the correct assignment's mean |score| exceeds the
mean over 10 random permutations of assignments by >= 0.05; (c) sign
discipline: for the three difficulty-based predictors the SIGN of the score
matches the explanation's claim (mlp16 positive-on-easy => negative Spearman
vs base loss for the delta... signs stated in code)."""
import json, torch, time, itertools, random
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bilin18_track1_pilot_results.json'

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    d=torch.load(PT+'bilin18_fingerprint_atlas.pt')
    base=d['base'].float()
    fps=d['fingerprints']
    n=len(base); T=256
    pos=(torch.arange(n)%T).float()
    # predictors compiled from the explanations
    preds={'mlp16': -base,      # easy tokens hurt most => delta anti-tracks loss
           'attn14': -base,     # hard tokens relieved (delta most negative there)
           'attn1': pos,        # damage grows with available context
           'mlp9': -base}       # helps hard, hurts easy => delta anti-tracks loss
    comps=['mlp16','attn14','attn1','mlp9']
    scores={c: spearman(preds[c], fps[c].float()) for c in comps}
    for c in comps: print(f'{c:7s}: score {scores[c]:+.3f}',flush=True)
    # expected signs per explanation:
    #  mlp16: delta positive on easy (low loss) => corr(delta, -loss) > 0
    #  attn14: delta negative on hard => corr(delta, -loss) > 0
    #  mlp9: same shape as attn14 => > 0
    #  attn1: > 0 with position
    sign_ok=sum(1 for c in comps if scores[c]>0)
    strong=sum(1 for c in comps if abs(scores[c])>=0.18)
    rng=random.Random(0)
    perm_means=[]
    for _ in range(10):
        p=comps[:]; rng.shuffle(p)
        perm_means.append(sum(abs(spearman(preds[a],fps[b].float()))
                              for a,b in zip(comps,p))/4)
    correct_mean=sum(abs(v) for v in scores.values())/4
    null_mean=sum(perm_means)/len(perm_means)
    pa=strong>=2
    pb=(correct_mean-null_mean)>=0.05
    pc=sign_ok>=3
    out={'scores':scores,'correct_mean':correct_mean,'shuffled_mean':null_mean,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\ncorrect-assignment mean |score| {correct_mean:.3f} vs shuffled '
          f'{null_mean:.3f}')
    print(f"(a) >=2/4 beat floor+0.05: {'HELD' if pa else 'FAILED'} ({strong}/4)")
    print(f"(b) matching matters (>= +0.05): {'HELD' if pb else 'FAILED'}")
    print(f"(c) signs match claims (>=3/4): {'HELD' if pc else 'FAILED'} ({sign_ok}/4)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
