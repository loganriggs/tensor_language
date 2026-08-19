"""Track-1 confound floor: how much of each ablation fingerprint is predictable
from TRIVIAL features an explanation gets for free? Features: (i) the base
model's per-token loss (the flattening confound of sections 140-142 -- damage
correlates with difficulty generically); (ii) position in sequence. Any
explanation's causal score must beat this floor to count as understanding.

REGISTERED PREDICTIONS: (a) base loss alone predicts fingerprints at median
|Spearman| >= 0.2 (the confound is real and must be published as the floor);
(b) after rank-regressing base loss out, the residual fingerprints remain
mutually distinguishable (median pairwise |Spearman| <= 0.5); (c) position adds
almost nothing (median |Spearman| <= 0.1)."""
import json, torch, time
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'bilin18_fingerprints.pt')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_fingerprint_floor_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    d=torch.load(PT)
    base=d['base'].float(); fps={k:v.float() for k,v in d['fingerprints'].items()}
    n=len(base)
    T=256
    pos=torch.arange(n)%T
    s_base={k:spearman(base,v) for k,v in fps.items()}
    s_pos={k:spearman(pos.float(),v) for k,v in fps.items()}
    mb=sorted(abs(x) for x in s_base.values())[len(s_base)//2]
    mp=sorted(abs(x) for x in s_pos.values())[len(s_pos)//2]
    rb=base.argsort().argsort().float()
    rb=(rb-rb.mean())/rb.std()
    resid={}
    for k,v in fps.items():
        rv=v.argsort().argsort().float(); rv=(rv-rv.mean())/rv.std()
        resid[k]=rv-(rv*rb).mean()*rb
    keys=list(resid); pw=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            pw.append(abs(spearman(resid[keys[i]],resid[keys[j]])))
    mpw=sorted(pw)[len(pw)//2]
    pa=mb>=0.2; pb2=mpw<=0.5; pc=mp<=0.1
    out={'spearman_base':s_base,'spearman_pos':s_pos,
         'median_abs_base':mb,'median_abs_pos':mp,
         'median_pairwise_residual':mpw,
         'pred_a':bool(pa),'pred_b':bool(pb2),'pred_c':bool(pc)}
    print('base-loss floor per component:')
    for k in sorted(s_base): print(f'  {k:8s}: {s_base[k]:+.2f}')
    print(f'median |base| {mb:.2f} | median |pos| {mp:.2f} | '
          f'residual pairwise {mpw:.2f}')
    print(f"(a) confound floor real (>=0.2): {'HELD' if pa else 'FAILED'}")
    print(f"(b) residuals distinguishable (<=0.5): {'HELD' if pb2 else 'FAILED'}")
    print(f"(c) position negligible (<=0.1): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
