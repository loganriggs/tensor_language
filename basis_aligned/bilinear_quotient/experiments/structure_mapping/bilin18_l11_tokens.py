"""Which tokens carry the dissident's damage? Section 89: the top decile of tokens
carries 51% of L11's ablation damage, with an 8.3x entropy side-effect. Classify
the damage profile by corpus token frequency.

REGISTERED PREDICTIONS: (a) rare-token specialist -- Spearman(per-target-token mean
CE damage, log corpus frequency) <= -0.2 (the entropy signature suggests L11
sharpens predictions of rare continuations); (b) the damage profile is stable
across two disjoint held-out halves (split-half Spearman >= 0.5 -- profile is
signal, not noise). Null: the same profile computed under the energy-matched
random shift should correlate with the real profile at < 0.3 (damage reflects L11
content, not generic perturbation)."""
import json, sys, time, torch, math
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_l11_function import run
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l11_tokens_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9); rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    rows=FW[300:380,:257]
    tg=torch.cat([rows[i:i+4,1:].reshape(-1) for i in range(0,len(rows),4)])
    acc=[]; fwd(FW[300:336,:513].to(DEV), collect=11, acc=acc)
    Y=acc[0]; mu=Y.mean(0); en=float((Y-mu).pow(2).sum(1).mean())
    ce0,_=run(rows)
    I=torch.eye(D,device=DEV)
    ce1,_=run(rows, ablate=(I, mu@I))
    g=torch.Generator(device=DEV).manual_seed(0)
    rvec=torch.randn(D,device=DEV,generator=g); rvec=rvec/rvec.norm()*en**0.5
    ce2,_=run(rows, ablate=(None, rvec))
    d=(ce1-ce0).cpu(); dr=(ce2-ce0).cpu(); tg=tg.cpu()
    freq=torch.bincount(FW[:452,:257].reshape(-1), minlength=int(tg.max())+1).float()
    uniq=torch.unique(tg)
    prof=[]; prof_r=[]; lf=[]; keep=[]
    half=len(d)//2
    profA=[]; profB=[]
    for t in uniq:
        mask=(tg==t)
        if int(mask.sum())<5: continue
        keep.append(int(t))
        prof.append(float(d[mask].mean())); prof_r.append(float(dr[mask].mean()))
        lf.append(math.log(float(freq[t])+1))
        mA=mask.clone(); mA[half:]=False
        mB=mask.clone(); mB[:half]=False
        profA.append(float(d[mA].mean()) if mA.any() else 0.0)
        profB.append(float(d[mB].mean()) if mB.any() else 0.0)
    prof=torch.tensor(prof); prof_r=torch.tensor(prof_r); lf=torch.tensor(lf)
    both=[i for i in range(len(keep)) if profA[i]!=0.0 and profB[i]!=0.0]
    sAB=spearman(torch.tensor([profA[i] for i in both]),
                 torch.tensor([profB[i] for i in both]))
    sF=spearman(prof, lf)
    sN=spearman(prof, prof_r)
    pa=sF<=-0.2; pb=sAB>=0.5; pc=sN<0.3
    out={'n_token_types':len(keep),'spearman_damage_vs_logfreq':sF,
         'split_half':sAB,'null_corr_with_random_shift':sN,
         'pred_a_rare':bool(pa),'pred_b_stable':bool(pb),'ctrl_held':bool(pc)}
    print(f'{len(keep)} token types (>=5 occurrences)')
    print(f'Spearman(damage, log freq): {sF:+.2f}')
    print(f'split-half stability: {sAB:+.2f} | corr with random-shift profile: {sN:+.2f}')
    print(f"(a) rare-token specialist: {'HELD' if pa else 'FAILED'}")
    print(f"(b) profile stable: {'HELD' if pb else 'FAILED'}")
    print(f"random-shift control (<0.3): {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
