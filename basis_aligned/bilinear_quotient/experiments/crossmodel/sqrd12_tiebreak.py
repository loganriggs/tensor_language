"""Tie-break for section 167: sqrd12's attn6 best-matched bilin18's attn6
(absolute) over attn9 (fractional). Is that a real preference or a flat curve?
Compute bilin18 attention fingerprints at L5,7,9,11 (attn6 cached) and the full
local curve for sqrd12-attn6 and (contrast) bilin12-attn6.

REGISTERED: (a) decisiveness -- the |L6 - L9| gap for sqrd12-attn6 exceeds 0.03
one way or the other (a tie within 0.03 = unresolved, reported as such);
(b) contrast check: bilin12-attn6's curve prefers L9 over L6 by > 0.01 (its
section-165 result reproduces on these fingerprints)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW, DEV
from bilin18_fingerprints import per_token, attn_mean, spearman
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'sqrd12_tiebreak_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    d18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprints.pt')
    ce0=d18['base'].to(DEV)
    f18={k:v.float() for k,v in d18['fingerprints'].items()}
    f12=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin12_fingerprints.pt')['fingerprints']
    fs=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                  'sqrd12_fingerprints.pt')['fingerprints']
    for li in (5,7,9,11):
        k=f'attn{li}'
        if k not in f18:
            mu=attn_mean(li)
            f18[k]=(per_token(attn_layer=(li,mu))-ce0).cpu().float()
            print(f'{k}: net {float(f18[k].mean()):+.4f}',flush=True)
    Ls=(5,6,7,9,11)
    for name,fp in (('sqrd12',fs['attn6'].float()),
                    ('bilin12',f12['attn6'].float())):
        cv={li:abs(spearman(fp,f18[f'attn{li}'])) for li in Ls}
        print(f'{name}-attn6: '+' '.join(f'L{li}:{cv[li]:.3f}' for li in Ls),
              flush=True)
        if name=='sqrd12': cvs=cv
        else: cvb=cv
    gap=cvs[6]-cvs[9]
    pa=abs(gap)>0.03
    pb=(cvb[9]-cvb[6])>0.01
    out={'sqrd12_curve':{str(k):v for k,v in cvs.items()},
         'bilin12_curve':{str(k):v for k,v in cvb.items()},
         'gap_6_minus_9':gap,'pred_a_decisive':bool(pa),'pred_b':bool(pb)}
    verdict=('absolute' if gap>0.03 else
             'fractional' if gap<-0.03 else 'tie/unresolved')
    print(f'\nsqrd12 L6-L9 gap {gap:+.3f} -> {verdict}')
    print(f"(a) decisive: {'HELD' if pa else 'FAILED (tie)'}")
    print(f"(b) bilin12 contrast prefers L9: {'HELD' if pb else 'FAILED'}")
    out['verdict']=verdict
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
