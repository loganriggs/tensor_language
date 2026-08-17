"""MLP correspondence matrix (completing section 165): bilin12 mlp1/5/8 span
fingerprints vs bilin18 mlp span fingerprints at layers 1,3,5,7,9,11,13,15,16.
REGISTERED: (a) best-fraction match within +-0.15 for >= 2/3 (MLP fingerprints
are weaker; looser bar); (b) bilin12 mlp8 (fraction 0.67) best-matches in
bilin18's L10-14 band.

Prior context -- the full correspondence matrix: bilin18
attention fingerprints at layers 1,3,5,7,9,11,13,15,17 vs bilin12's attn1/2/6
fingerprints. REGISTERED: (a) each bilin12 component's best-matching bilin18
depth fraction within +-0.15 of its own fraction (1/12, 2/12, 6/12); (b)
correspondence curve unimodal around the match for >= 2/3 components.

Prior context -- does cross-model correspondence follow RELATIVE depth?
Section 163 paired components at equal absolute indices (attn6~attn6). But
bilin12 has 12 layers to bilin18's 18: bilin12's L6 sits at depth fraction 0.5,
whose bilin18 analog is L9, not L6. Compute the missing bilin18 fingerprints
(attn9, mlp7 -- relative analogs of bilin12's attn6, mlp5) and compare pairings.

REGISTERED PREDICTIONS: (a) for the mid components, relative-depth pairing beats
absolute: |rho(b12.attn6, b18.attn9)| > |rho(b12.attn6, b18.attn6)| and
|rho(b12.mlp5, b18.mlp7)| > |rho(b12.mlp5, b18.mlp5)|; (b) at the front the
two coincide (attn1/attn2 pairs already aligned; their analogs are themselves);
(c) all analog pairings (either scheme) stay >= 3x the 0.05 non-analog floor."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import per_token, attn_mean, spearman
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_mlp_correspondence_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    d18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprints.pt')
    d12=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin12_fingerprints.pt')
    ce0=d18['base'].to(DEV)
    f18={k:v.float() for k,v in d18['fingerprints'].items()}
    f12={k:v.float() for k,v in d12['fingerprints'].items()}
    L18=(1,3,5,7,9,11,13,15,16)
    for li in L18:
        k=f'mlp{li}'
        if k not in f18:
            accs=[]
            for i in range(0,36,6):
                acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
                accs.append(acc[0])
            Y=torch.cat(accs); Ybar=Y.mean(0)
            _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
            Q=orth(Vh[:8].T)
            f18[k]=(per_token(mlp_span=(li,(Q,Ybar@Q)))-ce0).cpu().float()
            print(f'{k}: net {float(f18[k].mean()):+.4f}',flush=True)
    curves={}
    for b12 in ('mlp1','mlp5','mlp8'):
        curves[b12]=[abs(spearman(f12[b12],f18[f'mlp{li}'])) for li in L18]
        print(f'{b12}: '+' '.join(f'{v:.2f}' for v in curves[b12]),flush=True)
    fr12={'mlp1':1/12,'mlp5':5/12,'mlp8':8/12}
    hits=0
    out={'layers':list(L18),'curves':curves}
    for b12,cv in curves.items():
        pk=max(range(len(cv)),key=lambda i:cv[i])
        best=L18[pk]; bf=best/18
        if abs(bf-fr12[b12])<=0.15: hits+=1
        out[b12+'_best']=best
        print(f'{b12}: best L{best} (fraction {bf:.2f} vs own {fr12[b12]:.2f})',
              flush=True)
    pb=out['mlp8_best'] in (10,11,12,13,14)
    pa=hits>=2
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) fractions match >= 2/3: {'HELD' if pa else 'FAILED'} ({hits}/3)")
    print(f"(b) mlp8 in L10-14 band: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
