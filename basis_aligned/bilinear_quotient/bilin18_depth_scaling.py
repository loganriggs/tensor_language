"""Does cross-model correspondence follow RELATIVE depth or ABSOLUTE index?
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
     'bilin18_depth_scaling_results.json')

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
    # missing bilin18 fingerprints: attn9, mlp7
    mu9=attn_mean(9)
    f18['attn9']=(per_token(attn_layer=(9,mu9))-ce0).cpu().float()
    print(f'attn9: net {float(f18["attn9"].mean()):+.4f}',flush=True)
    accs=[]
    for i in range(0,36,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=7, acc=acc); accs.append(acc[0])
    Y=torch.cat(accs); Ybar=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
    Q=orth(Vh[:8].T)
    f18['mlp7']=(per_token(mlp_span=(7,(Q,Ybar@Q)))-ce0).cpu().float()
    print(f'mlp7: net {float(f18["mlp7"].mean()):+.4f}',flush=True)
    def r(a,b): return abs(spearman(f12[a],f18[b]))
    pairs={'attn6_abs':r('attn6','attn6'),'attn6_rel':r('attn6','attn9'),
           'mlp5_abs':r('mlp5','mlp5'),'mlp5_rel':r('mlp5','mlp7'),
           'attn1':r('attn1','attn1'),'attn2':r('attn2','attn2')}
    for k,v in pairs.items(): print(f'{k:10s}: {v:.3f}',flush=True)
    pa=(pairs['attn6_rel']>pairs['attn6_abs']) and \
       (pairs['mlp5_rel']>pairs['mlp5_abs'])
    pb=pairs['attn1']>=0.15 and pairs['attn2']>=0.15
    pc=all(v>=0.15 for k,v in pairs.items() if k in
           ('attn6_rel','mlp5_rel','attn1','attn2')) or \
       all(v>=0.15 for k,v in pairs.items() if k in
           ('attn6_abs','mlp5_abs','attn1','attn2'))
    out={'pairs':pairs,'pred_a_relative':bool(pa),'pred_b_front':bool(pb),
         'pred_c_above_floor':bool(pc)}
    print(f"\n(a) relative-depth beats absolute at mid: {'HELD' if pa else 'FAILED'}")
    print(f"(b) front aligned: {'HELD' if pb else 'FAILED'}")
    print(f"(c) analog scheme >= 3x floor: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
