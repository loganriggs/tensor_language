"""Fingerprint atlas completion: extend bilin18_fingerprints.pt to full coverage
-- MLP top-8 spans and full attention for every layer 0-17 (36 components).
Benchmark asset, plus a new registered question: is token-level causal
responsibility DEPTH-SMOOTH?

REGISTERED PREDICTIONS: (a) distinguishability persists at scale: median
pairwise |Spearman| across all 36 <= 0.3; (b) depth-smoothness: for >= 60% of
components, the most-similar other fingerprint OF THE SAME TYPE is within +-2
layers; (c) type separation at fingerprint level: median within-type similarity
exceeds cross-type similarity (the relay's two stages leave different marks)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import per_token, attn_mean, spearman
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_fingerprint_atlas_results.json')
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'bilin18_fingerprint_atlas.pt')

@torch.no_grad()
def main():
    t0=time.time()
    d18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprints.pt')
    ce0=d18['base'].to(DEV)
    fps={k:v.float() for k,v in d18['fingerprints'].items()}
    for li in range(18):
        k=f'mlp{li}'
        if k not in fps:
            accs=[]
            for i in range(0,36,6):
                acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
                accs.append(acc[0])
            Y=torch.cat(accs); Ybar=Y.mean(0)
            _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
            Q=orth(Vh[:8].T)
            fps[k]=(per_token(mlp_span=(li,(Q,Ybar@Q)))-ce0).cpu().float()
            print(f'{k}: net {float(fps[k].mean()):+.4f}',flush=True)
    for li in range(18):
        k=f'attn{li}'
        if k not in fps:
            mu=attn_mean(li)
            fps[k]=(per_token(attn_layer=(li,mu))-ce0).cpu().float()
            print(f'{k}: net {float(fps[k].mean()):+.4f}',flush=True)
    torch.save({'base':ce0.cpu(),'fingerprints':fps},PT)
    keys=sorted(fps)
    S={}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            S[(a,b)]=abs(spearman(fps[a],fps[b]))
    allv=sorted(S.values()); med=allv[len(allv)//2]
    def sim(a,b): return S.get((a,b),S.get((b,a),0))
    smooth=0; tot=0
    for a in keys:
        typ='mlp' if a.startswith('mlp') else 'attn'
        la=int(a[len(typ):])
        same=[b for b in keys if b.startswith(typ) and b!=a]
        best=max(same,key=lambda b:sim(a,b))
        lb=int(best[len(typ):])
        tot+=1
        if abs(lb-la)<=2: smooth+=1
    within=[v for (a,b),v in S.items()
            if a[:3]==b[:3] or (a.startswith('mlp') and b.startswith('mlp'))]
    within=[v for (a,b),v in S.items()
            if (a.startswith('mlp')==b.startswith('mlp'))]
    cross=[v for (a,b),v in S.items()
           if (a.startswith('mlp')!=b.startswith('mlp'))]
    mw=sorted(within)[len(within)//2]; mc=sorted(cross)[len(cross)//2]
    pa=med<=0.3; pb=smooth/tot>=0.6; pc=mw>mc
    out={'n_components':len(keys),'median_pairwise':med,
         'depth_smooth_frac':smooth/tot,'within_type':mw,'cross_type':mc,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\n{len(keys)} components | pairwise med {med:.2f} | depth-smooth '
          f'{smooth}/{tot} | within {mw:.2f} vs cross {mc:.2f}')
    print(f"(a) distinguishable at scale: {'HELD' if pa else 'FAILED'}")
    print(f"(b) depth-smooth (>=60%): {'HELD' if pb else 'FAILED'}")
    print(f"(c) type separation in fingerprints: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} and {PT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
