"""The decisive framing test. causal_split_loro refuted the causal story:
upstream (acausal) folds share at 0.51 -- readers' forms agree even on
signals they never receive. Hypothesis: the shared vocabulary is GLOBAL
geometry of the readers' weights, visible through any projection, and writer
identity only modulates it. Test: behavioral LORO with V = random orthonormal
D->48 projection (3 seeds) and activations for the R^2 eval taken as the
residual stream at L8 (a generic mid-depth signal) projected through the same
random V. Compare to the writer-V numbers (L0 0.70, L1 0.64, L6 0.16).

REGISTERED PREDICTIONS: (a) random-V LORO >= 0.35 (sharing is mostly global
weight geometry -- the writer-specific numbers measure excess over this
baseline, and sections 61/208/209 get rescoped); (b) random-V LORO stays
below writer-L0's 0.70 by >= 0.15 (writer coords still add something); (c) if
random-V < 0.2, the global-geometry story dies and writer-coordinate
vocabularies stand as measured."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_random_v_loro_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    readers=(2,3,5,9,13,17)
    Ymid=grab(8,384,448)          # generic mid-depth signal for the eval
    meds=[]
    for seed in range(3):
        g=torch.Generator(device=DEV).manual_seed(seed)
        V=orth(torch.randn(D,K,device=DEV,generator=g))
        yproj=((Ymid-Ymid.mean(0)).float()@V)[:20000]
        fams={}
        for j in readers:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            P=orth(Vhj[:NF].T)
            mlp=m.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
        r2s=[]
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            Basis=W[:80]
            for Mm in fams[jout][:12]:
                c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=c_true.var().clamp_min(1e-12)
                Mre=((Basis@Mm.flatten())@Basis).view(K,K)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((c_hat-c_true)**2).mean()/vt))
        med=sorted(r2s)[len(r2s)//2]
        meds.append(med)
        print(f'seed {seed}: random-V behavioral LORO {med:+.3f}',flush=True)
    mm=sorted(meds)[1]
    pa=mm>=0.35; pb=(0.699-mm)>=0.15; pc=mm<0.2
    out={'meds':meds,'median':mm,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f'\nrandom-V median {mm:+.3f} (writer-V: L0 0.70, L1 0.64, L6 0.16)')
    print(f"(a) >=0.35 global geometry: {'HELD' if pa else 'FAILED'}")
    print(f"(b) L0 exceeds by >=0.15: {'HELD' if pb else 'FAILED'}")
    print(f"(c) <0.2 story dies: {'YES' if pc else 'no'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
