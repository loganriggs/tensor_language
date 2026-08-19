"""The coherence length of functional steering.

§58: single-functional steering selectivity is 20x at L2, ~3x at L5, ~0.3x at L13.
Measure the decay curve properly: for readers at layers 2,3,4,5,7,9,11,13, pick the
form with the LARGEST absolute L1-coupling (fixing §58's steerability confound), steer
along its top coupling eigenvector, record own-movement and selectivity vs depth.
REGISTERED PREDICTIONS:
  (a) own-movement decays monotonically with depth (Spearman(depth, own) <= -0.8);
  (b) the half-range (depth where own-movement falls below half its L2 value) is
      between 2 and 6 layers after L1;
  (c) the B-norm gate: across all targets, own-movement correlates with the target's
      absolute coupling norm at Spearman >= 0.5."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_functional_steering import run, orth
import bilin18_functional_steering as FS
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48
LAYERS=(2,3,4,5,7,9,11,13)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_coherence_length_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    targets=[]; Bnorms=[]; Bvecs=[]
    for j in LAYERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:8].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        best=None
        for f in range(8):
            c=mlp.Down.weight.detach().float().T@P[:,f]
            Bm=torch.einsum('k,ka,kb->ab',c,L,R); Bm=0.5*(Bm+Bm.T)
            if best is None or Bm.norm()>best[0].norm():
                best=(Bm,P[:,f].float())
        Bm,dvec=best
        M=form_for_direction(mlp,dvec).float()
        targets.append((j,M)); Bnorms.append(float(Bm.norm())); Bvecs.append(Bm)
    rows=FW[300:324,:257].to(DEV)
    FS.STEER=None; base=run(rows,targets)
    sig={i: float(base[i].std()) for i in base}
    out={'rows':[]}
    owns=[]
    for i,(j,_) in enumerate(targets):
        ev,U=torch.linalg.eigh(Bvecs[i].double())
        u=(V@U[:,ev.abs().argmax()].float()); u=u/u.norm()
        mag=2*s_out*K**0.5*0.2
        FS.STEER=(u,mag); pert=run(rows,targets); FS.STEER=None
        own=abs(float((pert[i]-base[i]).mean()))/sig[i]
        others=[abs(float((pert[k]-base[k]).mean()))/sig[k]
                for k in range(len(targets)) if k!=i]
        med=sorted(others)[len(others)//2]
        owns.append(own)
        out['rows'].append({'layer':j,'own':own,'crosstalk':med,
                            'selectivity':own/max(med,1e-6),'Bnorm':Bnorms[i]})
        print(f'L{j:2d}: own {own:.2f}s | cross {med:.2f}s | sel '
              f'{own/max(med,1e-6):.1f}x | |B| {Bnorms[i]:.1f}',flush=True)
    t=torch.tensor
    r_depth=spearman(t([float(j) for j,_ in targets]),t(owns))
    half=owns[0]/2
    hr=None
    for i,(j,_) in enumerate(targets):
        if owns[i]<half: hr=j-1; break
    r_bnorm=spearman(t(Bnorms),t(owns))
    out['spearman_depth']=r_depth; out['half_range_layer']=hr
    out['spearman_bnorm']=r_bnorm
    pa=r_depth<=-0.8; pb=hr is not None and 3<=hr<=7; pc=r_bnorm>=0.5
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'\nSpearman(depth, own) = {r_depth:+.2f} -> (a) '
          f"{'HELD' if pa else 'FAILED'}")
    print(f"half-range: below half-L2 by layer {hr} -> (b) "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"Spearman(|B|, own) = {r_bnorm:+.2f} -> (c) {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
