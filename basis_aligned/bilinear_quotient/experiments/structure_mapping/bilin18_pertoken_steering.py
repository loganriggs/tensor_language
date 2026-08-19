"""Intervention ladder, final rung: per-token gradient steering.

§62: per-SEQUENCE gradients lift L13 reachability 3.7x with no addressability
(cross-talk 13x own). Per-TOKEN steering -- each position steered along its own
gradient -- is the strongest first-order intervention class. Twelve positions,
one backward each. REGISTERED PREDICTIONS (from the ladder's pattern):
  (a) L13 own-movement >= 0.4 sigma (power keeps rising);
  (b) selectivity still < 1 (addressability still absent) -- the pattern completes:
      power is recoverable at every rung, selectivity at none."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_pertoken_steering_results.json')

def main():
    t0=time.time()
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag=2*s_out*K**0.5*0.2
    targets=[]
    for j in (5,13):
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
        targets.append((j,form_for_direction(mlp,best[1]).float()))
    row=FW[300:301,:257].to(DEV)   # per-token on one sequence, 12 positions
    with torch.no_grad():
        base,_=run_forward(row,targets)
    out={'cases':[]}
    import torch.nn.functional as F
    for i,(j,_) in enumerate(targets):
        own_moves=[]; cross_moves=[]
        for q in range(40,160,10):
            # gradient of c_i at position q wrt static delta (approximates per-token
            # direction for the window ending at q)
            outc,delta=run_forward(row,targets,need_grad=True)
            outc[i][0,q].backward()
            g=delta.grad.detach(); g=g/g.norm().clamp_min(1e-12)
            with torch.no_grad():
                p,_=run_forward(row,targets,steer=(g,mag))
                b0,_=run_forward(row,targets)
                sig_i=float(base[i].std())
                own_moves.append(abs(float(p[i][0,q]-b0[i][0,q]))/sig_i)
                ks=[k for k in range(len(targets)) if k!=i]
                cross_moves.append(sum(abs(float((p[k]-b0[k]).mean()))
                                       /float(base[k].std()) for k in ks)/len(ks))
        own=sum(own_moves)/len(own_moves)
        cross=sum(cross_moves)/len(cross_moves)
        out['cases'].append({'layer':j,'own_pertoken':own,'cross':cross})
        print(f'L{j:2d}: per-token own {own:.2f}s | cross {cross:.2f}s',flush=True)
    l13=[c for c in out['cases'] if c['layer']==13][0]
    pa=l13['own_pertoken']>=0.4
    pb=l13['own_pertoken']<l13['cross']
    out['pred_a']=bool(pa); out['pred_b_no_addressability']=bool(pb)
    print(f"\n(a) L13 own >= 0.4: {'HELD' if pa else 'FAILED'} | "
          f"(b) still unaddressable: {'CONFIRMED' if pb else 'REFUTED -- addressable!'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
