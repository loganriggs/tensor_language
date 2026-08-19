"""The 'different intervention class': does input-dependent injection restore
long-range addressability?

§60: no STATIC direction addresses a deep coefficient from L1 -- the exact gradient
fails (0.06 sigma at L13). The verdict named the escape hatch: input-dependent
injection. Cheapest version: per-SEQUENCE gradient steering -- compute the gradient of
the L13 coefficient separately for each of 12 sequences and steer each sequence along
its own direction, same total magnitude. REGISTERED PREDICTIONS:
  (a) per-sequence gradient steering at L13 achieves own-movement >= 3x the static
      gradient's 0.06 sigma (i.e. >= 0.18);
  (b) if (a) holds, selectivity stays >= 1x (it moves the target at least as much as
      the median other coefficient) -- addressability, not just power.
Either outcome sharpens 60: HELD -> context-dependence is the missing ingredient and
per-token injection would go further; FAILED -> even first-order context-adaptive
control cannot address depth, and the opacity is deeper than input-independence."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_contextual_steering_results.json')

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
    rows_all=FW[300:312,:257].to(DEV)
    with torch.no_grad():
        base,_=run_forward(rows_all,targets)
        sig={i: float(base[i].std()) for i in base}
    out={'cases':[]}
    for i,(j,_) in enumerate(targets):
        # per-sequence gradients
        pert_own=[]; pert_others={k: [] for k in range(len(targets)) if k!=i}
        for b in range(12):
            row=rows_all[b:b+1]
            outc,delta=run_forward(row,targets,need_grad=True)
            outc[i].mean().backward()
            g=delta.grad.detach(); g=g/g.norm()
            with torch.no_grad():
                p,_=run_forward(row,targets,steer=(g,mag))
                pb,_={},None
                b0,_=run_forward(row,targets)
                pert_own.append(float((p[i]-b0[i]).mean()))
                for k in pert_others:
                    pert_others[k].append(float((p[k]-b0[k]).mean()))
        own=abs(sum(pert_own)/12)/sig[i]
        others=[abs(sum(v)/12)/sig[k] for k,v in pert_others.items()]
        med=sorted(others)[len(others)//2] if others else 0.0
        out['cases'].append({'layer':j,'own_perseq':own,'crosstalk':med})
        print(f'L{j:2d}: per-sequence gradient own {own:.2f}s | cross {med:.2f}s',
              flush=True)
    l13=[c for c in out['cases'] if c['layer']==13][0]
    pa=l13['own_perseq']>=0.18
    pb=pa and l13['own_perseq']>=l13['crosstalk']
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) L13 own >= 0.18 (3x static): {'HELD' if pa else 'FAILED'}")
    print(f"(b) addressable (own >= cross): "
          f"{'HELD' if pb else 'FAILED/NA'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
