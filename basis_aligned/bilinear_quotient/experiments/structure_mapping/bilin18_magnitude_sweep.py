"""The second-order mechanism's sharp prediction: selectivity ~ 1/magnitude.

§82: own-movement is linear in the injection, cross-talk quadratic. REGISTERED
PREDICTIONS at the L2 target (healthy linear response): (a) own-movement scales
~linearly with magnitude (log-log slope 0.8-1.3) while monitored cross-talk scales
superlinearly (slope >= 1.6); (b) selectivity (own/cross) at 0.25x magnitude is
>= 2.5x the selectivity at 2x magnitude. At L13: (c) no magnitude in the sweep
achieves selectivity >= 1 (the linear own-term is ~0; nothing rescues depth)."""
import json, sys, time, torch, math
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_quiet_steering import get_targets
from bilin18_joint_removal import fwd, m, FW, DEV
D=1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_magnitude_sweep_results.json')

def main():
    t0=time.time()
    Y1c=collect_basis()
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag0=2*s_out*K**0.5*0.2
    targets=get_targets()
    rows=FW[300:312,:257].to(DEV)
    with torch.no_grad():
        base,_=run_forward(rows,targets)
        sig={i: float(base[i].std()) for i in base}
    l2=[i for i,(j,_) in enumerate(targets) if j==2][0]
    l13=[i for i,(j,_) in enumerate(targets) if j==13][0]
    out={'targets':{}}
    for tgt in (l2,l13):
        others=[i for i in range(len(targets)) if i!=tgt][:8]
        outc,delta=run_forward(rows,targets,need_grad=True)
        outc[tgt].mean().backward()
        g=delta.grad.detach(); g=g/g.norm()
        rowsdat=[]
        for scale in (0.25,0.5,1.0,2.0):
            with torch.no_grad():
                p,_=run_forward(rows,targets,steer=(g,scale*mag0))
                own=abs(float((p[tgt]-base[tgt]).mean()))/sig[tgt]
                cr=[abs(float((p[i]-base[i]).mean()))/sig[i] for i in others]
                med=sorted(cr)[len(cr)//2]
            rowsdat.append({'scale':scale,'own':own,'cross':med,
                            'sel':own/max(med,1e-9)})
            print(f'L{targets[tgt][0]} x{scale:4.2f}: own {own:.3f}s | cross '
                  f'{med:.3f}s | sel {own/max(med,1e-9):.1f}x',flush=True)
        out['targets'][targets[tgt][0]]=rowsdat
    r2=out['targets'][2]
    def slope(key):
        xs=[math.log(r['scale']) for r in r2 if r[key]>1e-4]
        ys=[math.log(r[key]) for r in r2 if r[key]>1e-4]
        n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
        return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/ \
               max(sum((x-mx)**2 for x in xs),1e-9)
    so=slope('own'); sc=slope('cross')
    pa=0.8<=so<=1.3 and sc>=1.6
    pb=r2[0]['sel']>=2.5*r2[-1]['sel']
    r13=out['targets'][13]
    pc=all(r['sel']<1 for r in r13)
    out['slope_own']=so; out['slope_cross']=sc
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'\nL2 log-log slopes: own {so:.2f}, cross {sc:.2f}')
    print(f"(a) own linear, cross superlinear: {'HELD' if pa else 'FAILED'}")
    print(f"(b) selectivity at 0.25x >= 2.5x that at 2x: "
          f"{'HELD' if pb else 'FAILED'} "
          f"({r2[0]['sel']:.1f} vs {r2[-1]['sel']:.1f})")
    print(f"(c) L13 never selective: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
