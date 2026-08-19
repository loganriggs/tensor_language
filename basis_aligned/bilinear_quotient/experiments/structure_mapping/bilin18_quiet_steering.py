"""User's question (2026-08-17): the ladder never PENALIZED collateral. Fix that.

Constrained steering, linearized: g_t = grad of the target coefficient (L13) wrt an
additive L1-output injection; g_1..g_5 = grads of five monitored other coefficients.
Steer along P g_t, the projection of g_t off span{g_i} (Gram-Schmidt), same magnitude
as the section-60 runs. Diagnostics and REGISTERED PREDICTIONS:
  (d) overlap: ||(I-P) g_t|| / ||g_t|| -- how much of the target's sensitivity lies
      in the others' span. Registered from the intrinsic-opacity picture: >= 0.6
      (mostly shared), i.e. projection is costly;
  (a) projected steering keeps <= 50% of raw-gradient own-movement (the cost of
      quietness is high) -- the ALTERNATIVE (keeps >50% AND cuts monitored cross-talk
      >= 3x) would mean addressability was recoverable all along and sections 60-64
      need correcting;
  (b) Goodhart control: cross-talk on SIX UNMONITORED fresh coefficients drops by
      < 2x (quietness does not transfer to what you didn't penalize);
Also run at L5 where steering is healthy, same structure."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_quiet_steering_results.json')

def get_targets():
    tg=[]
    for j,fidxs in ((2,(0,3)),(5,(0,5)),(9,(1,)),(13,(0,4)),(17,(0,2)),(3,(0,1))):
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:8].T)
        for f in fidxs:
            tg.append((j,form_for_direction(m.transformer.h[j].mlp,
                                            P[:,f].float()).float()))
    return tg

def main():
    t0=time.time()
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag=2*s_out*K**0.5*0.2
    targets=get_targets()          # 11 coefficients across layers
    rows=FW[300:312,:257].to(DEV)
    with torch.no_grad():
        base,_=run_forward(rows,targets)
        sig={i: float(base[i].std()) for i in base}
    out={'cases':[]}
    # target index within `targets`: L13 f0 -> find; monitored: 5 others; unmonitored: 6 rest
    l13=[i for i,(j,_) in enumerate(targets) if j==13][0]
    l5=[i for i,(j,_) in enumerate(targets) if j==5][0]
    for tgt in (l5,l13):
        others=[i for i in range(len(targets)) if i!=tgt]
        monitored=others[:5]; unmon=others[5:]
        grads={}
        for i in [tgt]+monitored:
            outc,delta=run_forward(rows,targets,need_grad=True)
            outc[i].mean().backward()
            grads[i]=delta.grad.detach().clone()
        gt=grads[tgt]/grads[tgt].norm()
        Gm=torch.stack([grads[i]/grads[i].norm() for i in monitored])
        Q,_=torch.linalg.qr(Gm.T)
        proj=gt-Q@(Q.T@gt)
        overlap=1-float(proj.norm())          # fraction of gt inside span
        pdir=proj/proj.norm().clamp_min(1e-12)
        def arm(vec):
            with torch.no_grad():
                p,_=run_forward(rows,targets,steer=(vec,mag))
                own=abs(float((p[tgt]-base[tgt]).mean()))/sig[tgt]
                mon=[abs(float((p[i]-base[i]).mean()))/sig[i] for i in monitored]
                un=[abs(float((p[i]-base[i]).mean()))/sig[i] for i in unmon]
            def med(v): return sorted(v)[len(v)//2] if v else 0.0
            return own,med(mon),med(un)
        own_g,mon_g,un_g=arm(gt)
        own_p,mon_p,un_p=arm(pdir)
        case={'target_layer':targets[tgt][0],'overlap':overlap,
              'raw':{'own':own_g,'monitored':mon_g,'unmonitored':un_g},
              'projected':{'own':own_p,'monitored':mon_p,'unmonitored':un_p}}
        out['cases'].append(case)
        print(f"L{targets[tgt][0]}: overlap(g_t, span others) = {overlap:.2f}")
        print(f"  raw gradient : own {own_g:.2f}s | monitored {mon_g:.2f}s | "
              f"unmonitored {un_g:.2f}s")
        print(f"  projected    : own {own_p:.2f}s | monitored {mon_p:.2f}s | "
              f"unmonitored {un_p:.2f}s",flush=True)
    c13=[c for c in out['cases'] if c['target_layer']==13][0]
    pd_=c13['overlap']>=0.6
    keep=c13['projected']['own']/max(c13['raw']['own'],1e-9)
    cut=c13['raw']['monitored']/max(c13['projected']['monitored'],1e-9)
    pa=keep<=0.5
    alt=keep>0.5 and cut>=3
    gcut=c13['raw']['unmonitored']/max(c13['projected']['unmonitored'],1e-9)
    pb=gcut<2
    out['pred_d_shared_sensitivity']=bool(pd_)
    out['pred_a_quietness_costly']=bool(pa)
    out['alternative_addressable']=bool(alt)
    out['pred_b_goodhart']=bool(pb)
    print(f"\n(d) overlap >= 0.6: {'HELD' if pd_ else 'FAILED'} "
          f"({c13['overlap']:.2f})")
    print(f"(a) projection keeps <= 50% own: {'HELD' if pa else 'FAILED'} "
          f"(keeps {100*keep:.0f}%, cuts monitored cross-talk {cut:.1f}x)")
    if alt: print("ALTERNATIVE: addressability recoverable -- sections 60-64 need "
                  "correction")
    print(f"(b) Goodhart (unmonitored cut < 2x): {'HELD' if pb else 'FAILED'} "
          f"({gcut:.1f}x)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
