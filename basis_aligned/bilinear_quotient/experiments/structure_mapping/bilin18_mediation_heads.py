"""Which attn1 heads carry the L0->L1 edge? Freeze-one-head sweep.

§45: freezing all of attn1 kills 134% of the steered effect; head 4 alone kills 51%.
REGISTERED PREDICTIONS: (a) no other single head kills more than 30%; (b) the top-3
heads together account for >= 80% of the sum of single-head kills (concentrated, not
uniform). Uses the §45 machinery with freeze1={h} for each head."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_edge_mediation as EM
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs)
    _,_,Vh0=torch.linalg.svd((Y0-Y0.mean(0)).float(), full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    d0L0=orth(Vh0[:32].T)[:,int(phi0.argmax())].float()
    s0=float((((Y0-Y0.mean(0)).float())@d0L0).std())
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh1=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    EM.RUN['M1']=form_for_direction(m.transformer.h[1].mlp,
                                    orth(Vh1[:32].T)[:,0].float()).float()
    rows=FW[300:324,:257].to(DEV)
    EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
    c1b,caps_b=EM.run(rows); s1=float(c1b.std())
    EM.CFG['cap1']=caps_b[1]; EM.CFG['cap0']=caps_b[0]
    def arm(freeze1=None):
        EM.CFG.update({'steer':(d0L0,2*s0),'freeze1':freeze1,'freeze0':None})
        c1p,_=EM.run(rows)
        EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
        return float((c1p-c1b).mean())/s1
    full=arm()
    kills=[]
    print(f'steer alone: {full:+.3f} sigma')
    for h in range(9):
        d=arm([h]); k=1-d/max(full,1e-9)
        kills.append(k)
        print(f'  freeze head {h}: {d:+.3f}  ({100*k:.0f}% killed)',flush=True)
    out={'full':full,'kills':kills}
    srt=sorted(kills,reverse=True)
    others=[k for i,k in enumerate(kills) if i!=4]
    pa=max(others)<0.30
    ksum=sum(max(k,0) for k in kills)
    pb=sum(srt[:3])>=0.8*ksum if ksum>0 else False
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) no non-4 head kills >30%: {'HELD' if pa else 'FAILED'} "
          f"(max other {100*max(others):.0f}%)")
    print(f"(b) top-3 heads >= 80% of kill mass: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_mediation_heads_results.json','w'),indent=1)
    print(f'wrote bilin18_mediation_heads_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
