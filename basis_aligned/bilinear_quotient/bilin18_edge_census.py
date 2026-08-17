"""Edge census: response energy vs measured edge strength across all eight L0 leaders.

§52 validated the routing detector on five signals (three causal + two random). The
systematic version: all eight top Shapley directions of layer 0, energy first, then
measured |Delta c1| for each. REGISTERED PREDICTIONS: (a) Spearman(E, |dc1|) >= 0.7
over the eight; (b) at most one direction with E below 1e10 shows |dc1| >= 0.5 sigma
(low energy really means no edge)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_edge_mediation as EM
from bilin18_mediation_profile import per_head_E, spearman
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs); Y0c=(Y0-Y0.mean(0)).float()
    _,_,Vh0=torch.linalg.svd(Y0c, full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    order=phi0.argsort(descending=True)
    Q0=orth(Vh0[:32].T)
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh1=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    EM.RUN['M1']=form_for_direction(m.transformer.h[1].mlp,
                                    orth(Vh1[:32].T)[:,0].float()).float()
    rows=FW[300:324,:257].to(DEV)
    EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
    c1b,caps=EM.run(rows); s1=float(c1b.std())
    EM.CFG['cap1']=caps[1]; EM.CFG['cap0']=caps[0]
    out={'rows':[]}
    Es=[]; Ds=[]
    print(f"  {'rank':>5} {'dir':>4} {'total E':>10} {'|dc1|':>7}")
    for r_ in range(8):
        idx=int(order[r_]); d=Q0[:,idx].float()
        s0=float((Y0c@d).std())
        E=float(per_head_E(d,s0).sum())
        EM.CFG.update({'steer':(d,2*s0),'freeze1':None,'freeze0':None})
        c1p,_=EM.run(rows)
        EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
        dc=abs(float((c1p-c1b).mean())/s1)
        out['rows'].append({'rank':r_+1,'dir':idx,'E':E,'abs_dc1':dc})
        Es.append(E); Ds.append(dc)
        print(f"  {r_+1:>5} {idx:>4} {E:>10.2e} {dc:>7.3f}",flush=True)
    rho=spearman(torch.tensor(Es),torch.tensor(Ds))
    lowE_big=[r for r in out['rows'] if r['E']<1e10 and r['abs_dc1']>=0.5]
    pa=rho>=0.7; pb=len(lowE_big)<=1
    out['spearman']=rho; out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\nSpearman(E,|dc1|) = {rho:+.2f} -> (a) {'HELD' if pa else 'FAILED'}")
    print(f"low-E-but-routes exceptions: {len(lowE_big)} -> (b) "
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_edge_census_results.json','w'),indent=1)
    print(f'wrote bilin18_edge_census_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
