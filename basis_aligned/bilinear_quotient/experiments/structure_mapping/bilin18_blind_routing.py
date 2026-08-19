"""The operator calculus, blind: predict the carrier head for a NEW signal, then verify.

§50 showed first-order qk-enrichment retrodicts head 1 for the L0-leader signal. The
real test is a signal never studied: L0's #2 causal direction (the number/quantifier
axis, §20). Protocol inside one run, in order:
  STEP 1 (weights only): qk-enrichment ranking of the new direction across attn1's 9
     heads; the argmax is the REGISTERED PREDICTED CARRIER, printed before any
     intervention.
  STEP 2: does the edge exist at all? Steer L0-#2 by +2 sigma, measure Delta c1.
     Registered: the edge exists (|Delta c1| >= 0.3 sigma) -- uncertain, and that is
     the point.
  STEP 3 (only if the edge exists): freeze-one-head sweep under the steer. Registered:
     the predicted head is the top single-head killer, and kills >= 40%.
Controls: inert-head expectations ride along from the sweep itself."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_edge_mediation as EM
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
NH,HD,D=9,128,1152

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs)
    _,_,Vh0=torch.linalg.svd((Y0-Y0.mean(0)).float(), full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    lead2=int(phi0.argsort(descending=True)[1])
    d=orth(Vh0[:32].T)[:,lead2].float()
    s0=float((((Y0-Y0.mean(0)).float())@d).std())
    # STEP 1: weights-only prediction
    a=m.transformer.h[1].attn
    g=torch.Generator(device=DEV).manual_seed(0)
    Rnd=torch.randn(D,64,device=DEV,generator=g); Rnd=Rnd/Rnd.norm(dim=0,keepdim=True)
    qk_e=torch.zeros(NH,device=DEV)
    for W in (a.c_q,a.c_k,a.c_q2,a.c_k2):
        zd=(W.weight.detach().float()@d).view(NH,HD)
        zr=(W.weight.detach().float()@Rnd).view(NH,HD,-1)
        qk_e+=((zd**2).sum(1))/((zr**2).sum(1).mean(-1)).clamp_min(1e-12)/4
    pred=int(qk_e.argmax())
    print(f'L0 causal direction #2 (dir {lead2}, number/quantifier axis)')
    print(f'STEP 1 -- weights-only qk-enrichment: '
          f'{[round(float(v),2) for v in qk_e]}')
    print(f'REGISTERED PREDICTED CARRIER: head {pred}\n')
    out={'direction':lead2,'qk_enrichment':[float(v) for v in qk_e],
         'predicted_head':pred}
    # STEP 2: edge existence
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
    def arm(freeze1=None):
        EM.CFG.update({'steer':(d,2*s0),'freeze1':freeze1,'freeze0':None})
        c1p,_=EM.run(rows)
        EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
        return float((c1p-c1b).mean())/s1
    full=arm()
    out['edge_dc1_sigma']=full
    print(f'STEP 2 -- edge: Delta c1 = {full:+.3f} sigma '
          f'({"EXISTS" if abs(full)>=0.3 else "NO EDGE at this magnitude"})')
    if abs(full)>=0.3:
        kills=[]
        for h in range(NH):
            dd=arm([h]); k=1-dd/full
            kills.append(k)
            print(f'  freeze head {h}: {dd:+.3f} ({100*k:.0f}% killed)',flush=True)
        out['kills']=kills
        top=int(torch.tensor(kills).argmax())
        pa=top==pred; pb=kills[pred]>=0.40
        out['top_killer']=top
        out['pred_carrier_held']=bool(pa); out['pred_magnitude_held']=bool(pb)
        print(f'\nSTEP 3 -- top killer: head {top} | predicted: head {pred} -> '
              f"{'HELD' if pa else 'FAILED'} | predicted head kills "
              f"{100*kills[pred]:.0f}% (>=40%: {'HELD' if pb else 'FAILED'})")
    else:
        out['edge_exists']=False
        print('STEP 3 skipped -- registered edge-existence prediction FAILED; the '
              'carrier prediction is untestable on this signal')
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_blind_routing_results.json','w'),indent=1)
    print(f'wrote bilin18_blind_routing_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
