"""Where head 1 re-aims, take two: scale-controlled and properly powered.

The first attempt (§48) invalidated itself: 28 content keys (underpowered), inert-head
control violated, and a global scale confound (the injected offset perturbs rms norms,
shrinking all q/k products). Fixes: (1) per-row L1-normalised |pattern| so only
RELATIVE reallocation counts; (2) classes defined by corpus frequency -- punctuation
class vs the 200 most frequent non-punctuation tokens in the rows (hundreds of keys
each); (3) the control is the same statistic under a random-direction steer of matched
magnitude (scale confound affects it equally; only the punctuation-specific
reallocation should differ). REGISTERED PREDICTION: under +2s L0-leader steering,
head 1's relative pattern mass on punctuation keys changes by >= 3x the change under
the random steer; head 6 shows < 1.5x."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_head1_aim as A1

enc=tiktoken.get_encoding('gpt2')

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
    g=torch.Generator(device=DEV).manual_seed(0)
    dr=torch.randn(1152,device=DEV,generator=g); dr=dr/dr.norm()
    rows=FW[300:324,:257].to(DEV)
    isp=torch.zeros_like(rows,dtype=torch.bool)
    for t in A1.PUNCT: isp|=(rows==t)
    uniq,cnt=rows.reshape(-1).unique(return_counts=True)
    freq=uniq[cnt.argsort(descending=True)]
    top=[int(t) for t in freq if int(t) not in A1.PUNCT][:200]
    isc=torch.zeros_like(rows,dtype=torch.bool)
    for t in top: isc|=(rows==t)
    print(f'punct keys {int(isp.sum())} | frequent-content keys {int(isc.sum())}')
    def relmass(pat,h,msk):
        p=pat[:,h].abs()
        p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)     # per-query normalised
        return float(p.sum(1)[msk].mean())              # mass landing on masked keys
    A1.STEER=None; pb=A1.run(rows)
    A1.STEER=(d0L0,2*s0); ps=A1.run(rows)
    A1.STEER=(dr,2*s0); pr=A1.run(rows)
    A1.STEER=None
    out={'heads':{}}
    for h in (1,6):
        base_p=relmass(pb,h,isp)
        d_lead=relmass(ps,h,isp)-base_p
        d_rand=relmass(pr,h,isp)-base_p
        ratio=abs(d_lead)/max(abs(d_rand),1e-9)
        out['heads'][h]={'base_punct_mass':base_p,'d_leader':d_lead,
                         'd_random':d_rand,'ratio':ratio}
        print(f'head {h}: punct rel-mass {base_p:.4f} | Δ leader-steer '
              f'{d_lead:+.5f} | Δ random-steer {d_rand:+.5f} | ratio {ratio:.1f}x',
              flush=True)
    r1=out['heads'][1]['ratio']; r6=out['heads'][6]['ratio']
    pa=r1>=3.0; pc=r6<1.5
    out['pred_held']=bool(pa); out['ctrl_ok']=bool(pc)
    print(f"\nprediction (head-1 ratio >= 3x): {'HELD' if pa else 'FAILED'} | "
          f"head-6 control < 1.5x: {'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_head1_aim2_results.json','w'),indent=1)
    print(f'wrote bilin18_head1_aim2_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
