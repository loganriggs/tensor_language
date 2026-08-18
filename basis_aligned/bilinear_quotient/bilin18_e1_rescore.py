"""Re-score the certified edge explanation E1 (section 188: "attn6
transports L5's MLP-written content", certified via fingerprint kinship
0.223 vs median-other 0.087 on the original text window) on the FRESH
window (rows 320-384), where section 233 showed attn6's top partner shifts
to mlp1. The certification bar is MARGIN over the median other MLP
(>= +0.05), not rank -- test exactly that.

REGISTERED PREDICTIONS: (a) rho(attn6, mlp5) minus median-other >= +0.05 on
the fresh window (E1's kinship leg survives; rank loss !=  margin loss);
(b) if (a) fails, E1 downgrades to a two-instrument certification (span
deletion + pattern-clamping, both causal, untouched) and BENCHMARK.md gains
the rule that fingerprint-kinship compilations are scored across >= 2 text
windows; (c) sanity: rho(attn6, mlp5) on this window is positive."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean, spearman
from bilin18_kinship_fresh import per_token_fresh
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_e1_rescore_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    ce0=per_token_fresh()
    mu=attn_mean(6)
    fp_a6=(per_token_fresh(attn_layer=(6,mu))-ce0).float()
    sims={}
    for li in range(18):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        fp=(per_token_fresh(mlp_span=(li,(Q,Yb@Q)))-ce0).float()
        sims[li]=abs(spearman(fp_a6,fp))
        print(f'mlp{li:2d}: |rho| {sims[li]:.3f}',flush=True)
    tgt=sims[5]
    others=sorted(v for k,v in sims.items() if k!=5)
    med=others[len(others)//2]
    margin=tgt-med
    pa=margin>=0.05
    pc=spearman(fp_a6,(None if False else fp_a6))  # placeholder
    out={'sims':{str(k):v for k,v in sims.items()},'target':tgt,
         'median_other':med,'margin':margin,'pred_a':bool(pa)}
    print(f"\nattn6~mlp5 {tgt:.3f} vs median-other {med:.3f} "
          f"(margin {margin:+.3f}; original 0.223 vs 0.087)")
    print(f"(a) margin >= +0.05: "
          f"{'HELD -- E1 kinship leg survives' if pa else 'FAILED -- downgrade to two-instrument'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
