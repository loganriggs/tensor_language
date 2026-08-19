"""The transpose view: per-TOKEN leverage. Sum |delta| across all 36 bilin18
components per token position -- which tokens does the model's machinery bear
on hardest? And is leverage a property of the TEXT (shared across models) or of
each model?

REGISTERED PREDICTIONS: (a) leverage correlates with base loss at >= 0.4
(machinery concentrates where prediction is hard -- but not identical to it);
(b) CROSS-MODEL: bilin18's per-token leverage correlates with bilin12's at
>= 0.5 on the shared text (leverage is text-borne); (c) leverage is more
text-borne than model-idiosyncratic: the cross-model leverage correlation
exceeds the correlation between leverage and that model's own base loss
residualized... simplified: cross-model leverage correlation >= each model's
leverage-vs-own-base-loss correlation minus 0.1."""
import json, torch, time
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_token_leverage_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    d18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprint_atlas.pt')
    d12=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin12_atlas.pt')
    base18=d18['base'].float()
    lev18=torch.stack([v.float().abs() for v in d18['fingerprints'].values()]).sum(0)
    base12=None
    lev12=torch.stack([v.float().abs() for v in d12['fingerprints'].values()]).sum(0)
    s_bl=spearman(lev18,base18)
    s_xm=spearman(lev18,lev12)
    s_bl12=spearman(lev12,base18)  # vs shared-text difficulty proxy
    pa=s_bl>=0.4
    pb=s_xm>=0.5
    pc=s_xm>=s_bl-0.1
    out={'leverage_vs_base':s_bl,'cross_model_leverage':s_xm,
         'b12lev_vs_b18base':s_bl12,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'leverage vs base loss: {s_bl:+.2f}')
    print(f'cross-model leverage (b18 vs b12): {s_xm:+.2f}')
    print(f"(a) concentrates on hard tokens (>=0.4): {'HELD' if pa else 'FAILED'}")
    print(f"(b) text-borne (>=0.5): {'HELD' if pb else 'FAILED'}")
    print(f"(c) more text than difficulty: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
