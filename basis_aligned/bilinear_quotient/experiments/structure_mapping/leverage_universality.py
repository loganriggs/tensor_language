"""Four-model leverage universality: do sqrd12 and swiglu18 share the same
per-token leverage profile? REGISTERED: (a) all six model-pair leverage
correlations >= 0.5; (b) the cross-family pairs (involving sqrd12/swiglu18) are
not lower than the bilinear pair by more than 0.15 (text-borne leverage
transcends architecture).

Prior context -- the transpose view: per-TOKEN leverage. Sum |delta| across all 36 bilin18
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
     'leverage_universality_results.json')

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
    dsq=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'swiglu18_atlas.pt')
    dsr=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'sqrd12_fingerprints.pt')
    levs={'b18':lev18,
          'b12':torch.stack([v.float().abs() for v in
                             d12['fingerprints'].values()]).sum(0),
          'sw18':torch.stack([v.float().abs() for v in
                              dsq['fingerprints'].values()]).sum(0),
          'sq12':torch.stack([v.float().abs() for v in
                              dsr['fingerprints'].values()]).sum(0)}
    import itertools
    pairs={}
    for a,b in itertools.combinations(levs,2):
        pairs[f'{a}-{b}']=spearman(levs[a],levs[b])
        print(f'{a}-{b}: {pairs[f"{a}-{b}"]:+.2f}',flush=True)
    bilpair=pairs['b18-b12']
    crossvals=[v for k,v in pairs.items() if 'sw18' in k or 'sq12' in k]
    pa=all(v>=0.5 for v in pairs.values())
    pb=min(crossvals)>=bilpair-0.15
    out={'pairs':pairs,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) all pairs >= 0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) cross-family within 0.15 of bilinear pair: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
