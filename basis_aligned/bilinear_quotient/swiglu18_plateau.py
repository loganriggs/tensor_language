"""The last loose end: swiglu18's front plateau (its MLP components at L2-L5 all
best-matched bilin18's L1 on the direct instrument). Real compression, or argmax
degeneracy? Per plateau layer, report the L1 match, the runner-up, and the
margin.

REGISTERED (either outcome closes it): (a) if >= 3/4 plateau layers have
margin < 0.05, the plateau is argmax noise near a tie -- dissolved; (b) if
>= 3/4 have margin >= 0.05, swiglu18's early stack genuinely compresses onto
the bilinear crown layer -- a bounded, real family difference."""
import json, torch, time
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'swiglu18_plateau_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    asw=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'swiglu18_atlas.pt')['fingerprints']
    a18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprint_atlas.pt')['fingerprints']
    f18={lj:a18[f'mlp{lj}'].float() for lj in range(18)}
    res={}; big=0
    for li in (2,3,4,5):
        cur={lj:abs(spearman(asw[f'mlp{li}'].float(),f18[lj])) for lj in range(18)}
        order=sorted(cur,key=cur.get,reverse=True)
        best,ru=order[0],order[1]
        margin=cur[best]-cur[ru]
        res[li]={'best':best,'best_r':cur[best],'runner':ru,'runner_r':cur[ru],
                 'margin':margin}
        if margin>=0.05: big+=1
        print(f'swiglu mlp{li}: best L{best} ({cur[best]:.3f}) | runner L{ru} '
              f'({cur[ru]:.3f}) | margin {margin:.3f}',flush=True)
    verdict='real compression' if big>=3 else \
            ('argmax noise' if big<=1 else 'mixed')
    out={'per_layer':{str(k):v for k,v in res.items()},'n_big_margin':big,
         'verdict':verdict}
    print(f'\nverdict: {verdict} ({big}/4 margins >= 0.05)')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
