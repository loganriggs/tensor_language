"""Certify the 16->17 interchange as an edge explanation using the corrected
relative-form kinship (section 189's rule): "attn17 reads what mlp16 writes."

REGISTERED PREDICTIONS: (a) kinship(attn17, mlp16) exceeds attn17's median
kinship with other MLPs by >= 0.05; (b) directionality: kinship(attn17, mlp16)
> kinship(attn17, mlp17) (it reads layer 16's write, not its own layer's);
(c) symmetric check from the MLP side: kinship(mlp17, mlp16) is mlp17's top
same-type partner (the interchange also marks consecutive MLPs)."""
import json, torch, time
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bilin18_interchange_kinship_results.json'

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    d=torch.load(PT+'bilin18_fingerprint_atlas.pt')
    fps={k:v.float() for k,v in d['fingerprints'].items()}
    ks={mj:abs(spearman(fps['attn17'],fps[f'mlp{mj}'])) for mj in range(18)}
    others=[v for mj,v in ks.items() if mj!=16]
    med=sorted(others)[len(others)//2]
    pa=(ks[16]-med)>=0.05
    pb=ks[16]>ks[17]
    m17={mj:abs(spearman(fps['mlp17'],fps[f'mlp{mj}']))
         for mj in range(18) if mj!=17}
    top=max(m17,key=m17.get)
    pc=top==16
    out={'k_attn17_mlp16':ks[16],'median_other':med,'k_attn17_mlp17':ks[17],
         'mlp17_top_partner':top,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f'attn17~mlp16 {ks[16]:.3f} vs median-other {med:.3f} | '
          f'attn17~mlp17 {ks[17]:.3f} | mlp17 top partner mlp{top}')
    print(f"(a) edge certifies (>=+0.05): {'HELD' if pa else 'FAILED'}")
    print(f"(b) reads 16 not 17: {'HELD' if pb else 'FAILED'}")
    print(f"(c) mlp17's top partner is mlp16: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
