"""Seriation: is the stack's ordering recoverable from causal marks alone?
The atlas (section 173) is depth-smooth; the strong form: embed the fingerprint
similarity matrix (1D spectral ordering) and compare to true layer order.

REGISTERED PREDICTIONS: (a) MLP fingerprints: |Spearman(first nontrivial
eigenvector of the similarity graph, layer index)| >= 0.8; (b) attention
fingerprints: >= 0.6; (c) shuffled-fingerprint null (permute token axis
independently per component): |Spearman| <= 0.3."""
import json, torch, time
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'bilin18_fingerprint_atlas.pt')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_seriation_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def seriate(vecs):
    n=len(vecs)
    S=torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            S[i,j]=abs(spearman(vecs[i],vecs[j]))
    Dg=torch.diag(S.sum(1))
    L=Dg-S
    ev,U=torch.linalg.eigh(L)
    fiedler=U[:,1]
    return fiedler,S

def main():
    t0=time.time()
    d=torch.load(PT)
    fps={k:v.float() for k,v in d['fingerprints'].items()}
    g=torch.Generator().manual_seed(0)
    out={}
    for typ,bar in (('mlp',0.8),('attn',0.6)):
        idxs=list(range(18))
        vecs=[fps[f'{typ}{li}'] for li in idxs]
        f,S=seriate(vecs)
        sp=abs(spearman(f,torch.tensor(idxs,dtype=torch.float)))
        vecs_n=[v[torch.randperm(len(v),generator=g)] for v in vecs]
        fn,_=seriate(vecs_n)
        spn=abs(spearman(fn,torch.tensor(idxs,dtype=torch.float)))
        out[typ]={'spearman':sp,'null':spn,'bar':bar,'held':bool(sp>=bar)}
        print(f'{typ}: seriation |rho| {sp:.2f} (null {spn:.2f}) -> '
              f'{"HELD" if sp>=bar else "FAILED"}',flush=True)
    pc=all(out[t]['null']<=0.3 for t in out)
    out['null_ok']=bool(pc)
    print(f"(c) shuffled null <=0.3: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
