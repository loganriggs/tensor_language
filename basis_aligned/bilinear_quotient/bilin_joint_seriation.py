"""Joint cross-model seriation: embed BOTH models' MLP fingerprints (18 + 12
components) in one spectral ordering of the pooled similarity graph. If causal
marks form one family-wide coordinate system, the joint embedding should (i)
order each model's own layers correctly and (ii) interleave the two models by
depth fraction (the family law, sections 165-168).

REGISTERED PREDICTIONS: (a) within-model order preserved in the joint
embedding: |Spearman| >= 0.8 for both models; (b) cross-model alignment: the
joint coordinate rank-correlates with depth FRACTION across all 30 components
at >= 0.75; (c) shuffled-token null <= 0.3."""
import json, torch, time
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin_joint_seriation_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def fiedler(vecs):
    n=len(vecs)
    S=torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            S[i,j]=abs(spearman(vecs[i],vecs[j]))
    L=torch.diag(S.sum(1))-S
    ev,U=torch.linalg.eigh(L)
    return U[:,1]

def main():
    t0=time.time()
    a18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprint_atlas.pt')['fingerprints']
    a12=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin12_atlas.pt')['fingerprints']
    vecs=[]; labels=[]
    for li in range(18):
        vecs.append(a18[f'mlp{li}'].float()); labels.append(('b18',li,li/18))
    for li in range(12):
        vecs.append(a12[f'mlp{li}'].float()); labels.append(('b12',li,li/12))
    f=fiedler(vecs)
    i18=[i for i,l in enumerate(labels) if l[0]=='b18']
    i12=[i for i,l in enumerate(labels) if l[0]=='b12']
    s18=abs(spearman(f[i18],torch.tensor([labels[i][1] for i in i18],
                                         dtype=torch.float)))
    s12=abs(spearman(f[i12],torch.tensor([labels[i][1] for i in i12],
                                         dtype=torch.float)))
    sfrac=abs(spearman(f,torch.tensor([l[2] for l in labels])))
    g=torch.Generator().manual_seed(0)
    vecs_n=[v[torch.randperm(len(v),generator=g)] for v in vecs]
    fn=fiedler(vecs_n)
    sn=abs(spearman(fn,torch.tensor([l[2] for l in labels])))
    pa=s18>=0.8 and s12>=0.8
    pb=sfrac>=0.75
    pc=sn<=0.3
    out={'within_b18':s18,'within_b12':s12,'joint_fraction':sfrac,'null':sn,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'within-model order: b18 {s18:.2f} | b12 {s12:.2f}')
    print(f'joint coordinate vs depth fraction (30 comps): {sfrac:.2f} '
          f'(null {sn:.2f})')
    print(f"(a) within-model preserved: {'HELD' if pa else 'FAILED'}")
    print(f"(b) family coordinate (>=0.75): {'HELD' if pb else 'FAILED'}")
    print(f"(c) null: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
