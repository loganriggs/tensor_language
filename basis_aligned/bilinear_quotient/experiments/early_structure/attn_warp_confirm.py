"""Confirmatory run for section 177's localization: the cross-family warp
should appear directly in the ATTENTION-side joint embedding. Embed bilin18's
18 + bilin12's 12 + sqrd12's 3 attention fingerprints jointly; orient the axis
by depth; measure each sqrd12 component's axis-implied fraction (nominal
fraction of its nearest bilinear-model neighbor on the axis).

REGISTERED PREDICTIONS: (a) the attention joint coordinate tracks fraction for
the two bilinear models at >= 0.7; (b) warp signature: all 3 sqrd12 attention
components' axis-implied fractions are LESS than nominal (front-shifted), and
the median shift is >= 0.05; (c) shuffled null <= 0.3."""
import json, torch, time
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'attn_warp_confirm_results.json')

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
    asq=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'sqrd12_fingerprints.pt')['fingerprints']
    vecs=[];labels=[]
    for li in range(18):
        vecs.append(a18[f'attn{li}'].float()); labels.append(('b18',li/18))
    for li in range(12):
        vecs.append(a12[f'attn{li}'].float()); labels.append(('b12',li/12))
    for li in (1,2,6):
        vecs.append(asq[f'attn{li}'].float()); labels.append(('sq',li/12))
    f=fiedler(vecs)
    bl=[i for i,l in enumerate(labels) if l[0]!='sq']
    sq=[i for i,l in enumerate(labels) if l[0]=='sq']
    sb=spearman(f[bl],torch.tensor([labels[i][1] for i in bl]))
    if sb<0: f=-f; sb=-sb
    shifts=[]
    for i in sq:
        nb=min(bl,key=lambda j:abs(float(f[j]-f[i])))
        shifts.append(labels[i][1]-labels[nb][1])
        print(f'sqrd12 attn fraction {labels[i][1]:.2f} -> axis-implied '
              f'{labels[nb][1]:.2f} (shift {labels[i][1]-labels[nb][1]:+.2f})',
              flush=True)
    g=torch.Generator().manual_seed(0)
    vecs_n=[v[torch.randperm(len(v),generator=g)] for v in vecs]
    fn=fiedler(vecs_n)
    sn=abs(spearman(fn,torch.tensor([l[1] for l in labels])))
    med=sorted(shifts)[len(shifts)//2]
    pa=sb>=0.7
    pb=all(s>0 for s in shifts) and med>=0.05
    pc=sn<=0.3
    out={'bilinear_axis':sb,'shifts':shifts,'median_shift':med,'null':sn,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'bilinear-pair axis {sb:.2f} | median front-shift {med:+.2f} | '
          f'null {sn:.2f}')
    print(f"(a) axis holds (>=0.7): {'HELD' if pa else 'FAILED'}")
    print(f"(b) all front-shifted, median >=0.05: {'HELD' if pb else 'FAILED'}")
    print(f"(c) null: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
