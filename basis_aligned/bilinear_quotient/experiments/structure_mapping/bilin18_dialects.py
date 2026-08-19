"""Dialects or noise? The reader population shares no vocabulary over the
private span (6:1-8), but two very different worlds are consistent with that:
(i) DIALECTS -- each reader's forms over the span are individually
low-dimensional and structured, just mutually misaligned (every reader
privately understands the whisper, differently); (ii) HARD CONTENT -- no
reader has structure over the span at all. Discriminator at matched rank 18:
per reader, behavioral R^2 of reconstructing its held-out 20 forms (over the
writer's top-8 coords, forms are 8x8 sym) from (SELF) a rank-18 basis fit on
its own other 20 forms vs (CROSS) a rank-18 basis fit on the other five
readers' 40 forms each. Writers: L6 (private) and L9 (shared control).

REGISTERED PREDICTIONS: (a) private writer: median (self - cross) >= 0.25
(dialects -- own basis far better than the population's); (b) control writer
L9: median (self - cross) <= 0.15 (a shared code means the population basis
is nearly as good as your own); (c) random rank-18 basis <= 0.1 everywhere."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; NF=40; KS=8; RB=18
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_dialects_results.json')

@torch.no_grad()
def run(Wl):
    readers=tuple(r for r in (2,3,5,9,13,17) if r!=Wl)
    Yw=grab(Wl,0,300); mu=Yw.mean(0)
    _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
    V=orth(Vh[:KS].T)
    yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
    fams={}
    for j in readers:
        Yj=grab(j,0,60)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        fams[j]=torch.stack([(0.5*(M+M.T)).flatten() for M in
                 (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                  for f in range(NF))])
    g=torch.Generator(device=DEV).manual_seed(0)
    gaps=[]; selfs=[]; crosses=[]; rnds=[]
    for j in readers:
        train=fams[j][0::2]; test=fams[j][1::2]
        _,_,Wt=torch.linalg.svd(train, full_matrices=False)
        Bself=Wt[:RB]
        X=torch.cat([fams[j2] for j2 in readers if j2!=j])
        _,_,Wx=torch.linalg.svd(X, full_matrices=False)
        Bcross=Wx[:RB]
        Brnd=torch.randn(RB,KS*KS,device=DEV,generator=g)
        Brnd=Brnd/Brnd.norm(dim=1,keepdim=True)
        med={}
        for tag,B in (('self',Bself),('cross',Bcross),('rnd',Brnd)):
            r2s=[]
            for v in test:
                Mm=v.view(KS,KS)
                c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=c_true.var().clamp_min(1e-12)
                Mre=((B@v)@B).view(KS,KS)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((c_hat-c_true)**2).mean()/vt))
            med[tag]=sorted(r2s)[len(r2s)//2]
        selfs.append(med['self']); crosses.append(med['cross'])
        rnds.append(med['rnd']); gaps.append(med['self']-med['cross'])
        print(f'W L{Wl} reader L{j:2d}: self {med["self"]:+.3f} '
              f'cross {med["cross"]:+.3f} rnd {med["rnd"]:+.3f}',flush=True)
    mg=sorted(gaps)[len(gaps)//2]
    return {'self':selfs,'cross':crosses,'rnd':rnds,'median_gap':mg}

@torch.no_grad()
def main():
    t0=time.time()
    r6=run(6); r9=run(9)
    pa=r6['median_gap']>=0.25
    pb=r9['median_gap']<=0.15
    pc=max(max(r6['rnd']),max(r9['rnd']))<=0.1
    out={'private_L6':r6,'control_L9':r9,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\nL6 median gap {r6['median_gap']:+.3f} | "
          f"L9 median gap {r9['median_gap']:+.3f}")
    print(f"(a) L6 gap >=0.25 dialects: {'HELD' if pa else 'FAILED'}")
    print(f"(b) L9 gap <=0.15 shared: {'HELD' if pb else 'FAILED'}")
    print(f"(c) randoms <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
