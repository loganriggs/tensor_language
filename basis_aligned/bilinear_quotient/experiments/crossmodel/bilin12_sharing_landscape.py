"""Universality test for the sharing landscape (sections 209-212): bilin18
shows behavioral vocabulary sharing that is writer-general (0.5-0.7) with one
private writer (L6, depth fraction 0.33) whose privacy lives in its top-8
span. Scan bilin12 (12 layers, D=768, squared attention): behavioral LORO
over writers 0-9, readers (1,3,5,7,9,11) minus self, K=48 writer coords,
rank-80 basis, fresh eval rows; plus a random-V floor (2 seeds).

REGISTERED PREDICTIONS: (a) sharing is writer-general in bilin12 -- at least
7 of 10 writers with LORO >= 0.35 (weaker bar than 18L: the smaller model
shares less, cf its weaker cross-model couplings); (b) a private writer
exists: the minimum writer sits within 0.05 of the random-V floor or below;
(c) placement long-shot via the depth-fraction law: IF (b) holds, the private
writer's fraction li/12 lies in [0.25,0.45] (bilin18's L6/18=0.33); (d) null:
random 80-basis <= 0.1 for every writer."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
import torch.nn.functional as F
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_sharing_landscape_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs('bilin12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def grab(li, r0, r1):
        outs=[]
        h=m2.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(r0,r1,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    def fam(j, V):
        Yj=grab(j,0,60)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m2.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        return [0.5*(M+M.T) for M in
                (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R) for f in range(NF))]
    def loro(V, yproj, readers):
        fams={j:fam(j,V) for j in readers}
        g=torch.Generator(device=DEV).manual_seed(0)
        r2s=[]; r2r=[]
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            Basis=W[:80]
            Rb=torch.randn(80,K*K,device=DEV,generator=g)
            Rb=Rb/Rb.norm(dim=1,keepdim=True)
            for Mm in fams[jout][:12]:
                c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=c_true.var().clamp_min(1e-12)
                for Bset,acc_ in ((Basis,r2s),(Rb,r2r)):
                    Mre=((Bset@Mm.flatten())@Bset).view(K,K)
                    c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                    acc_.append(1-float(((c_hat-c_true)**2).mean()/vt))
        return sorted(r2s)[len(r2s)//2], sorted(r2r)[len(r2r)//2]
    res={}
    for Wl in range(10):
        readers=tuple(r for r in (1,3,5,7,9,11) if r!=Wl)
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:K].T)
        yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
        med,rnd=loro(V,yproj,readers)
        res[Wl]=(med,rnd)
        print(f'writer L{Wl}: LORO {med:+.3f} (random-basis {rnd:+.3f})',
              flush=True)
    floors=[]
    Ymid=grab(5,384,448)
    for seed in range(2):
        g=torch.Generator(device=DEV).manual_seed(100+seed)
        V=orth(torch.randn(D,K,device=DEV,generator=g))
        yproj=((Ymid-Ymid.mean(0)).float()@V)[:20000]
        fmed,_=loro(V,yproj,(1,3,5,7,9,11))
        floors.append(fmed)
        print(f'random-V floor seed {seed}: {fmed:+.3f}',flush=True)
    floor=sum(floors)/2
    lo=[Wl for Wl,v in res.items()]
    mn=min(res, key=lambda k: res[k][0])
    pa=sum(1 for v in res.values() if v[0]>=0.35)>=7
    pb=res[mn][0]<=floor+0.05
    pc=(0.25<=mn/12<=0.45) if pb else None
    pd=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]} for k,v in res.items()},
         'floor':floor,'min_writer':mn,
         'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':None if pc is None else bool(pc),'pred_d':bool(pd)}
    print(f'\nfloor {floor:+.3f} | min writer L{mn} at {res[mn][0]:+.3f} '
          f'(fraction {mn/12:.2f})')
    print(f"(a) >=7/10 writers >=0.35: {'HELD' if pa else 'FAILED'}")
    print(f"(b) private writer exists: {'HELD' if pb else 'FAILED'}")
    if pc is not None:
        print(f"(c) placement in [0.25,0.45]: {'HELD' if pc else 'FAILED'}")
    print(f"(d) random-basis <=0.1: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
