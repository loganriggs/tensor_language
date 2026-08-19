"""URGENT DIAGNOSTIC: the prereg scorecard, using reader set (1,2,3,6,9,11),
measured bilin12 writer L4 at +0.63 -- the landscape scan with readers
(1,3,5,7,9,11) measured -0.08 (the notch, section 215's headline). A 0.7
swing from reader choice would make the private-writer claim ensemble-
dependent and force a scope correction. Measure L4 (and control L8) under
three reader sets: A=(1,3,5,7,9,11) [original], B=(1,2,3,6,9,11)
[scorecard], C=(2,3,5,7,9,10) [fresh mix].

REGISTERED PREDICTIONS: (a) if the discrepancy is real, set A reproduces
<= 0.05 and set B reproduces >= 0.5 (no bug -- true ensemble dependence);
(b) control L8 varies across sets by < 0.15 (the sensitivity, if real, is
notch-specific); (c) diagnosis fork recorded in advance: the A/B difference
isolates to the two readers that differ -- rerun A with 5 replaced by 2
(set D=(1,2,3,7,9,11)) and with 7 replaced by 6 (set E=(1,3,5,6,9,11));
whichever single swap flips the notch names the load-bearing reader."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_notch_readerset_results.json')
SETS={'A':(1,3,5,7,9,11),'B':(1,2,3,6,9,11),'C':(2,3,5,7,9,10),
      'D':(1,2,3,7,9,11),'E':(1,3,5,6,9,11)}

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
    cacheP={}
    def famsP(j):
        if j not in cacheP:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            cacheP[j]=orth(Vhj[:NF].T)
        return cacheP[j]
    def loro(Wl, readers):
        readers=tuple(r for r in readers if r!=Wl)
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:K].T)
        yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
        fams={}
        for j in readers:
            P=famsP(j)
            mlp=m2.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
        r2s=[]
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            B=W[:80]
            for Mm in fams[jout][:12]:
                ct=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=ct.var().clamp_min(1e-12)
                Mre=((B@Mm.flatten())@B).view(K,K)
                ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((ch-ct)**2).mean()/vt))
        return sorted(r2s)[len(r2s)//2]
    res={}
    for tag,rs in SETS.items():
        l4=loro(4,rs); l8=loro(8,rs)
        res[tag]={'L4':l4,'L8':l8}
        print(f'set {tag} {rs}: L4 {l4:+.3f}  L8 {l8:+.3f}',flush=True)
    pa=res['A']['L4']<=0.05 and res['B']['L4']>=0.5
    l8s=[v['L8'] for v in res.values()]
    pb=max(l8s)-min(l8s)<0.15
    out={'sets':res,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\n(a) A reproduces notch, B kills it: {'HELD -- ensemble-dependent' if pa else 'FAILED'}")
    print(f"(b) L8 stable across sets: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
