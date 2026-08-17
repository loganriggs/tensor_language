"""Last replication gap: the functional vocabulary's LORO (leave-one-reader-out)
R^2 = 0.71 -- the program's oldest headline (section 58 era), never re-verified
on fresh rows since the correction era. Rebuild the six-reader coupling family
over L1's top-48 output coords with FRESH stats rows (36-96), fit the top-80
basis on five readers, reconstruct the held-out reader's 40 functionals; median
over the six folds.

REGISTERED PREDICTIONS: (a) LORO median R^2 >= 0.55 (original 0.71); (b) random
symmetric-matrix control <= 0.45 (note: section 84's replicate showed the
fitted basis captures a generic share ~0.4-0.55 of random matrices -- the bar
reflects that); (c) the LORO median exceeds the random control by >= 0.15."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D,K,NF=1152,48,40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_loro_replicate_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    def collect(li):
        outs=[]
        h=m.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(36,96,6):
            b=FW[i:i+6,:513].to(DEV)
            m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    fams={}
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        rs=[]
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T)
            rs.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
        fams[j]=torch.stack(rs)
    folds=[]
    for hold in READERS:
        X=torch.cat([fams[j] for j in READERS if j!=hold])
        _,_,W=torch.linalg.svd(X, full_matrices=False)
        B=W[:80]
        te=fams[hold]
        rec=(te@B.T)@B
        r2=[1-float(((te[i]-rec[i])**2).sum()) for i in range(NF)]
        folds.append(sorted(r2)[NF//2])
        print(f'hold L{hold:2d}: median R^2 {folds[-1]:+.3f}',flush=True)
    loro=sorted(folds)[len(folds)//2]
    g=torch.Generator(device=DEV).manual_seed(0)
    X=torch.cat([fams[j] for j in READERS])
    _,_,W=torch.linalg.svd(X, full_matrices=False)
    B=W[:80]
    rr=[]
    for _ in range(NF):
        A=torch.randn(K,K,device=DEV,generator=g); A=0.5*(A+A.T)
        v=(A/A.norm()).flatten()
        rec=(v@B.T)@B
        rr.append(1-float(((v-rec)**2).sum()))
    rnd=sorted(rr)[len(rr)//2]
    pa=loro>=0.55; pb=rnd<=0.45; pc=(loro-rnd)>=0.15
    out={'folds':folds,'loro_median':loro,'random':rnd,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\nLORO median {loro:+.3f} (orig 0.71) | random {rnd:+.3f}')
    print(f"(a) >=0.55: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random <=0.45: {'HELD' if pb else 'FAILED'}")
    print(f"(c) gap >=0.15: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
