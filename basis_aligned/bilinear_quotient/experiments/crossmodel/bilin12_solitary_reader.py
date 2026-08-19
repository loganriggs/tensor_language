"""The landscape's second exceptional object. bilin18 has a solitary READER:
L17 (depth fraction 0.94) is the worst LORO fold for every writer tested
(-0.31 to 0.42, section 210). The private WRITER proved universal at matched
fraction (section 215). Same test for the solitary reader: per-fold
behavioral LORO table for bilin12, writers (0,1,2,5,8), readers (1,3,5,7,9,11)
minus self. bilin12's fraction-0.94 reader is L11.

REGISTERED PREDICTIONS: (a) L11 is the WORST fold for at least 4 of 5
writers (solitary reader universal at matched fraction); (b) L11's median
fold across writers <= 0.25; (c) control: the second-deepest reader L9 is
NOT uniformly worst (solitariness is specific to the last reader, not a
depth gradient artifact -- L9 worst for at most 1 writer)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_solitary_reader_results.json')

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
    table={}
    for Wl in (0,1,2,5,8):
        readers=tuple(r for r in (1,3,5,7,9,11) if r!=Wl)
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:K].T)
        yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
        fams={}
        for j in readers:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            P=orth(Vhj[:NF].T)
            mlp=m2.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
        table[Wl]={}
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            Basis=W[:80]
            r2s=[]
            for Mm in fams[jout][:12]:
                c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=c_true.var().clamp_min(1e-12)
                Mre=((Basis@Mm.flatten())@Basis).view(K,K)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((c_hat-c_true)**2).mean()/vt))
            table[Wl][jout]=sorted(r2s)[len(r2s)//2]
        row=' '.join(f'L{j}:{table[Wl][j]:+.2f}' for j in readers)
        print(f'writer L{Wl}: {row}',flush=True)
    worst={Wl:min(d,key=d.get) for Wl,d in table.items()}
    n11=sum(1 for v in worst.values() if v==11)
    n9=sum(1 for v in worst.values() if v==9)
    l11=[d[11] for d in table.values() if 11 in d]
    med11=sorted(l11)[len(l11)//2]
    pa=n11>=4; pb=med11<=0.25; pc=n9<=1
    out={'table':{str(k):{str(j):v for j,v in d.items()}
                  for k,d in table.items()},
         'worst':{str(k):v for k,v in worst.items()},
         'l11_median':med11,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f'\nworst reader per writer: {worst} | L11 median {med11:+.3f}')
    print(f"(a) L11 worst for >=4/5: {'HELD' if pa else 'FAILED'} ({n11}/5)")
    print(f"(b) L11 median <=0.25: {'HELD' if pb else 'FAILED'}")
    print(f"(c) L9 worst <=1: {'HELD' if pc else 'FAILED'} ({n9}/5)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
