"""Family check of ledger-18's finding. In bilin18, attn6's apparent privacy
was BORROWED -- carrying the mlp6 span's content -- and recovered fully when
orthogonalized (0.13 -> 0.45). Same test in bilin12: attn4 raw and
orthogonalized-against-mlp4-span behavioral LORO, control attn8, readers
(1,3,5,7,9,11) minus self.

REGISTERED PREDICTIONS: (a) bilin12 attn4 raw LORO <= 0.25 (the carrier
looks private there too); (b) orthogonalized attn4 >= 0.30 (recovery --
borrowed privacy is universal, completing the family symmetry of the whole
anomaly); (c) control attn8 moved < 0.10 by orthogonalization; (d) measured
random-basis nulls <= 0.1."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_attn_borrowed_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs('bilin12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def grab(li, r0, r1, typ='mlp'):
        outs=[]
        mod=getattr(m2.transformer.h[li],typ)
        h=mod.register_forward_hook(
            lambda mo_,i_,o_: outs.append(
                (o_[0] if isinstance(o_,tuple) else o_)
                .detach().reshape(-1,D).float()))
        for i in range(r0,r1,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y4=grab(4,0,120); mu4=Y4.mean(0)
    _,_,Vh4=torch.linalg.svd((Y4-mu4).float(), full_matrices=False)
    Qspan=orth(Vh4[:8].T)
    def loro_attn(Wl, orth_span):
        readers=tuple(r for r in (1,3,5,7,9,11) if r!=Wl)
        Yw=grab(Wl,0,300,'attn'); mu=Yw.mean(0)
        Ywc=(Yw-mu).float()
        if orth_span: Ywc=Ywc-(Ywc@Qspan)@Qspan.T
        _,_,Vh=torch.linalg.svd(Ywc, full_matrices=False)
        V=orth(Vh[:K].T)
        Yf=(grab(Wl,384,448,'attn')-mu).float()
        if orth_span: Yf=Yf-(Yf@Qspan)@Qspan.T
        yproj=(Yf@V)[:20000]
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
    for Wl,os_ in ((4,False),(4,True),(8,False),(8,True)):
        med,rnd=loro_attn(Wl,os_)
        res[(Wl,os_)]=(med,rnd)
        print(f'attn{Wl} {"orth" if os_ else "raw "}: LORO {med:+.3f} '
              f'(random {rnd:+.3f})',flush=True)
    pa=res[(4,False)][0]<=0.25
    pb=res[(4,True)][0]>=0.30
    pc=abs(res[(8,True)][0]-res[(8,False)][0])<0.10
    pd=all(v[1]<=0.1 for v in res.values())
    out={'attn4_raw':res[(4,False)][0],'attn4_orth':res[(4,True)][0],
         'attn8_raw':res[(8,False)][0],'attn8_orth':res[(8,True)][0],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd)}
    print(f"\n(a) attn4 raw <=0.25 borrowed-looking: {'HELD' if pa else 'FAILED'}")
    print(f"(b) attn4 orth >=0.30 recovery: {'HELD' if pb else 'FAILED'}")
    print(f"(c) control moved <0.10: {'HELD' if pc else 'FAILED'}")
    print(f"(d) nulls <=0.1: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
