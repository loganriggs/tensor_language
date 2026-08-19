"""The scope-note run section 225 flagged: bilin12's "attn4 mildly
depressed" rested on one control. Full attention-writer profile, attn0-attn9,
readers (1,3,5,7,9,11) minus self, behavioral LORO, nulls measured.

REGISTERED PREDICTIONS: (a) attn4 is the MINIMUM of the ten (an attention-
side notch exists at 12L too, merely mild); alternative: attn4 sits within
one median-absolute-deviation of the profile median = no attention notch at
12L, and section 225's "mild depression" was sampling luck; (b) profile
median >= 0.30 (attention sharing is real if shallow at 12L); (c) measured
random-basis nulls <= 0.1 everywhere."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_attn_profile_results.json')

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
    res={}
    for Wl in range(10):
        readers=tuple(r for r in (1,3,5,7,9,11) if r!=Wl)
        Yw=grab(Wl,0,300,'attn'); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:K].T)
        yproj=(((grab(Wl,384,448,'attn'))-mu).float()@V)[:20000]
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
        med=sorted(r2s)[len(r2s)//2]; rnd=sorted(r2r)[len(r2r)//2]
        res[Wl]=(med,rnd)
        print(f'attn{Wl}: LORO {med:+.3f} (random {rnd:+.3f})',flush=True)
    meds=sorted(v[0] for v in res.values())
    prof_med=meds[len(meds)//2]
    mad=sorted(abs(v[0]-prof_med) for v in res.values())[5]
    mn=min(res,key=lambda k:res[k][0])
    pa=(mn==4)
    within=abs(res[4][0]-prof_med)<=mad
    pb=prof_med>=0.30
    pc=all(v[1]<=0.1 for v in res.values())
    out={'profile':{str(k):{'loro':v[0],'random':v[1]}
                    for k,v in res.items()},
         'median':prof_med,'mad':mad,'min_writer':mn,
         'pred_a':bool(pa),'attn4_within_mad':bool(within),
         'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\nmedian {prof_med:+.3f} MAD {mad:.3f} | minimum attn{mn}')
    print(f"(a) attn4 is minimum: {'HELD' if pa else 'FAILED'}"
          f"{' (attn4 within 1 MAD: no notch)' if within else ''}")
    print(f"(b) median >=0.30: {'HELD' if pb else 'FAILED'}")
    print(f"(c) nulls <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
