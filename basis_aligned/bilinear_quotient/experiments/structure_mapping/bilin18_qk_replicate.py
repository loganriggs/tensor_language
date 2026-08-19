"""Replication of the below-chance QK result (section 84) on fresh rows:
basis and projections from rows 36-96, key stats from rows 24-48. REGISTERED:
(a) QK median R^2 <= 0.0; (b) random control >= +0.25; (c) gap >= 0.3.

Original docstring: Do attention score-readers speak the same vocabulary as MLP readers?

The model's score is (x_q^T Wq^T k1)(x_q^T Wq2^T k2) -- a per-query-position
quadratic in the stream with key-dependent coupling Wq^T (k1 k2^T) Wq2. Using cached
key statistics E[k1 k2^T] per head (layers 2-4, 27 heads), each head contributes one
expected query-side coupling matrix over L1's output coordinates. REGISTERED
PREDICTIONS: (a) QK functionals reconstruct from the MLP-fit top-80 basis at median
R^2 >= 0.5 (one shared code across reader types); (b) the joint family (240 MLP + 27
QK) eff-rank stays <= 130 (QK adds few new dimensions);
Null: random symmetric matrices reconstruct at <= 0.1."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_qk_replicate_results.json')

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
    yproj=(Y1c@V)[:20000]
    rows=[]
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T)
            rows.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
    X=torch.stack(rows)
    _,_,W=torch.linalg.svd(X, full_matrices=False)
    B=W[:80]
    # cached key stats per head for layers 2-4
    caps={}
    def run_capture(idx):
        B_,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(5):
            blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            a=blk.attn
            hcur=F.rms_norm(x,(D,))
            def qk(l):
                z=F.rms_norm(l(hcur).view(B_,T,NH,HD),(HD,))
                return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B_,T,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            if li in (2,3,4):
                for h in range(NH):
                    key=(li,h)
                    kk=torch.einsum('btd,bte->de',k1_[:,:,h,:],k2[:,:,h,:])
                    caps[key]=caps.get(key,0)+kk
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
            s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B_,T,-1))
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
    for i in range(24,48,6):
        run_capture(FW[i:i+6,:257].to(DEV))
    qk_mats=[]
    for li in (2,3,4):
        a=m.transformer.h[li].attn
        Wq=(a.c_q.weight.detach().float()@V).view(NH,HD,K)
        Wq2=(a.c_q2.weight.detach().float()@V).view(NH,HD,K)
        for h in range(NH):
            Ek=caps[(li,h)]/24.0
            M=torch.einsum('da,de,eb->ab',Wq[h],Ek,Wq2[h])
            Ms=0.5*(M+M.T)
            qk_mats.append(Ms/Ms.norm().clamp_min(1e-12))
    r2s=[]
    g=torch.Generator(device=DEV).manual_seed(0)
    r2r=[]
    for Ms in qk_mats:
        c=torch.einsum('na,ab,nb->n',yproj,Ms,yproj)
        co=B@Ms.flatten(); Mre=(co@B).view(K,K)
        ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
        r2s.append(1-float(((ch-c)**2).mean()/c.var().clamp_min(1e-12)))
        A=torch.randn(K,K,device=DEV,generator=g); A=0.5*(A+A.T); A=A/A.norm()
        cA=torch.einsum('na,ab,nb->n',yproj,A,yproj)
        coA=B@A.flatten(); MreA=(coA@B).view(K,K)
        chA=torch.einsum('na,ab,nb->n',yproj,MreA,yproj)
        r2r.append(1-float(((chA-cA)**2).mean()/cA.var().clamp_min(1e-12)))
    med=sorted(r2s)[len(r2s)//2]; medr=sorted(r2r)[len(r2r)//2]
    Xj=torch.cat([X,torch.stack([Mm.flatten() for Mm in qk_mats])])
    sv=torch.linalg.svdvals(Xj); e=sv**2
    erj=float(e.sum()**2/(e**2).sum())
    out={'qk_median_r2':med,'random_r2':medr,'joint_effrank':erj,
         'n_qk':len(qk_mats)}
    pa=med<=0.0; pb=medr>=0.25; pc=(medr-med)>=0.3
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['ctrl']=bool(pc)
    print(f'QK functionals (27 heads): median R^2 from MLP basis {med:.2f} '
          f'(random {medr:.2f})')
    print(f'joint family eff-rank: {erj:.0f}')
    print(f"(a) shared code (>=0.5): {'HELD' if pa else 'FAILED'} | "
          f"(b) joint rank <=130: {'HELD' if pb else 'FAILED'} | "
          f"random ctrl: {'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
