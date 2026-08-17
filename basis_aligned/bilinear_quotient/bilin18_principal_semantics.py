"""What ARE the top principal functionals? Structural naming against verified axes.

The 80-functional basis is shared vocabulary (§61). Its top components should relate
to the program's verified structures. For the top-5 principal coupling matrices
(leave-none-out fit on all 240): eigendecompose each (rank structure), and measure
alignment of their top eigenvectors with the program's verified L1-basis objects:
the register core direction u (§19), the L0-punct axis image, the L16/17 syntax-bus
axis pulled back... measurable cleanly: alignment with (i) u (the verified z
direction), (ii) the top-4 whitened form directions of the L1 leader, (iii) random
directions (null, 200 draws). REGISTERED PREDICTIONS:
  (a) principal functional #1's top eigenvector aligns with u at |cos| >= 0.4
      (the shared vocabulary is anchored on the verified register core);
  (b) at least 3 of the top-5 principal functionals have effective rank <= 6
      (the vocabulary is made of low-rank quadratics -- nameable structure);
  (c) null control: random-direction alignments stay under the 95th percentile."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from bilin18_whitened import sqrtm_psd
from bilin18_source_folding import forward_tracked
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_principal_semantics_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    rows=[]
    for j in READERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            rows.append((0.5*(M+M.T)).flatten())
    X=torch.stack(rows)
    _,sv,W=torch.linalg.svd(X, full_matrices=False)
    # verified axis: the z direction u (section 19 surrogate), in V-coordinates
    Xl=[]
    for i in range(0,96,6):
        _,xh,_=forward_tracked(FW[i:i+6,:513].to(DEV)); Xl.append(xh)
    Xh=torch.cat(Xl)
    S=(Xh.T@Xh/Xh.shape[0]).double(); ev,U=torch.linalg.eigh(S)
    kd=ev>1e-8*ev.max()
    Sih=(U[:,kd]*ev[kd].rsqrt())@U[:,kd].T; Shh=(U[:,kd]*ev[kd].sqrt())@U[:,kd].T
    d0=orth(torch.linalg.svd(Y1c,full_matrices=False)[2][:32].T)[:,0].float()
    M1=form_for_direction(m.transformer.h[1].mlp,d0).float()
    Mw=Shh@M1.double()@Shh; ew,Uw=torch.linalg.eigh(Mw)
    u=(Sih@Uw[:,ew.abs().argmax()]).float(); u=u/u.norm()
    uV=(V.T@u); uV=uV/uV.norm()
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'principals':[]}
    n_lowrank=0
    for p_ in range(5):
        Pm=W[p_].view(K,K)
        Pm=0.5*(Pm+Pm.T)
        evp,Up=torch.linalg.eigh(Pm.double())
        e=evp.abs()
        er=float(e.sum()**2/(e**2).sum())
        top=Up[:,e.argmax()].float()
        cu=abs(float(top@uV))
        nulls=[]
        for _ in range(200):
            r=torch.randn(K,device=DEV,generator=g); r=r/r.norm()
            nulls.append(abs(float(top@r)))
        p95=sorted(nulls)[190]
        if er<=6: n_lowrank+=1
        out['principals'].append({'rank':p_+1,'share':float(sv[p_]**2/(sv**2).sum()),
                                  'effrank':er,'cos_u':cu,'null_p95':p95})
        print(f'principal #{p_+1}: mass {float(sv[p_]**2/(sv**2).sum()):.3f} | '
              f'eff-rank {er:.1f} | |cos| with z-direction {cu:.2f} '
              f'(null p95 {p95:.2f})',flush=True)
    pa=out['principals'][0]['cos_u']>=0.4
    pb=n_lowrank>=3
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['n_lowrank']=n_lowrank
    print(f"\n(a) #1 aligns with z (|cos|>=0.4): {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=3 of top-5 low-rank (<=6): {'HELD' if pb else 'FAILED'} "
          f"({n_lowrank}/5)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
