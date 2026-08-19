"""Are the verified axes INSIDE the vocabulary's span (just not principal)?

§63: the top principal functionals are not the verified axes. §61: the vocabulary's
span reconstructs reader forms at R^2 0.71. Containment reconciles: project each
verified object's L1-coupling matrix onto the top-80 principal span. Objects: (i) the
z/register surrogate's coupling (rank-1 u u^T in V-coords weighted), (ii) the L1
leader's full coupling in V-coords, (iii) the L16-axis pullback... measurable cleanly:
(i) and (ii). REGISTERED PREDICTIONS: (a) energy of each verified object's coupling in
the top-80 span >= 0.55 (contained); (b) matched random symmetric matrices land at
80/1176 ~ 0.07 baseline, at least 5x below."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from bilin18_source_folding import forward_tracked
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_vocab_containment_results.json')

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
    B80=W[:80]
    # verified objects
    from bilin18_whitened import sqrtm_psd
    Xl=[]
    for i in range(0,96,6):
        _,xh,_=forward_tracked(FW[i:i+6,:513].to(DEV)); Xl.append(xh)
    Xh=torch.cat(Xl)
    S=(Xh.T@Xh/Xh.shape[0]).double(); ev,U=torch.linalg.eigh(S)
    kd=ev>1e-8*ev.max()
    Sih=(U[:,kd]*ev[kd].rsqrt())@U[:,kd].T; Shh=(U[:,kd]*ev[kd].sqrt())@U[:,kd].T
    d0=orth(Vh[:32].T)[:,0].float()
    M1=form_for_direction(m.transformer.h[1].mlp,d0).float()
    Mw=Shh@M1.double()@Shh; ew,Uw=torch.linalg.eigh(Mw)
    u=(Sih@Uw[:,ew.abs().argmax()]).float(); u=u/u.norm()
    uV=V.T@u; uV=uV/uV.norm()
    objs={'z-surrogate (u u^T)':torch.outer(uV,uV),
          'L1-leader coupling':(V.T@M1@V)}
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'objects':{}}
    rn=[]
    for _ in range(50):
        A=torch.randn(K,K,device=DEV,generator=g); A=0.5*(A+A.T)
        A=A/A.norm()
        rn.append(float(((B80@A.flatten())**2).sum()))
    rbase=sum(rn)/len(rn)
    for tag,Mo in objs.items():
        Mo=Mo/Mo.norm()
        e=float(((B80@Mo.flatten())**2).sum())
        out['objects'][tag]={'energy_in_span':e}
        print(f'{tag:24s}: energy in top-80 span {e:.3f} (random {rbase:.3f})',
              flush=True)
    out['random_baseline']=rbase
    pa=all(v['energy_in_span']>=0.55 for v in out['objects'].values())
    pb=all(v['energy_in_span']>=5*rbase for v in out['objects'].values())
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) contained (>=0.55): {'HELD' if pa else 'FAILED'} | "
          f"(b) >=5x random: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
