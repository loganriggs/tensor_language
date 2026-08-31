"""HOSVD PLATEAU DIAGNOSIS (rung 214): flat multilinear spectrum, or pipeline bug?

CONVENTION (S2135): probe only - no census. S2310: HOSVD at rin 34 and 128 gave identical damage at m16.
This run separates the two readings. (1) IDENTITY CHECK: the hook path (a = xW; y_d = a^T G_d a; out = yU^T)
is algebraically equal to direct two-sided projection x^T (WW^T Q_d WW^T) x - compare both on 256 captured
inputs at rin=34. (2) SPECTRUM: Frobenius capture of the shared projector, cap(r) = sum_d ||W_r^T Q_d W_r||^2
/ sum_d ||Q_d||^2, at rin 34/128/512.
REGISTERED PREDICTIONS:
  (a) NO BUG: hook-path vs direct-projection rel-Frobenius <= 1e-3 (fp32 floor ~4e-4 per S2309).
  (b) FLATNESS EXPLAINS THE PLATEAU: cap(128) - cap(34) <= 0.10.
  (c) THE SPECTRUM DOES RISE EVENTUALLY: cap(512) >= cap(128) + 0.10.
NULL: pred_a fails -> bug (S2310 stays "failed as run", hunt); or pred_b fails (capture rises but CE
does not) -> Frobenius-CE mismatch, a deeper and more interesting failure. PRICE: probe. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['m16_hosvd_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: HOSVD plateau diagnosis'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m16_hosvd_diag_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    mlp=m.transformer.h[16].mlp
    cap={}
    def pre(mo,i_): cap['x']=i_[0].detach().reshape(-1,D).float()[:256].clone()
    hp=mlp.register_forward_pre_hook(pre)
    bb=ROWS[:4,:257].to(DEV); idx=bb[:,:-1].contiguous()
    with torch.no_grad():
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    hp.remove()
    X=cap['x']
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float()
    U,_,_=torch.linalg.svd(Dw,full_matrices=False)
    C=U.T@Dw
    CH=64
    M=torch.zeros(D,D,device=DEV)
    for d0 in range(0,D,CH):
        Qc=torch.einsum('dk,kp,kq->dpq',C[d0:d0+CH],L,Rw)
        Qs=0.5*(Qc+Qc.transpose(1,2))
        M+=torch.einsum('dpq,drq->pr',Qs,Qs)
        del Qc,Qs
    ew,evec=torch.linalg.eigh(M)
    Wfull=evec.flip(-1)
    RINS=(34,128,512)
    Ws={r:Wfull[:,:r].contiguous() for r in RINS}
    num={r:0.0 for r in RINS}; den=0.0
    G34=torch.zeros(D,34,34,device=DEV)
    ydir=torch.zeros(X.shape[0],D,device=DEV)
    P34=Ws[34]@Ws[34].T
    XP=X@P34
    for d0 in range(0,D,CH):
        Qc=torch.einsum('dk,kp,kq->dpq',C[d0:d0+CH],L,Rw)
        Qs=0.5*(Qc+Qc.transpose(1,2))
        den+=float(Qs.pow(2).sum())
        for r in RINS:
            T=torch.einsum('pr,dpq,qs->drs',Ws[r],Qs,Ws[r])
            num[r]+=float(T.pow(2).sum())
            if r==34: G34[d0:d0+CH]=T
        ydir[:,d0:d0+CH]=torch.einsum('bp,dpq,bq->bd',XP,Qs,XP)
        del Qc,Qs
    capr={r:num[r]/den for r in RINS}
    a=X@Ws[34]
    yhook=torch.einsum('br,drs,bs->bd',a,G34,a)
    idfrob=float((yhook-ydir).norm()/max(float(ydir.norm()),1e-12))
    print(f'identity check rel-Frobenius {idfrob:.2e}; capture {capr}',flush=True)
    pa=idfrob<=1e-3
    pb=(capr[128]-capr[34])<=0.10
    pc=capr[512]>=capr[128]+0.10
    out={'identity_frob':idfrob,'capture':{str(r):round(capr[r],4) for r in RINS},
         'top_eigs':[round(float(v),3) for v in ew.flip(0)[:12].cpu()],
         'pred_a_no_bug':bool(pa),'pred_b_flat':bool(pb),'pred_c_rises':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) identity {idfrob:.2e} <= 1e-3: {'HELD' if pa else 'FAILED'}")
    print(f"(b) cap128-cap34 {capr[128]-capr[34]:.3f} <= 0.10: {'HELD' if pb else 'FAILED'}")
    print(f"(c) cap512 {capr[512]:.3f} >= cap128 {capr[128]:.3f} + 0.10: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
