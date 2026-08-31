"""FP64 EXACTNESS CONTROL (rung 211): is S2307's 1.54e-2 max/RMS residual fp32 precision or a bug?

CONVENTION (S2135): probe only - no census, no damage claims. The eigen pipeline (out basis U = SVD(Down);
Q_d = L^T diag((U^T Down)_d) R; out = U @ (x^T Q x) + b) is mathematically exact. v2 measured rel-Frobenius
4.02e-4 but max-abs/RMS 1.54e-2 in fp32. Here the identical pipeline runs in BOTH precisions on the same
256 captured m16 inputs.
REGISTERED PREDICTIONS (arms: FP32 = recomputation; FP64 = double-precision path):
  (a) PRECISION EXPLANATION: FP64 max-abs/RMS <= 1e-6 (quarantine on S2307's ratios lifts).
  (b) FP64 rel-Frobenius <= 1e-9.
  (c) REPRODUCTION: FP32 max/RMS in [0.5e-2, 5e-2] (same instrument as v2).
NULL: FP64 residual stays >= 1e-4 - a real pipeline bug; S2307 stays unpublished and the hunt starts.
PRICE: probe. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['m16_eigenbasis_v2_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: fp64 exactness control'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m16_exact_fp64_results.json'

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
    res={}
    for tag,dt,ch in (('FP32',torch.float32,64),('FP64',torch.float64,32)):
        X=cap['x'].to(dt)
        L=mlp.Left.weight.detach().to(dt); Rw=mlp.Right.weight.detach().to(dt)
        Dw=mlp.Down.weight.detach().to(dt); db=mlp.Down_bias.detach().to(dt)
        yreal=((X@L.T)*(X@Rw.T))@Dw.T+db
        U,_,_=torch.linalg.svd(Dw,full_matrices=False)
        C=U.T@Dw
        yfull=torch.zeros(X.shape[0],D,device=DEV,dtype=dt)
        for d0 in range(0,D,ch):
            Qc=torch.einsum('dk,kp,kq->dpq',C[d0:d0+ch],L,Rw)
            yfull[:,d0:d0+ch]=torch.einsum('bp,dpq,bq->bd',X,Qc,X)
            del Qc
        yo=yfull@U.T+db
        frob=float((yo-yreal).norm()/yreal.norm())
        mxs=float((yo-yreal).abs().max()/yreal.pow(2).mean().sqrt())
        res[tag]={'frob':frob,'max_over_rms':mxs}
        print(f'  {tag}: rel-Frobenius {frob:.2e}, max-abs/RMS {mxs:.2e}',flush=True)
    pa=res['FP64']['max_over_rms']<=1e-6
    pb=res['FP64']['frob']<=1e-9
    pc=0.5e-2<=res['FP32']['max_over_rms']<=5e-2
    out={'arms':res,'pred_a_fp64_max':bool(pa),'pred_b_fp64_frob':bool(pb),'pred_c_fp32_repro':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) FP64 max/RMS {res['FP64']['max_over_rms']:.2e} <= 1e-6: {'HELD' if pa else 'FAILED'}")
    print(f"(b) FP64 frob {res['FP64']['frob']:.2e} <= 1e-9: {'HELD' if pb else 'FAILED'}")
    print(f"(c) FP32 max/RMS {res['FP32']['max_over_rms']:.2e} in [0.5e-2, 5e-2]: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
