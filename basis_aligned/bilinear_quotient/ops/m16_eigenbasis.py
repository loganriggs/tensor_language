"""GAUGE-FREE COMPONENT BASIS AT m16 (rung 207): eigenfeatures of the invariant bilinear tensor vs neurons.

CONVENTION (S2135): per-position dCE = CE(m16 replaced, all else real) - CE(real model); LOWER IS BETTER.
USER DIRECTIVE (2026-08-31): the 4608-unit h-dimension is gauge - the neuron dictionary is one arbitrary
rank-1 decomposition of the layer's invariant third-order tensor. Principled alternative, NO fitting:
output basis = left singular vectors U of Down; per direction d, interaction matrix
Q_d = sym(L^T diag((U^T Down)_d) R); its eigenpairs (lambda, v) are gauge-free components:
out = sum_d u_d sum_j lambda_dj (v_dj . x)^2 + b, EXACT at full spectrum (checked in-run on real inputs).
ARMS (single-site m16 - 6 own circuits, battery ref 1.353 - everything else real):
  NEUR = fixed top-1152/4608 neurons by norm score (3.98M stored values).
  EIG  = top-1152 (d,j) eigenfeatures by |lambda| (1.33M values - 3x cheaper at matched component count).
REGISTERED PREDICTIONS:
  (a) CIRCUITS SEPARATE IN THE GAUGE-FREE BASIS: mean own-circuit member |dCE|: EIG <= 0.5 x NEUR.
  (b) GLOBAL: census mean |dCE|: EIG <= 0.7 x NEUR.
  (c) EXACTNESS GAUGE CHECK: full-spectrum reconstruction max rel err <= 1e-3 on 256 captured m16 inputs;
      both arms census non-inert.
NULL: the neuron basis is already aligned (both ratios >= 0.9) - gauge concern immaterial at m16.
PRICE: probe (one module, no config claim). Distinct from the CLOSED metric-constructed bases: this is an
exact re-decomposition of the same tensor; damage enters only at truncation. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['circuits/BATTERY.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: m16 eigenbasis vs neuron basis'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m16_eigenbasis_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu(); NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    CINFO={}
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        CINFO[t]={'mask':mm,'ref':v['mean_ablation']['top'][0]['abs_dce_members'],
                  'top':v['mean_ablation']['top'][0]['component']}
    ownt=[t for t,v in CINFO.items() if v['top']=='m16']
    print(f'{len(CINFO)} circuits; m16 owns {len(ownt)}: {ownt}',flush=True)
    mlp=m.transformer.h[16].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    # capture 256 real m16 inputs from the first census batch
    cap={}
    def pre(mo,i_): cap['x']=i_[0].detach().reshape(-1,D).float()[:256].clone()
    hp=mlp.register_forward_pre_hook(pre)
    bb=ROWS[:4,:257].to(DEV); idx=bb[:,:-1].contiguous()
    with torch.no_grad():
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    hp.remove()
    X=cap['x']
    yreal=((X@L.T)*(X@Rw.T))@Dw.T+db
    # gauge-free build: SVD output basis + per-direction interaction eigenpairs
    U,_,_=torch.linalg.svd(Dw,full_matrices=False)      # (1152,1152)
    C=U.T@Dw                                            # (1152,4608)
    lam=torch.empty(D,D)
    EV=torch.empty(D,D,D,dtype=torch.float16)           # [d,j]=v_dj on CPU (~3GB)
    yfull=torch.zeros(X.shape[0],D,device=DEV)
    CH=64
    for d0 in range(0,D,CH):
        c=C[d0:d0+CH]
        Qc=torch.einsum('dk,kp,kq->dpq',c,L,Rw)
        yfull[:,d0:d0+CH]=torch.einsum('bp,dpq,bq->bd',X,Qc,X)
        Qs=0.5*(Qc+Qc.transpose(1,2))
        w,v=torch.linalg.eigh(Qs)
        lam[d0:d0+CH]=w.cpu(); EV[d0:d0+CH]=v.transpose(1,2).half().cpu()
        del Qc,Qs,w,v
    yfullo=yfull@U.T+db
    relerr=float(((yfullo-yreal).abs()/(yreal.abs()+1e-3)).max())
    print(f'full-spectrum reconstruction max rel err {relerr:.2e} ({time.time()-T00:.0f}s)',flush=True)
    K=1152
    topi=lam.abs().reshape(-1).topk(K).indices
    dsel=topi//D; jsel=topi%D
    lsel=lam[dsel,jsel].to(DEV)
    Vsel=EV[dsel,jsel].float().to(DEV)                  # (K,1152)
    Usel=U[:,dsel].T.contiguous()                       # (K,1152)
    kp=(Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)).argsort(descending=True)[:K]
    Ln=L[kp].contiguous(); Rn=Rw[kp].contiguous(); Dn=Dw[:,kp].contiguous()
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            b=ROWS[i:i+4,:257].to(DEV)
            ix=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    def hook_n(mo,i_,o_):
        x=i_[0].float()
        return (((x@Ln.T)*(x@Rn.T))@Dn.T+db).to(o_.dtype)
    def hook_e(mo,i_,o_):
        x=i_[0].float()
        a=x@Vsel.T
        return ((lsel*a*a)@Usel+db).to(o_.dtype)
    res={}
    for nm,hk in (('NEUR',hook_n),('EIG',hook_e)):
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {nm} arm inert')
        own={t:round(float(d[CINFO[t]['mask']].abs().mean()),4) for t in ownt}
        res[nm]={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4),
                 'own_members':own,'own_mean':round(sum(own.values())/max(len(own),1),4)}
        print(f"  {nm}: agg {res[nm]['agg']:+.4f} | census|dCE| {res[nm]['aggabs']:.4f} | own-circuit mean {res[nm]['own_mean']:.4f}",flush=True)
    ra=res['EIG']['own_mean']/max(res['NEUR']['own_mean'],1e-9)
    rb=res['EIG']['aggabs']/max(res['NEUR']['aggabs'],1e-9)
    pa=ra<=0.5; pb=rb<=0.7; pc=relerr<=1e-3
    out={'arms':res,'ratio_own':round(ra,4),'ratio_census':round(rb,4),'full_relerr':relerr,
         'values':{'NEUR':1152*3456,'EIG':1152*1153},
         'convention':'dCE = CE(m16 replaced, all else real) - CE(real); lower is better',
         'pred_a_circuits':bool(pa),'pred_b_global':bool(pb),'pred_c_exactness':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'ratios: own-circuit {ra:.3f}, census {rb:.3f}; relerr {relerr:.2e}')
    print(f"(a) EIG own-mean <= 0.5 x NEUR (ratio {ra:.3f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) EIG census <= 0.7 x NEUR (ratio {rb:.3f}): {'HELD' if pb else 'FAILED'}")
    print(f"(c) full-spectrum relerr {relerr:.2e} <= 1e-3: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
