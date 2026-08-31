"""BASIS COMPETITION AT m13 (rung 216): is m16 the outlier?

CONVENTION (S2135): per-position dCE = CE(m13 replaced, all else real) - CE(real model); LOWER IS BETTER.
Score so far: eigenfeatures beat neurons at m16 (0.770 own, S2309); neurons beat eigenfeatures at m14
(1.264, S2313). Identical instrument at m13 (4 own circuits). Working hypothesis: eigen wins only at
high-damage concentrated modules like m16 - so here the prediction is NEURON HOLDS.
ARMS: NEUR = fixed top-1152/4608 neurons at m13; EIG = top-1152 eigenfeatures at m13.
REGISTERED PREDICTIONS:
  (a) EXACTNESS: rel-Frobenius <= 1e-3 AND max/RMS <= 5e-2.
  (b) NEURON HOLDS: own-circuit ratio EIG/NEUR >= 0.95.
  (c) census ratio >= 0.9.
NULL: eigen wins again (own ratio <= 0.9) - the competition is a genuine split needing a predictor, not
an m16 anomaly. PRICE: probe (one module). Self-reviewed."""



import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['circuits/BATTERY.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: m13 basis competition'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m13_eigenbasis_results.json'

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
    ownt=[t for t,v in CINFO.items() if v['top']=='m13']
    print(f'{len(CINFO)} circuits; m13 owns {len(ownt)}: {ownt}',flush=True)
    mlp=m.transformer.h[13].mlp
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
    EV=torch.empty(D,D,D,dtype=torch.float32)           # [d,j]=v_dj on CPU (~6GB, full fp32)
    yfull=torch.zeros(X.shape[0],D,device=DEV)
    CH=64
    for d0 in range(0,D,CH):
        c=C[d0:d0+CH]
        Qc=torch.einsum('dk,kp,kq->dpq',c,L,Rw)
        yfull[:,d0:d0+CH]=torch.einsum('bp,dpq,bq->bd',X,Qc,X)
        Qs=0.5*(Qc+Qc.transpose(1,2))
        w,v=torch.linalg.eigh(Qs)
        lam[d0:d0+CH]=w.cpu(); EV[d0:d0+CH]=v.transpose(1,2).cpu()
        del Qc,Qs,w,v
    yfullo=yfull@U.T+db
    frob=float((yfullo-yreal).norm()/yreal.norm())
    mxs=float((yfullo-yreal).abs().max()/yreal.pow(2).mean().sqrt())
    print(f'full-spectrum reconstruction: rel-Frobenius {frob:.2e}, max-abs/RMS {mxs:.2e} ({time.time()-T00:.0f}s)',flush=True)
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
    pa=(frob<=1e-3 and mxs<=5e-2); pb=ra>=0.95; pc=rb>=0.9
    out={'arms':res,'ratio_own':round(ra,4),'ratio_census':round(rb,4),'frob':frob,'max_over_rms':mxs,
         'values':{'NEUR':1152*3456,'EIG':1152*1153},
         'convention':'dCE = CE(m16 replaced, all else real) - CE(real); lower is better',
         'pred_a_circuits':bool(pa),'pred_b_global':bool(pb),'pred_c_exactness':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'ratios: own-circuit {ra:.3f}, census {rb:.3f}; frob {frob:.2e}, max/RMS {mxs:.2e}')
    print(f"(a) EXACTNESS frob {frob:.2e} <= 1e-3 and max/RMS {mxs:.2e} <= 5e-2: {'HELD' if pa else 'FAILED'}")
    print(f"(b) NEURON HOLDS: own ratio {ra:.3f} >= 0.95: {'HELD' if pb else 'FAILED'}")
    print(f"(c) census ratio {rb:.3f} >= 0.9: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
