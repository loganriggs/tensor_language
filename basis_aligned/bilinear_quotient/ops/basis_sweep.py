"""BASIS-LAW DISAMBIGUATION SWEEP (rung 221): four fresh modules; threshold vs depth.

CONVENTION (S2135): per-position dCE = CE(module replaced, all else real) - CE(real model); LOWER IS
BETTER. S2318: two-regime threshold survives on 5 modules but difficulty and depth are confounded (eigen
wins only at late blocks 16/17). This run measures difficulty (NEUR static top-1152 census |dCE|) and the
EIG/NEUR census ratio IN-RUN at m2, m7, m9, m11 - early-to-mid blocks - giving nine points total.
REGISTERED PREDICTIONS (points = 5 receipts + 4 new; diff = NEUR census |dCE|; ratio = EIG/NEUR census):
  (a) EXACTNESS at all four new modules: rel-Frobenius <= 1e-3 AND max/RMS <= 5e-2.
  (b) TWO-REGIME CONSISTENCY over all nine: diff >= 0.5 -> ratio <= 1.0; diff <= 0.2 -> ratio >= 0.95;
      the band (0.2, 0.5) unconstrained. No violations.
  (c) NINE-POINT Spearman(diff, ratio) <= -0.5.
NULL: depth is the true variable - Spearman(block, ratio) <= -0.8 while (c) fails; an early hard module
staying neuron-favoring would kill concentration. PRICE: probe, 8 census passes + 4 eigen builds.
Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['m13_eigenbasis_results.json','m14_eigenbasis_results.json','m5_eigenbasis_results.json',
           'm16_eigenbasis_v2_results.json','m17_eigenbasis_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: basis-law disambiguation sweep'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'basis_sweep_results.json'
MODS=[2,7,9,11]

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
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
    # capture inputs for all four modules in one pass
    cap={}
    hps=[]
    for li in MODS:
        def pre(mo,i_,li=li): cap[li]=i_[0].detach().reshape(-1,D).float()[:256].clone()
        hps.append(m.transformer.h[li].mlp.register_forward_pre_hook(pre))
    bb=ROWS[:4,:257].to(DEV); idx=bb[:,:-1].contiguous()
    with torch.no_grad():
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hps: h.remove()
    lam=torch.empty(D,D)
    EV=torch.empty(D,D,D,dtype=torch.float32)
    CH=64; K=1152
    res={}; exact_ok=True
    for li in MODS:
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
        X=cap[li]
        yreal=((X@L.T)*(X@Rw.T))@Dw.T+db
        U,_,_=torch.linalg.svd(Dw,full_matrices=False)
        C=U.T@Dw
        yfull=torch.zeros(X.shape[0],D,device=DEV)
        for d0 in range(0,D,CH):
            Qc=torch.einsum('dk,kp,kq->dpq',C[d0:d0+CH],L,Rw)
            yfull[:,d0:d0+CH]=torch.einsum('bp,dpq,bq->bd',X,Qc,X)
            Qs=0.5*(Qc+Qc.transpose(1,2))
            w,v=torch.linalg.eigh(Qs)
            lam[d0:d0+CH]=w.cpu(); EV[d0:d0+CH]=v.transpose(1,2).cpu()
            del Qc,Qs,w,v
        yo=yfull@U.T+db
        frob=float((yo-yreal).norm()/yreal.norm())
        mxs=float((yo-yreal).abs().max()/yreal.pow(2).mean().sqrt())
        if not (frob<=1e-3 and mxs<=5e-2): exact_ok=False
        topi=lam.abs().reshape(-1).topk(K).indices
        dsel=topi//D; jsel=topi%D
        lsel=lam[dsel,jsel].to(DEV); Vsel=EV[dsel,jsel].to(DEV)
        Usel=U[:,dsel].T.contiguous()
        kp=(Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)).argsort(descending=True)[:K]
        Ln=L[kp].contiguous(); Rn=Rw[kp].contiguous(); Dn=Dw[:,kp].contiguous()
        arms={}
        def hk_n(mo,i_,o_,Ln=Ln,Rn=Rn,Dn=Dn,db=db):
            x=i_[0].float()
            return (((x@Ln.T)*(x@Rn.T))@Dn.T+db).to(o_.dtype)
        def hk_e(mo,i_,o_,Vsel=Vsel,lsel=lsel,Usel=Usel,db=db):
            x=i_[0].float()
            a=x@Vsel.T
            return ((lsel*a*a)@Usel+db).to(o_.dtype)
        for nm,hk in (('NEUR',hk_n),('EIG',hk_e)):
            h=mlp.register_forward_hook(hk)
            cev=evalce()
            h.remove()
            d=cev-CBASE
            if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: m{li} {nm} inert')
            arms[nm]=round(float(d.abs().mean()),4)
        res[f'm{li}']={'diff':arms['NEUR'],'ratio':round(arms['EIG']/max(arms['NEUR'],1e-9),4),
                       'frob':frob,'max_over_rms':mxs,'block':li}
        print(f"  m{li}: diff {arms['NEUR']:.4f} | EIG {arms['EIG']:.4f} | ratio {res[f'm{li}']['ratio']:.3f} | frob {frob:.1e}",flush=True)
    RECS={'m13':('m13_eigenbasis_results.json',13),'m14':('m14_eigenbasis_results.json',14),
          'm5':('m5_eigenbasis_results.json',5),'m16':('m16_eigenbasis_v2_results.json',16),
          'm17':('m17_eigenbasis_results.json',17)}
    pts=[]
    for k,(f,blk) in RECS.items():
        r=json.load(open(PT+f))
        pts.append((r['arms']['NEUR']['aggabs'],r['ratio_census'],blk))
    for k,v in res.items(): pts.append((v['diff'],v['ratio'],v['block']))
    viol=[p for p in pts if (p[0]>=0.5 and p[1]>1.0) or (p[0]<=0.2 and p[1]<0.95)]
    def _rk(v):
        s_=sorted(range(len(v)),key=lambda i:v[i]); o=[0]*len(v)
        for j,i in enumerate(s_): o[i]=j
        return o
    def spear(xs,ys):
        xr=_rk(xs); yr=_rk(ys); n=len(xs)
        mx=sum(xr)/n; my=sum(yr)/n
        num=sum((a-mx)*(b-my) for a,b in zip(xr,yr))
        den=(sum((a-mx)**2 for a in xr)*sum((b-my)**2 for b in yr))**0.5
        return num/max(den,1e-9)
    rho_d=spear([p[0] for p in pts],[p[1] for p in pts])
    rho_b=spear([p[2] for p in pts],[p[1] for p in pts])
    pa=exact_ok; pb=len(viol)==0; pc=rho_d<=-0.5
    out={'new':res,'points':pts,'violations':viol,'spearman_diff':round(rho_d,4),'spearman_block':round(rho_b,4),
         'pred_a_exact':bool(pa),'pred_b_two_regime':bool(pb),'pred_c_corr':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'nine points: {pts}')
    print(f'Spearman diff {rho_d:.3f} | block {rho_b:.3f}; violations {len(viol)}')
    print(f"(a) exactness all four: {'HELD' if pa else 'FAILED'}")
    print(f"(b) two-regime, no violations: {'HELD' if pb else 'FAILED'}")
    print(f"(c) 9-pt Spearman(diff) {rho_d:.3f} <= -0.5: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
