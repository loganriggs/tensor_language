"""SHARED MULTILINEAR BASIS AT m16 (rung 212): symmetric Tucker/HOSVD vs neurons vs eigenfeatures.

CONVENTION (S2135): per-position dCE = CE(m16 replaced, all else real) - CE(real model); LOWER IS BETTER.
MATH REVIEW 22:07 move #1. The eigenfeature basis (S2307, quarantined) is per-output-direction - no
sharing. The Tucker/HOSVD canonical form of the invariant tensor gives ONE shared input basis W
(top eigvecs of M = sum_d Q_d Q_d^T) and per-direction small cores G_d = W^T Q_d W:
out_d = (W^T x)^T G_d (W^T x). This is the "shared dictionary across components" object - multilinear
rank, not CP rank.
ARMS (single-site m16): HOSVD34 (W 1152x34 + cores 1152x34x34 = 1.37M values, matched to EIG's 1.33M)
and HOSVD128 (headroom, 18.9M). Anchors read from the v2 receipt: NEUR own 1.6259 / census 1.0146;
EIG census 0.6650 (quarantined but usable as a bar).
REGISTERED PREDICTIONS:
  (a) SHARED BASIS BEATS NEURONS at matched values: own-circuit HOSVD34/NEUR <= 0.9.
  (b) COMPETITIVE WITH EIGENFEATURES at matched values: census|dCE| HOSVD34 <= 1.1 x EIG (<= 0.7315).
  (c) MONOTONE + SANITY: census(HOSVD128) < census(HOSVD34), both non-inert, HOSVD128 <= 0.5.
NULL: per-direction eigenstructure is essential - the shared basis at matched values is no better than
neurons (own ratio >= 0.95). PRICE: probe (one module). Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['m16_eigenbasis_v2_results.json','circuits/BATTERY.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: m16 HOSVD shared basis'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m16_hosvd_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu(); NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    V2=json.load(open(PT+'m16_eigenbasis_v2_results.json'))
    NEUR_OWN=V2['arms']['NEUR']['own_mean']; NEUR_C=V2['arms']['NEUR']['aggabs']; EIG_C=V2['arms']['EIG']['aggabs']
    CINFO={}
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        CINFO[t]={'mask':mm,'top':v['mean_ablation']['top'][0]['component']}
    ownt=[t for t,v in CINFO.items() if v['top']=='m16']
    mlp=m.transformer.h[16].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
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
    Wfull=evec.flip(-1)                    # descending eigenvalue order, columns
    RINS=(34,128)
    Ws={r:Wfull[:,:r].contiguous() for r in RINS}
    G={r:torch.zeros(D,r,r,device=DEV) for r in RINS}
    for d0 in range(0,D,CH):
        Qc=torch.einsum('dk,kp,kq->dpq',C[d0:d0+CH],L,Rw)
        Qs=0.5*(Qc+Qc.transpose(1,2))
        for r in RINS:
            G[r][d0:d0+CH]=torch.einsum('pr,dpq,qs->drs',Ws[r],Qs,Ws[r])
        del Qc,Qs
    UT=U.T.contiguous()
    print(f'HOSVD built ({time.time()-T00:.0f}s)',flush=True)
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
    res={}
    for r in RINS:
        Wr=Ws[r]; Gr=G[r]
        def hk(mo,i_,o_,Wr=Wr,Gr=Gr):
            x=i_[0].float()
            a=x@Wr
            y=torch.einsum('btr,drs,bts->btd',a,Gr,a)
            return (y@UT+db).to(o_.dtype)
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: HOSVD{r} inert')
        own={t:round(float(d[CINFO[t]['mask']].abs().mean()),4) for t in ownt}
        res[f'HOSVD{r}']={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4),
                          'own_mean':round(sum(own.values())/max(len(own),1),4),'own_members':own,
                          'values':D*r+D*r*r}
        print(f"  HOSVD{r}: agg {res[f'HOSVD{r}']['agg']:+.4f} | census|dCE| {res[f'HOSVD{r}']['aggabs']:.4f} | own {res[f'HOSVD{r}']['own_mean']:.4f}",flush=True)
    ra=res['HOSVD34']['own_mean']/NEUR_OWN
    rb=res['HOSVD34']['aggabs']/EIG_C
    pa=ra<=0.9
    pb=rb<=1.1
    pc=(res['HOSVD128']['aggabs']<res['HOSVD34']['aggabs']) and res['HOSVD128']['aggabs']<=0.5
    out={'arms':res,'anchors':{'NEUR_own':NEUR_OWN,'NEUR_census':NEUR_C,'EIG_census':EIG_C},
         'ratio_own_vs_neur':round(ra,4),'ratio_census_vs_eig':round(rb,4),
         'pred_a_beats_neurons':bool(pa),'pred_b_matches_eig':bool(pb),'pred_c_monotone':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'ratios: own-vs-NEUR {ra:.3f}; census-vs-EIG {rb:.3f}')
    print(f"(a) HOSVD34 own <= 0.9 x NEUR (ratio {ra:.3f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) HOSVD34 census <= 1.1 x EIG (ratio {rb:.3f}): {'HELD' if pb else 'FAILED'}")
    print(f"(c) monotone + HOSVD128 <= 0.5 ({res['HOSVD128']['aggabs']:.4f} < {res['HOSVD34']['aggabs']:.4f}): {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
