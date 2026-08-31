"""USAGE CONCENTRATION BY BASIS (rung 219): per-token top-k inside the eigenfeature basis at m16.

CONVENTION (S2135): per-position dCE = CE(m16 replaced, all else real) - CE(real model); LOWER IS BETTER.
The two winning ideas meet: per-token selection beats any fixed subset (S2304), and the gauge-free
eigenbasis beats neurons at m16 for STATIC truncation (S2309, 0.655 census). Question: does the eigenbasis
also concentrate per-token USAGE? Matched design: dictionary size 4608 in both bases; each token activates
its top 1152.
ARMS (single-site m16; anchor NEUR-static-1152 census |dCE| 1.0146 from the v2 receipt):
  E4608  = static top-4608 eigenfeatures by |lambda), all on (baseline; NOT exact - 4608 of 1.3M pairs).
  EIGTK  = per-token top-1152 of those 4608 eigenfeatures (score |lambda| * a^2; u_d orthonormal).
  NEURTK = per-token top-1152 of the 4608 neurons (score |u_i| * ||D_i||).
REGISTERED PREDICTIONS:
  (a) PER-TOKEN BEATS STATIC IN THE NEURON BASIS MODULE-LOCALLY: census|dCE|(NEURTK) <= 0.5 x 1.0146.
  (b) THE EIGENBASIS CONCENTRATES USAGE: EIGTK <= 0.9 x NEURTK.
  (c) QUARTER USAGE NEARLY FREE IN THE EIGEN BASIS: EIGTK <= E4608 + 0.02; all arms non-inert.
NULL: usage concentration is basis-independent (EIGTK/NEURTK >= 0.95). PRICE: probe, 3 census passes.
Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['m16_eigenbasis_v2_results.json','circuits/BATTERY.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: usage concentration by basis at m16'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m16_eig_topk_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu(); NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    ANC=json.load(open(PT+'m16_eigenbasis_v2_results.json'))['arms']['NEUR']['aggabs']
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
    dn=Dw.norm(dim=0)
    U,_,_=torch.linalg.svd(Dw,full_matrices=False)
    C=U.T@Dw
    lam=torch.empty(D,D)
    EV=torch.empty(D,D,D,dtype=torch.float32)
    CH=64
    for d0 in range(0,D,CH):
        Qc=torch.einsum('dk,kp,kq->dpq',C[d0:d0+CH],L,Rw)
        Qs=0.5*(Qc+Qc.transpose(1,2))
        w,v=torch.linalg.eigh(Qs)
        lam[d0:d0+CH]=w.cpu(); EV[d0:d0+CH]=v.transpose(1,2).cpu()
        del Qc,Qs,w,v
    KD=4608; KA=1152
    topi=lam.abs().reshape(-1).topk(KD).indices
    dsel=topi//D; jsel=topi%D
    lsel=lam[dsel,jsel].to(DEV)
    Vsel=EV[dsel,jsel].to(DEV)
    Usel=U[:,dsel].T.contiguous()
    del EV
    print(f'eigen dictionary built ({time.time()-T00:.0f}s)',flush=True)
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
    def hk_e4608(mo,i_,o_):
        x=i_[0].float()
        a=x@Vsel.T
        return ((lsel*a*a)@Usel+db).to(o_.dtype)
    def hk_eigtk(mo,i_,o_):
        x=i_[0].float()
        a=x@Vsel.T
        s=(lsel.abs()*a*a)
        idx=s.topk(KA,dim=-1).indices
        msk=torch.zeros_like(a).scatter_(-1,idx,1.0)
        return ((lsel*a*a*msk)@Usel+db).to(o_.dtype)
    def hk_neurtk(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=(u.abs()*dn).topk(KA,dim=-1).indices
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return ((u*msk)@Dw.T+db).to(o_.dtype)
    res={}
    for nm,hk in (('E4608',hk_e4608),('EIGTK',hk_eigtk),('NEURTK',hk_neurtk)):
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {nm} inert')
        own={t:round(float(d[CINFO[t]['mask']].abs().mean()),4) for t in ownt}
        res[nm]={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4),
                 'own_mean':round(sum(own.values())/max(len(own),1),4)}
        print(f"  {nm}: agg {res[nm]['agg']:+.4f} | census|dCE| {res[nm]['aggabs']:.4f} | own {res[nm]['own_mean']:.4f}",flush=True)
    pa=res['NEURTK']['aggabs']<=0.5*ANC
    rb=res['EIGTK']['aggabs']/max(res['NEURTK']['aggabs'],1e-9)
    pb=rb<=0.9
    pc=res['EIGTK']['aggabs']<=res['E4608']['aggabs']+0.02
    out={'arms':res,'anchor_neur_static':ANC,'ratio_eigtk_neurtk':round(rb,4),
         'pred_a_pertoken_beats_static':bool(pa),'pred_b_eig_concentrates':bool(pb),'pred_c_quarter_free':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) NEURTK {res['NEURTK']['aggabs']:.4f} <= 0.5 x {ANC:.4f}: {'HELD' if pa else 'FAILED'}")
    print(f"(b) EIGTK/NEURTK {rb:.3f} <= 0.9: {'HELD' if pb else 'FAILED'}")
    print(f"(c) EIGTK {res['EIGTK']['aggabs']:.4f} <= E4608 {res['E4608']['aggabs']:.4f} + 0.02: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
