"""CARRIER-PROJECTION REMOVAL (rung 168): the operator's removal property with the new tool.

CONVENTION (S2135): per-position dCE = CE(intervention) - CE(real model) on the census rows. Deployable,
position-independent surgery: a8's output y -> y - P P^T (y - mu) at ALL positions (project the deviation
out of the a8-family pca32 carrier; mu = a8's census mean). Baseline arm: a8 mean-ablation (the S2248
protocol, rerun in-script for aggregate + repro). Scored on the operator's removal criteria: does the
carrier hold the family function, and is its removal more surgical than ablation?

REGISTERED PREDICTIONS (family = the 16 a8-topped circuits; others = the rest):
  (a) CARRIER HOLDS THE FUNCTION: family-median member damage(projection) >= 0.5 x family-median(mean-abl).
  (b) MORE SURGICAL: selectivity(projection) >= 3 x selectivity(mean-abl), selectivity = family-median
      member damage / others-median member damage.
  (c) CHEAPER COLLATERAL: census aggregate(projection) <= 0.5 x aggregate(mean-abl) AND mean-abl arm
      reproduces the S2248 row (median ratio in [0.8, 1.25]).
NULL: projection removal is as blunt as ablation (selectivity ratio < 1.5) - the carrier does not separate
the family from the substrate. PRICE: the removal operator = one 32 x 1152 basis + mean (38,016 values).
Tripwire: INSTRUMENT FAIL on inert arms. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['carrier_necessity_results.json','removal_matrix_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: carrier-projection removal')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'carrier_removal_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    R153=json.load(open(PT+'removal_matrix_results.json'))['matrix']['a8']
    CINFO={}
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        CINFO[t]={'mask':mm,'top':v['mean_ablation']['top'][0]['component']}
    fam=[t for t,v in CINFO.items() if v['top']=='a8']
    oth=[t for t,v in CINFO.items() if v['top']!='a8']
    mod=m.transformer.h[8].attn
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    CAP=[]
    hh=mod.register_forward_hook(lambda mo,i_,o_: CAP.append(o_[0].detach().reshape(-1,D).to(torch.float16).cpu()))
    _=evalce()
    hh.remove()
    Y=torch.cat(CAP)
    MU=Y.float().mean(0).to(DEV)
    pool=torch.cat([CINFO[t]['mask'].nonzero().squeeze(1) for t in fam])
    Ym=Y[pool].float()
    _,_,Vh=torch.linalg.svd((Ym-Ym.mean(0))[:20000].to(DEV),full_matrices=False)
    P=Vh[:32].T.contiguous(); PPT=(P@P.T).to(DEV)
    def arm(mode):
        def h(mo,i_,o_):
            y,v1=o_
            yf=y.reshape(-1,D).float()
            if mode=='mean':
                yn=MU.expand_as(yf)
            else:
                yn=yf-(yf-MU)@PPT
            return (yn.view_as(y).to(y.dtype),v1)
        hh=mod.register_forward_hook(h)
        cev=evalce()
        hh.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {mode} inert')
        return d
    import statistics as stt
    res={}
    for mode in ('mean','proj'):
        d=arm(mode)
        famv=stt.median([float(d[CINFO[t]['mask']].abs().mean()) for t in fam])
        othv=stt.median([float(d[CINFO[t]['mask']].abs().mean()) for t in oth])
        res[mode]={'family_median':round(famv,4),'others_median':round(othv,4),
                   'selectivity':round(famv/max(othv,1e-9),3),'agg':round(float(d.mean()),4),
                   'per_circuit_family':{t:round(float(d[CINFO[t]['mask']].abs().mean()),4) for t in fam}}
        print(f"  {mode}: family {famv:.3f} others {othv:.3f} sel {res[mode]['selectivity']:.2f} agg {res[mode]['agg']:+.4f}",flush=True)
    reps=[res['mean']['per_circuit_family'][t]/max(R153[t],1e-9) for t in fam if t in R153]
    medrep=stt.median(reps)
    pa=res['proj']['family_median']>=0.5*res['mean']['family_median']
    pb=res['proj']['selectivity']>=3*res['mean']['selectivity']
    pc=(res['proj']['agg']<=0.5*res['mean']['agg']) and 0.8<=medrep<=1.25
    out={'arms':res,'median_153_repro':round(medrep,3),
         'convention':'per-position dCE = CE(intervention) - CE(real model); proj = y - PPT(y-mu) everywhere',
         'pred_a_carrier_holds':bool(pa),'pred_b_surgical':bool(pb),'pred_c_cheap_and_repro':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) proj family {res['proj']['family_median']:.3f} >= 0.5 x mean ({0.5*res['mean']['family_median']:.3f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) sel ratio {res['proj']['selectivity']/max(res['mean']['selectivity'],1e-9):.2f} >= 3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) agg {res['proj']['agg']:+.4f} <= 0.5 x {res['mean']['agg']:+.4f} and repro {medrep:.3f}: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
