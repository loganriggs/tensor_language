"""CAUSAL VALIDATION OF THE TOP-K SCORE (rung 223): does |activation| x ||decoder|| track real removal damage?

CONVENTION (S2135): per-position dCE = CE(m16 band-ablated, all else real) - CE(real model); LOWER IS
BETTER. USER QUESTION (2026-08-31): the top-k criterion folds in the decoder norm (score = |u_i|*||D_i||),
but is that proxy CAUSALLY right? A unit's write is u_i D_i exactly at the module output, but downstream
propagation is nonlinear - big-norm writes may land in directions the network ignores, or cancel pairwise.
DESIGN: at m16 (all else real), rank all 4608 units by expected score E|u_i|*||D_i|| (256 real captured
inputs), split into 16 contiguous bands of 288, zero ONE BAND at a time (16 census passes), and correlate
band mean score with band causal damage.
REGISTERED PREDICTIONS (arms: BAND_00 (top score) ... BAND_15 (bottom)):
  (a) THE PROXY IS CAUSALLY ALIGNED: Spearman(band mean score, band census|dCE|) >= 0.8 over 16 bands.
  (b) DYNAMIC RANGE: damage(BAND_00) >= 10 x damage(BAND_15).
  (c) SANITY: all bands non-inert; adjacent inversions (damage rising as score falls) <= 3.
NULL: rho < 0.4 - the norm-times-activation score misranks causally; selection needs causal correction.
PRICE: probe, 16 census passes. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['m16_eigenbasis_v2_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: causal band validation of the top-k score'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'unit_causal_bands_results.json'
NB=16; BS=288

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
    mlp=m.transformer.h[16].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    dn=Dw.norm(dim=0)
    cap={}
    def pre(mo,i_): cap['x']=i_[0].detach().reshape(-1,D).float()[:256].clone()
    hp=mlp.register_forward_pre_hook(pre)
    bb=ROWS[:4,:257].to(DEV); idx=bb[:,:-1].contiguous()
    with torch.no_grad():
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    hp.remove()
    X=cap['x']
    es=((X@L.T)*(X@Rw.T)).abs().mean(0)*dn
    order=es.argsort(descending=True)
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
    dmg=[]; msc=[]
    for b in range(NB):
        ids=order[b*BS:(b+1)*BS]
        msk=torch.ones(4608,device=DEV); msk[ids]=0.0
        def hk(mo,i_,o_,msk=msk):
            x=i_[0].float()
            u=(x@L.T)*(x@Rw.T)
            return (((u*msk)@Dw.T)+db).to(o_.dtype)
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: band {b} inert')
        dmg.append(round(float(d.abs().mean()),5)); msc.append(round(float(es[ids].mean()),4))
        print(f'  BAND_{b:02d}: score {msc[-1]:.4f} | census|dCE| {dmg[-1]:.5f}',flush=True)
    def _rk(v):
        s_=sorted(range(len(v)),key=lambda i:v[i]); o=[0]*len(v)
        for j,i in enumerate(s_): o[i]=j
        return o
    xr=_rk(msc); yr=_rk(dmg); n=NB
    mx=sum(xr)/n; my=sum(yr)/n
    num=sum((a-mx)*(b-my) for a,b in zip(xr,yr))
    den=(sum((a-mx)**2 for a in xr)*sum((b-my)**2 for b in yr))**0.5
    rho=num/max(den,1e-9)
    inv=sum(1 for b in range(NB-1) if dmg[b+1]>dmg[b])
    pa=rho>=0.8; pb=dmg[0]>=10*dmg[-1]; pc=inv<=3
    out={'band_scores':msc,'band_damage':dmg,'spearman':round(rho,4),'inversions':inv,
         'ratio_top_bottom':round(dmg[0]/max(dmg[-1],1e-9),2),
         'pred_a_aligned':bool(pa),'pred_b_range':bool(pb),'pred_c_monotone':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'Spearman {rho:.3f}; top/bottom {out["ratio_top_bottom"]}; inversions {inv}')
    print(f"(a) Spearman {rho:.3f} >= 0.8: {'HELD' if pa else 'FAILED'}")
    print(f"(b) top {dmg[0]:.5f} >= 10 x bottom {dmg[-1]:.5f}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) inversions {inv} <= 3: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
