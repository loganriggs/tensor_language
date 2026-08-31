"""REMOVAL COLLATERAL MATRIX (rung 153): the missing cross-circuit removal table.

CONVENTION (S2135): per-position dCE = CE(knockout) - CE(real model) on the census rows; LOWER IS BETTER
for configs, HIGHER here means the knockout matters. The battery gives each circuit its top component and a
member-vs-offslice concentration; what it never measured is the full CROSS table: mean-ablate component c,
read EVERY circuit's member damage. Protocol: each distinct battery top component's output is replaced by
its global census-mean vector (position-independent); a protocol mismatch with the battery would show in
pred_c and is recorded as such, not as physics.

REGISTERED PREDICTIONS (M[c][t] = member mean|dCE| of circuit t under knockout of component c):
  (a) SELECTIVITY: median over components of [mean M over OWN circuits] / [median M over OTHER circuits]
      >= 3 (battery concentration ~4 suggests this survives at matrix grain).
  (b) SUBSTRATE SHARING: the a8 knockout damages >= 20 NON-a8 circuits above 0.25 x their battery ref.
  (c) BATTERY REPRODUCTION: median over circuits of M[top(t)][t] / battery ref(t) in [0.67, 1.5].
NULL: (a) selectivity < 2 at matrix grain (the concentration statistic flattered removal); (b) sharing < 10
(circuits are more separable than the rho-0.96 damage profiles implied). PRICE: none (attribution).
Tripwire: INSTRUMENT FAIL if any knockout cev is bitwise equal to base. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['frontier_biasdisp_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: removal collateral matrix')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'removal_matrix_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    CINFO={}
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        CINFO[t]={'mask':mm,'ref':v['mean_ablation']['top'][0]['abs_dce_members'],
                  'top':v['mean_ablation']['top'][0]['component']}
    comps=sorted({v['top'] for v in CINFO.values()})
    print(f'{len(CINFO)} circuits; {len(comps)} distinct top components: {comps}',flush=True)
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    def evalce(hooks):
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
    # component census means
    MU={}
    hs=[]; acc={}
    for c in comps:
        mod,kind=module_of(c)
        acc[c]=[torch.zeros(D,device=DEV),0]
        def mk(c=c,kind=kind):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                acc[c][0]+=y.detach().reshape(-1,D).float().sum(0)
                acc[c][1]+=y.reshape(-1,D).shape[0]
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce([])
    for h in hs: h.remove()
    for c in comps: MU[c]=(acc[c][0]/max(acc[c][1],1))
    print('component means captured',flush=True)
    Mrows={}
    for c in comps:
        mod,kind=module_of(c)
        def abl(mo,i_,o_,c=c,kind=kind):
            if kind=='attn':
                y,v1=o_
                return (MU[c].expand_as(y).to(y.dtype),v1)
            return MU[c].expand_as(o_).to(o_.dtype)
        h=mod.register_forward_hook(abl)
        cev=evalce([])
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6:
            raise SystemExit(f'INSTRUMENT FAIL: knockout {c} bitwise equal to base')
        Mrows[c]={t:round(float(d[info['mask']].abs().mean()),4) for t,info in CINFO.items()}
        own=[t for t,info in CINFO.items() if info['top']==c]
        print(f'  {c}: own-mean {sum(Mrows[c][t] for t in own)/max(len(own),1):.3f} '
              f'({len(own)} own circuits)',flush=True)
    import statistics as stt
    sel=[]
    for c in comps:
        own=[Mrows[c][t] for t,info in CINFO.items() if info['top']==c]
        oth=[Mrows[c][t] for t,info in CINFO.items() if info['top']!=c]
        if own and oth: sel.append((sum(own)/len(own))/max(stt.median(oth),1e-9))
    medsel=stt.median(sel)
    a8=sum(1 for t,info in CINFO.items()
           if info['top']!='a8' and Mrows.get('a8',{}).get(t,0)>0.25*info['ref'])
    reps=[Mrows[info['top']][t]/max(info['ref'],1e-9) for t,info in CINFO.items()]
    medrep=stt.median(reps)
    pa=medsel>=3
    pb=a8>=20
    pc=0.67<=medrep<=1.5
    res={'matrix':Mrows,'median_selectivity':round(medsel,3),'a8_collateral_count':a8,
         'median_battery_repro_ratio':round(medrep,3),
         'convention':'per-position dCE = CE(knockout) - CE(real model) on census rows',
         'pred_a_selective':bool(pa),'pred_b_substrate_sharing':bool(pb),'pred_c_battery_repro':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'median selectivity {medsel:.3f}; a8 collateral {a8}; battery repro ratio {medrep:.3f}')
    print(f"(a) selectivity {medsel:.3f} >= 3: {'HELD' if pa else 'FAILED'}")
    print(f"(b) a8 collateral {a8} >= 20: {'HELD' if pb else 'FAILED'}")
    print(f"(c) repro ratio {medrep:.3f} in [0.67, 1.5]: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
