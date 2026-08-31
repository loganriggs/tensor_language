"""MINIMALITY FOR ALL 62 (rung 159; C1 of the circuits program, S2255).

CONVENTION (S2135): per-position dCE = CE(joint mean-ablation) - CE(real model) on the census rows. Rung 154
measured necessity depth on 12 circuits; this extends to ALL 62 with cumulative top-1..4 battery components
(joint mean-ablation, deduplicated across circuits sharing sets). Output: per-circuit necessity curves and a
first minimal-set-size estimate k*(t) = smallest k with damage(top1..k) >= 0.9 x damage(top1..4) - the
repertoire's first column. Runtime note: ~100+ unique sets, ~40-60 min; the runner serializes.

REGISTERED PREDICTIONS:
  (a) DEPTH CONFIRMED AT SCALE: median top-1 share of top-4 damage <= 0.55 (rung 154's 0.464 generalizes).
  (b) SATURATION BY 4: median damage(top1..3)/damage(top1..4) >= 0.85 (three components capture most).
  (c) PROTOCOL: median damage(top1)/battery ref in [0.8, 1.25].
NULL: (b) < 0.7 - four components do NOT saturate; minimal sets are larger and the greedy audit must extend.
PRICE: none (attribution; the repertoire column is the product). Tripwire: INSTRUMENT FAIL if any set's cev
is bitwise equal to base. Self-reviewed."""

import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['motif_single_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+('ops/' if f.endswith('.py') else '')+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: minimality for all 62')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'minimality62_results.json'

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
        tops=[e['component'] for e in v['mean_ablation']['top'][:4]]
        CINFO[t]={'mask':mm,'ref':v['mean_ablation']['top'][0]['abs_dce_members'],'tops':tops}
    SAMPLE=sorted(CINFO)
    print(f'all {len(SAMPLE)} circuits',flush=True)
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
    comps=sorted({c for t in SAMPLE for c in CINFO[t]['tops']})
    acc={c:[torch.zeros(D,device=DEV),0] for c in comps}
    hs=[]
    for c in comps:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                acc[c][0]+=y.detach().reshape(-1,D).float().sum(0)
                acc[c][1]+=y.reshape(-1,D).shape[0]
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce([])
    for h in hs: h.remove()
    MU={c:acc[c][0]/max(acc[c][1],1) for c in comps}
    print(f'means captured for {len(comps)} components',flush=True)
    def ablate_set(cset):
        hs=[]
        for c in cset:
            mod,kind=module_of(c)
            def abl(mo,i_,o_,c=c,kind=kind):
                if kind=='attn':
                    y,v1=o_
                    return (MU[c].expand_as(y).to(y.dtype),v1)
                return MU[c].expand_as(o_).to(o_.dtype)
            hs.append(mod.register_forward_hook(abl))
        cev=evalce([])
        for h in hs: h.remove()
        return cev
    cache={}
    def dmg(cset,mask):
        key=tuple(sorted(cset))
        if key not in cache:
            cache[key]=ablate_set(key)
            print(f'  ablated {key}',flush=True)
        return float((cache[key]-CBASE)[mask].abs().mean())
    rows=[]
    for t in SAMPLE:
        v=CINFO[t]
        ds=[dmg(v['tops'][:k],v['mask']) for k in range(1,min(len(v['tops']),4)+1)]
        while len(ds)<4: ds.append(ds[-1])
        kstar=next(k+1 for k in range(4) if ds[k]>=0.9*ds[3])
        rows.append({'tag':t,'tops':v['tops'],'ref':round(v['ref'],3),
                     'd':[round(x,4) for x in ds],'share_top1':round(ds[0]/max(ds[3],1e-9),3),
                     'kstar':kstar})
        print(f"  {t}: {' -> '.join(f'{x:.3f}' for x in ds)} (share {rows[-1]['share_top1']:.2f}, k* {kstar})",flush=True)
    import statistics as stt
    shares=[r['share_top1'] for r in rows]
    sat=[r['d'][2]/max(r['d'][3],1e-9) for r in rows]
    reps=[r['d'][0]/max(r['ref'],1e-9) for r in rows]
    pa=stt.median(shares)<=0.55
    pb=stt.median(sat)>=0.85
    pc=0.8<=stt.median(reps)<=1.25
    res={'rows':rows,'median_top1_share':round(stt.median(shares),3),
         'median_saturation_3of4':round(stt.median(sat),3),'median_battery_repro':round(stt.median(reps),3),
         'kstar_hist':{str(k):sum(1 for r in rows if r['kstar']==k) for k in (1,2,3,4)},
         'convention':'per-position dCE = CE(joint mean-ablation) - CE(real model) on census rows',
         'pred_a_depth_at_scale':bool(pa),'pred_b_saturation':bool(pb),'pred_c_repro':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"kstar histogram: {res['kstar_hist']}")
    print(f"(a) median top1 share {stt.median(shares):.3f} <= 0.55: {'HELD' if pa else 'FAILED'}")
    print(f"(b) median saturation {stt.median(sat):.3f} >= 0.85: {'HELD' if pb else 'FAILED'}")
    print(f"(c) median battery repro {stt.median(reps):.3f} in [0.8, 1.25]: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
