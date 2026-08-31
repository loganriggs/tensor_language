"""INTERCHANGE INSTRUMENT (rung 160; C2 of the circuits program, S2255): the foundation DAS needs.

CONVENTION (S2135): per-position dCE = CE(interchange) - CE(real model) on the census rows. Protocol,
stated: at circuit t's top component, MEMBER positions receive the component output of a seeded random OTHER
member position (activation swap within the member population, captured from a clean pass); non-members
untouched. This is the full-rank counterfactual patch that DAS-lite (rung 161) will restrict to low-rank
subspaces; the battery's interchange refs are the comparison (its protocol may differ - pred_c is loose and
a failure is recorded as protocol divergence, not physics). Ten circuits sampled (largest member counts,
distinct tops).

REGISTERED PREDICTIONS:
  (a) INTERCHANGE BITES: median member |dCE| >= 0.5 x the battery interchange ref.
  (b) SELECTIVE: median member/offslice damage ratio >= 3.
  (c) PROTOCOL: median ratio to battery interchange ref in [0.5, 2.0].
NULL: (a) < 0.2 - the swap protocol is too weak an intervention; DAS needs the battery's own machinery.
PRICE: none (instrument). Tripwire: INSTRUMENT FAIL if any patched cev is bitwise equal to base.
Self-reviewed."""

import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['a16_single_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: interchange instrument')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'interchange_inst_results.json'

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
    for t,v in BATC.items():
        if t in CINFO:
            CINFO[t]['iref']=v['interchange']['top'][0]['abs_dce_members']
            CINFO[t]['itop']=v['interchange']['top'][0]['component']
    SAMPLE=sorted(CINFO,key=lambda t:-int(CINFO[t]['mask'].sum()))
    seen=set(); PICK=[]
    for t in SAMPLE:
        if CINFO[t]['itop'] in seen: continue
        seen.add(CINFO[t]['itop']); PICK.append(t)
        if len(PICK)==10: break
    print(f'picked {PICK}',flush=True)
    comps=sorted({CINFO[t]['itop'] for t in PICK})
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
    CAPOUT={c:[] for c in comps}
    hs=[]
    for c in comps:
        mod,kind=module_of(c)
        def mkc(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                CAPOUT[c].append(y.detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mkc()))
    _=evalce([])
    for h in hs: h.remove()
    CAPOUT={c:torch.cat(v) for c,v in CAPOUT.items()}
    print('outputs captured',flush=True)
    g9=torch.Generator().manual_seed(77)
    import statistics as stt
    rows9=[]
    for t in PICK:
        c=CINFO[t]['itop']; mod,kind=module_of(c)
        mi=CINFO[t]['mask'].nonzero().squeeze(1)
        perm=mi[torch.randperm(mi.numel(),generator=g9)]
        SRC=torch.zeros(NFLAT,dtype=torch.long); SRC[mi]=perm
        ISM=CINFO[t]['mask']
        cur9={'bi':0}
        def ih(mo,i_,o_,c=c,kind=kind):
            y=o_[0] if isinstance(o_,tuple) else o_
            B9=y.reshape(-1,D).shape[0]
            lo=cur9['bi']*1024
            sel=ISM[lo:lo+B9]
            yn=y.reshape(-1,D).clone()
            if sel.any():
                yn[sel]=CAPOUT[c][SRC[lo:lo+B9][sel]].to(y.device).float().to(yn.dtype)
            cur9['bi']+=1
            yn=yn.view_as(y)
            if kind=='attn': return (yn,o_[1])
            return yn
        hh=mod.register_forward_hook(ih)
        cur9['bi']=0
        cev=evalce([])
        hh.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6:
            raise SystemExit(f'INSTRUMENT FAIL: interchange {t} bitwise equal to base')
        md=float(d[ISM].abs().mean()); od=float(d[~ISM].abs().mean())
        rows9.append({'tag':t,'comp':c,'member_absdce':round(md,4),'offslice_absdce':round(od,4),
                      'iref':round(CINFO[t]['iref'],3),'ratio_to_iref':round(md/max(CINFO[t]['iref'],1e-9),3),
                      'selectivity':round(md/max(od,1e-9),3)})
        print(f"  {t} @ {c}: member {md:.3f} (iref {CINFO[t]['iref']:.3f}), selectivity {rows9[-1]['selectivity']:.1f}",flush=True)
    medm=stt.median([r['member_absdce'] for r in rows9])
    medri=stt.median([r['ratio_to_iref'] for r in rows9])
    medsel9=stt.median([r['selectivity'] for r in rows9])
    pa=medri>=0.5
    pb=medsel9>=3
    pc=0.5<=medri<=2.0
    res={'rows':rows9,'median_member':round(medm,4),'median_ratio_to_iref':round(medri,3),
         'median_selectivity':round(medsel9,3),
         'convention':'per-position dCE = CE(member-permutation interchange at top component) - CE(real model)',
         'pred_a_bites':bool(pa),'pred_b_selective':bool(pb),'pred_c_protocol':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'median member {medm:.3f}; ratio-to-iref {medri:.3f}; selectivity {medsel9:.1f}')
    print(f"(a) ratio {medri:.3f} >= 0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) selectivity {medsel9:.1f} >= 3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) ratio in [0.5, 2.0]: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')
    return
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
