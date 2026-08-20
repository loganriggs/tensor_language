"""LEAF INPUT DECOMP -- the swarm's generic MECHANISM tool
(SOP v3 step 3M). Usage:  python leaf_input_decomp.py <tag>

For each component named by a leaf's probe bundle (its machinery),
decompose the residual ENTERING that component into exact writer
contributions (wte + every component write below it; the block
lambda mix and the rms scale are per-position scalars, so the
split is exact), then contrast MEMBER positions against OFF-SLICE
positions in the same rows. What comes out is a mechanism lead in
named parts: "this leaf's machinery fires where writer X's
contribution to component C's input is enriched".

Computed bars (printed, so the agent cannot certify junk):
  ENRICHED: top writer's member-vs-offslice share ratio >= 1.3
  NOT-NULL: that ratio exceeds the row-matched random-position
            null's top ratio
Writes leaf_mech/<tag>.json with the per-component tables; the
agent quotes these numbers in the record's mechanism field.
"""
import json, os, sys, time, ast, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
MAXROWS=24

@torch.no_grad()
def main(tag):
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(tag)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    keys=[]
    for p in probes:
        k=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
        if k not in keys: keys.append(k)
    mem=lf['member']; sl=lf['slice']
    NF=cl.nflat()
    memm=torch.zeros(NF,dtype=torch.bool); memm[mem]=True
    slm=torch.zeros(NF,dtype=torch.bool); slm[sl]=True
    g=torch.Generator().manual_seed(5)
    rr=(mem//256).unique()
    if len(rr)>MAXROWS:
        rr=rr[torch.randperm(len(rr),generator=g)[:MAXROWS]] \
            .sort().values
    rows=cl.rows()
    print(f'{tag}: machinery {keys} | {len(rr)} rows',flush=True)
    tables={}
    for key in keys:
        li=int(key[1:])
        WR=['wte']+[f'{k}{l}' for l in range(li)
                    for k in ('a','m')]
        if key[0]=='m': WR.append(f'a{li}')   # attn of same block
        share={'mem':{w:0.0 for w in WR},'off':{w:0.0 for w in WR},
               'null':{w:0.0 for w in WR}}
        cnt={'mem':0,'off':0,'null':0}
        for i in range(0,len(rr),4):
            rid=rr[i:i+4]
            bb=rows[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=len(rid)
            outs={}; hs=[]
            for lj in range(li+1):
                for kind,mod in (('a',m.transformer.h[lj].attn),
                                 ('m',m.transformer.h[lj].mlp)):
                    def mk(k9=f'{kind}{lj}'):
                        def h(mo,i_,o_):
                            y=o_[0] if isinstance(o_,tuple) else o_
                            outs[k9]=y.detach().float()
                        return h
                    hs.append(mod.register_forward_hook(mk()))
            E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            lam=m.transformer.h[li].lambdas.detach().float()
            parts={w:((lam[0]+lam[1])*E if w=='wte'
                      else lam[0]*outs[w]) for w in WR
                   if w!=f'a{li}'}
            if f'a{li}' in WR: parts[f'a{li}']=outs[f'a{li}']
            tot=sum(parts.values())
            tn=(tot*tot).sum(-1).clamp_min(1e-9)
            frac={w:((p*tot).sum(-1)/tn) for w,p in parts.items()}
            gi=(rid[:,None].to(DEV)*256
                +torch.arange(T,device=DEV)[None,:])
            mmask=memm.to(DEV)[gi]
            omask=(~slm.to(DEV)[gi])
            npick=torch.rand(mmask.shape,generator=None,
                             device=DEV)<(mmask.float().mean()+1e-6)
            for nm,msk in (('mem',mmask),('off',omask),
                           ('null',npick)):
                c=int(msk.sum())
                if c==0: continue
                cnt[nm]+=c
                for w,fv in frac.items():
                    share[nm][w]+=float(fv[msk].sum())
        tbl={}
        for w in share['mem']:
            mv=share['mem'][w]/max(cnt['mem'],1)
            ov=share['off'][w]/max(cnt['off'],1)
            nv=share['null'][w]/max(cnt['null'],1)
            tbl[w]={'member':round(mv,4),'offslice':round(ov,4),
                    'null':round(nv,4),
                    'ratio':round(mv/ov if abs(ov)>1e-4
                                  else float('nan'),3)}
        top=sorted(tbl.items(),
                   key=lambda kv:-(kv[1]['ratio']
                                   if kv[1]['ratio']==kv[1]['ratio']
                                   else 0))[:5]
        nullratio=max((v['null']/v['offslice']
                       if abs(v['offslice'])>1e-4 else 0)
                      for v in tbl.values())
        enriched=top[0][1]['ratio'] if top else 0
        tables[key]={'writers':tbl,'top':top,
                     'top_ratio':enriched,
                     'null_top_ratio':round(nullratio,3),
                     'ENRICHED':bool(enriched>=1.3),
                     'BEATS_NULL':bool(enriched>nullratio)}
        print(f"  {key}: top writers {[(w,v['ratio']) for w,v in top]}"
              f" | ENRICHED={tables[key]['ENRICHED']}"
              f" BEATS_NULL={tables[key]['BEATS_NULL']}",flush=True)
    os.makedirs(PT+'leaf_mech',exist_ok=True)
    out={'tag':tag,'machinery':keys,'tables':tables,
         'runtime_s':time.time()-t0}
    json.dump(out,open(PT+f'leaf_mech/{tag}.json','w'),indent=1)
    print(f'wrote leaf_mech/{tag}.json ({out["runtime_s"]:.0f}s)')

def baseline(key,tag,k=3):
    """Cross-leaf baseline (added 2026-08-20 from a wave-3
    reviewer catch: a14 dominates a15's input for unrelated leaves
    too, so a ratio against 1.0 overstates specificity). Returns
    the same component's top-writer ratios for k other leaves whose
    machinery includes this component."""
    import random
    st=cl.state()
    cand=[lf['tag'] for lf in st['leaves'] if lf['tag']!=tag and
          any(key in str(p) for p in lf['top_probes'])]
    random.Random(7).shuffle(cand)
    return cand[:k]

if __name__=='__main__':
    main(sys.argv[1])
    if '--baseline' in sys.argv:
        tag=sys.argv[1]
        seen=json.load(open(PT+f'leaf_mech/{tag}.json'))
        for key in seen['machinery']:
            peers=baseline(key,tag)
            print(f'-- baseline for {key}: peers {peers} '
                  f'(run leaf_input_decomp.py on each and compare '
                  f'{key}\'s top-writer ratio; a ratio that '
                  f'reproduces on unrelated leaves is a LAYER '
                  f'property, not this circuit\'s mechanism)')
