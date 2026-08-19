"""CENSUS CONTAINMENT -- 381's fairness follow-up: the 5% same-data
identity used Jaccard, which punishes granularity mismatch (77 fine
vs 29 coarse leaves). Test CONTAINMENT instead: is each fresh-alone
leaf mostly inside SOME 424-tree leaf?
REGISTERED PREDICTIONS:
  (a) >=50% of fresh-alone leaves are >=0.7-contained in a 424 leaf
      -> the instability is largely COARSE-GRAINING (hierarchical
      consistency), not arbitrary re-carving;
  (b) if <50%, re-carving is real and the SOP identity revision
      stands at full strength;
  (c) containment distribution reported. CPU-only."""
import json, time, torch
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'census_containment_results.json'

def main():
    t0=time.time()
    OLDGRID=212*256
    fresh=torch.load(PT+'census_state_fresh212.pt',
                     map_location='cpu')['leaves']
    big=torch.load(PT+'census_state_424.pt',
                   map_location='cpu')['leaves']
    bigs=[(l['tag'],set(int(x)-OLDGRID for x in l['member']
                        if int(x)>=OLDGRID)) for l in big]
    rows=[]
    n70=0
    for lf in fresh:
        A=set(int(x) for x in lf['member'])
        bc,bt=0,None
        for tag,B in bigs:
            if not B: continue
            c=len(A&B)/max(len(A),1)
            if c>bc: bc,bt=c,tag
        rows.append({'fresh':lf['tag'],'in':bt,'containment':round(bc,3)})
        if bc>=0.7: n70+=1
    frac=n70/max(len(fresh),1)
    pa=frac>=0.5
    dist=sorted(r['containment'] for r in rows)
    out={'n_fresh':len(fresh),'frac_contained_70':round(frac,3),
         'containment_q':[dist[int(q*len(dist))] for q in
                          (0.1,0.5,0.9)],
         'pairs':rows,'pred_a':bool(pa),'pred_b':bool(not pa),
         'pred_c':True}
    print(f'contained>=0.7: {n70}/{len(fresh)} ({frac:.0%}) | '
          f'q10/50/90 {out["containment_q"]}')
    print(f"(a) >=50% contained: {'HELD' if pa else 'FAILED'}"
          f" -> {'coarse-graining' if pa else 're-carving is real'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
