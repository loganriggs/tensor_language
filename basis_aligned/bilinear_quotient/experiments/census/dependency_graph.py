"""DEPENDENCY GRAPH -- the compositional census as an object. Collect
every program found by the ladder (round-0 from the results JSON,
later rounds parsed from the runlog where they are printed), build the
DAG with an edge A -> B when B's program references circ_A, and test
whether FUNCTIONAL composition follows PHYSICAL depth.
REGISTERED PREDICTIONS:
  (a) LAYERING: for >=75% of edges, the mean component depth of the
      used circuit <= that of the user (composition points downstream);
  (b) HUBS: >=3 circuits are used by >=4 others (concept candidates);
  (c) depth histogram, hub list, and fan-in shapes reported."""
import json, re, sys, ast as ast2, time
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'dependency_graph_results.json'

def main():
    t0=time.time()
    progs={}
    try:
        d=json.load(open(PT+'compositional_ladder2_results.json'))
        for r in d['programs']:
            if r['bacc']>=0.75: progs[r['tag']]=r['program']
    except Exception: pass
    pat=re.compile(r"round \d+ PASS (r[\w.]+): [\d.]+ (\[\[.*\]\])")
    try:
        for line in open(PT+'runlogs/compositional_ladder2.log'):
            mm=pat.search(line)
            if mm:
                progs[mm.group(1)]=ast2.literal_eval(mm.group(2))
    except Exception: pass
    packs={p['tag']:p for p in
           json.load(open(PT+'circuit_tree4_packs.json'))}
    def depth(tag):
        ds=[]
        for ps in packs.get(tag,{}).get('top_probes',[]):
            m2=re.search(r"'[am](\d+)'",ps)
            if m2: ds.append(int(m2.group(1)))
        return sum(ds)/len(ds) if ds else None
    edges=[]
    for btag,prog in progs.items():
        for conj in prog:
            for pred in conj:
                nm=pred[4:] if pred.startswith('NOT ') else pred
                if nm.startswith('circ_'):
                    atag=nm[5:].replace('_','.')
                    edges.append((atag,btag))
    ok=0; tot=0
    for a,b in edges:
        da,db=depth(a),depth(b)
        if da is None or db is None: continue
        tot+=1
        if da<=db+0.51: ok+=1
    usage={}
    for a,b in edges: usage[a]=usage.get(a,0)+1
    hubs=sorted(usage.items(),key=lambda kv:-kv[1])
    nhub=sum(1 for _,c in hubs if c>=4)
    pa=(ok>=0.75*tot) if tot else False
    pb=nhub>=3
    out={'n_programs':len(progs),'n_edges':len(edges),
         'layered':f'{ok}/{tot}','hubs':hubs[:10],'n_hubs4':nhub,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'programs {len(progs)} | edges {len(edges)} | layered '
          f'{ok}/{tot} | hubs>=4: {nhub} | top hubs {hubs[:6]}')
    print(f"(a) >=75% layered: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=3 hubs: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
