"""OWNER-GRAPH FAMILY ANALYSIS (CPU+light GPU): relate the 147 structural
circuits to the ten function classes and to each other. For each certified
structural circuit: its majority function class (from oracle classify on
its discovery tokens) and its top-2 owners. Build (function x owner)
count table and report: (i) MULTIPLEXED OWNERS -- components serving >= 3
function classes as top owner; (ii) SPLIT FUNCTIONS -- classes served by
>= 3 distinct top-owner components (depth multiplexing); (iii) family
count: connected components of the circuit graph linking circuits sharing
BOTH majority class and >= 1 top owner.

REGISTERED PREDICTIONS: (a) >= 3 multiplexed owner components (attn2/5,
mlp17-class generalists); (b) subword and induction are split functions
(>= 3 owner components each); (c) the family count is far below 147
(>= 3x compression of the circuit list -- the amortization claim)."""
import json, sys, time, torch, collections
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from circuit_dictionary import classify, CLS
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'owner_family_graph_results.json'

def main():
    t0=time.time()
    reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
    clsA=classify(300,512).reshape(-1)
    disc=reg['disc_mask']; ad=reg['assign_disc']
    idxA=disc.nonzero().squeeze(1)
    rows=[]
    for c in reg['certified']:
        j=c['id']
        toks=idxA[(ad==j).nonzero().squeeze(1)]
        kk=clsA[toks]
        maj=int(torch.mode(kk).values)
        owners=[k for k,_ in c['top'][:2]]
        rows.append({'id':j,'class':CLS[maj],'owners':owners,
                     'n':len(toks)})
    fo=collections.defaultdict(collections.Counter)
    for r in rows:
        for o in r['owners']: fo[r['class']][o]+=1
    mux=collections.Counter()
    for cl,cnt in fo.items():
        for o in cnt: mux[o]+=1
    multiplexed=[o for o,k in mux.items() if k>=3]
    split=[cl for cl,cnt in fo.items() if len(cnt)>=3]
    # families: union-find on (class, shared owner)
    parent={r['id']:r['id'] for r in rows}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['class']==b['class'] and set(a['owners'])&set(b['owners']):
                ra,rb=find(a['id']),find(b['id'])
                if ra!=rb: parent[ra]=rb
    fams=len({find(r['id']) for r in rows})
    pa=len(multiplexed)>=3
    pb=('subword' in split) and ('ind' in split)
    pc=fams<=len(rows)//3
    out={'circuits':len(rows),'families':fams,
         'multiplexed_owners':multiplexed,'split_functions':split,
         'class_owner_table':{cl:dict(cnt.most_common(5))
                              for cl,cnt in fo.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'{len(rows)} circuits -> {fams} families')
    print(f'multiplexed owners (>=3 classes): {multiplexed}')
    print(f'split functions (>=3 owners): {split}')
    for cl,cnt in fo.items():
        print(f'  {cl:8s}: {dict(cnt.most_common(4))}')
    print(f"(a) >=3 multiplexed: {'HELD' if pa else 'FAILED'}")
    print(f"(b) subword+ind split: {'HELD' if pb else 'FAILED'}")
    print(f"(c) families <= 1/3 circuits: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
