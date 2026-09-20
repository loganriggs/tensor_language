"""Independent finite-face arithmetic and primitive replay audit, CPU only."""
import itertools,json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;ART=P.parent/'bilinear_quotient/circuits/followups'

def main():
    rng=np.random.default_rng(20260920)
    # Planted polynomial on Boolean switches: constant cancels in damage.
    linear=rng.normal(size=3);pair=rng.normal(size=3);triple=.17;base=3.4
    def y(bits):
        return base+linear@bits+sum(v*bits[i]*bits[j] for v,(i,j) in zip(pair,itertools.combinations(range(3),2)))+triple*np.prod(bits)
    e={mask:base-y(np.array([(mask>>i)&1 for i in range(3)])) for mask in range(8)}
    correction={ij:e[(1<<ij[0])+(1<<ij[1])]-e[1<<ij[0]]-e[1<<ij[1]] for ij in itertools.combinations(range(3),2)}
    assert np.max(np.abs(np.array(list(correction.values()))+pair))<1e-12
    assert abs(e[7]-sum(e[1<<i] for i in range(3))-sum(correction.values())+triple)<1e-12
    old=json.loads((ART/'semantic_port_census_v1_result.json').read_text());r=json.loads((ART/'semantic_port_pairs_v1_result.json').read_text())
    checks=[]
    for panel,roles in old['data'].items():
        for role,arms in roles.items():
            for arm,cells in arms.items():
                for family,c in cells.items():checks.append(float(np.max(np.abs(np.array(c['effects'])-r['data'][panel][role][arm][family]['effects']))))
    assert max(checks)==0
    results={}
    for name in ['semantic_port_pairs_v1','semantic_port_pairs_fresh_v1']:
        file=ART/(name+'_result.json')
        if not file.exists():continue
        result=json.loads(file.read_text());records=result['pair_records'];errors=[];den=[]
        for c in records:
            y=np.array(c['target']);pred=np.array(c['additive'])+sum(np.array(v) for v in c['pair_terms'].values())
            errors.append(float(np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-30)));den.append(float(np.linalg.norm(y)))
            assert abs(errors[-1]-c['all_pair_error'])<1e-12
            assert np.max(np.abs(y-pred-c['third_order']))<1e-12
        subsets=[]
        for size in [0,1,2,3]:
            for subset in itertools.combinations(['pair23','pair24','pair34'],size):
                es=[]
                for c in records:
                    y=np.array(c['target']);pred=np.array(c['additive'])+sum((np.array(c['pair_terms'][k]) for k in subset),start=np.zeros_like(y))
                    es.append(float(np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-30)))
                subsets.append(dict(pairs=list(subset),worst_relative_error=max(es),passes_10_percent=max(es)<=.1))
        results[name]=dict(max_all_pair_error=max(errors),min_target_norm=min(den),subset_audit=subsets,selection_scope='Opened subset census only; fresh registered subset remains pair23+pair34, irrespective of this audit.')
    out=dict(planted_sign_and_third_order_test=True,primitive_replay_max_abs=max(checks),results=results)
    (P/'SEMANTIC_PAIR_GRAPH_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
