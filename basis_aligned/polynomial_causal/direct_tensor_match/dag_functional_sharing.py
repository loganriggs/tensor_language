"""Bounded exact polynomial signatures propose functional intermediate sharing.
Oversized subexpressions become independent opaque symbols. Equality of the resulting
formal polynomials proves equality; unequal signatures do NOT prove inequivalence.
No probe-based approximate equality or full native coefficient-tensor expansion.
"""
from collections import defaultdict
from fractions import Fraction
from arithmetic_dag import score

def signatures(dag,outputs,max_terms=256,max_pairs=4096):
    values={};opaque=[]
    for node in dag.reachable(outputs):
        key=dag.nodes[node];op=key[0];p=defaultdict(Fraction);oversized=False
        if op=='input':p[((0,key[1]),)]=Fraction(1)
        elif op=='constant':p[()]=Fraction(1)
        elif op=='linear':
            for child,c in key[1]:
                for monomial,value in values[child].items():p[monomial]+=c*value
                p={m:v for m,v in p.items() if v}
                if len(p)>max_terms:oversized=True;break
                p=defaultdict(Fraction,p)
        else:
            a,b=key[1:]
            if len(values[a])*len(values[b])>max_pairs:oversized=True
            else:
                for ma,ca in values[a].items():
                    for mb,cb in values[b].items():p[tuple(sorted(ma+mb))]+=ca*cb
                    if len(p)>max_terms:oversized=True;break
        if oversized:
            p={((1,node),):Fraction(1)};opaque.append(node)
        values[node]={m:v for m,v in p.items() if v}
    return values,opaque

def best_equivalence_edit(dag,outputs,max_terms=256,max_pairs=4096,weights=(1.,.01,.001)):
    values,opaque=signatures(dag,outputs,max_terms,max_pairs);groups=defaultdict(list)
    for node,p in values.items():groups[tuple(sorted(p.items()))].append(node)
    baseline=score(dag.cost(outputs),weights);best=baseline;winner=None;proposals=[]
    for group in groups.values():
        if len(group)<2:continue
        for target in group:
            for replacement in group:
                if replacement==target or target in dag.reachable([replacement]):continue
                if dag.degree_limit is not None and dag.degrees[replacement]>dag.degrees[target]:continue
                updated=dag.replace(outputs,target,replacement);price=dag.cost(updated);value=score(price,weights)
                proposals.append(dict(target=target,replacement=replacement,cost=price,score=value))
                if value<best:best=value;winner=updated
    return winner,dict(opaque_nodes=len(opaque),equivalent_groups=sum(len(g)>1 for g in groups.values()),baseline_score=baseline,proposals=proposals,accepted=winner is not None)
