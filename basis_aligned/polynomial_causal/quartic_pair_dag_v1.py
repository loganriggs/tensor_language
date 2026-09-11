"""Minimum distinct quadratic pairs for fixed quartic monomials via binary MILP.
Restricted 2+2 multiplication schedules only; not general arithmetic-circuit minimization.
"""
import itertools
import numpy as np
import torch
from scipy.optimize import milp,Bounds,LinearConstraint


def options(term):
    i,j,k,l=term
    return sorted({tuple(sorted((tuple(sorted(a)),tuple(sorted(b))))) for a,b in [((i,j),(k,l)),((i,k),(j,l)),((i,l),(j,k))]})


def compile_pairs(terms,seconds=30):
    choices=[options(t) for t in terms.T.tolist()]
    pairs=sorted({p for cs in choices for c in cs for p in c});lookup={p:i for i,p in enumerate(pairs)}
    nx=len(pairs);flat=[(e,c) for e,cs in enumerate(choices) for c in cs];n=nx+len(flat)
    objective=np.r_[np.ones(nx),np.zeros(len(flat))];constraints=[];lower=[];upper=[]
    for e in range(len(choices)):
        row=np.zeros(n)
        for j,(edge,_) in enumerate(flat):
            if edge==e:row[nx+j]=1
        constraints.append(row);lower.append(1);upper.append(1)
    for j,(_,c) in enumerate(flat):
        for pair in set(c):
            row=np.zeros(n);row[nx+j]=1;row[lookup[pair]]=-1
            constraints.append(row);lower.append(-np.inf);upper.append(0)
    solved=milp(objective,integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(constraints),lower,upper),
        options={'time_limit':seconds,'mip_rel_gap':0.0})
    assert solved.x is not None and solved.status in (0,1),solved.message
    selected=[flat[j] for j in range(len(flat)) if solved.x[nx+j]>.5]
    assert len(selected)==len(choices) and len({e for e,c in selected})==len(choices)
    used=sorted({pair for _,c in selected for pair in c});ids={pair:i for i,pair in enumerate(used)}
    selected.sort();children=torch.tensor([[ids[c[0]],ids[c[1]]] for _,c in selected]).T
    return dict(pairs=torch.tensor(used).T,children=children,optimal=solved.status==0,
        solver_status=int(solved.status),pair_count=len(used),quartic_count=len(choices),scalar_products=len(used)+len(choices),
        independent_scalar_products=3*len(choices),dual_bound=float(solved.mip_dual_bound),mip_gap=float(solved.mip_gap))


def execute(dag,reads,multiplicity_root,writer,zero_nodes=(),zero_edges=()):
    pairs=dag['pairs'].to(reads.device);children=dag['children'].to(reads.device)
    values=reads[...,pairs[0]]*reads[...,pairs[1]]
    if zero_nodes:
        values=values.clone();values[...,list(zero_nodes)]=0
    terms=values[...,children[0]]*values[...,children[1]]*multiplicity_root
    if zero_edges:
        terms=terms.clone();terms[...,list(zero_edges)]=0
    return terms@writer.T
