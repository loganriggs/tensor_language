"""Exact completion of the product in a lean conditional CP DAG."""
from lean_conditional_cp import MATCHINGS

def compile_program(p):
    # Work in the caller's dtype. Do not mutate the source or its tensors.
    out={k:v for k,v in p.items() if k!='pair_weights'}
    a,b=p['pair_weights']
    out['pair_shifts']=[b,a]
    out['constant']=p['constant']-(a*b)@p['coefficients'].T
    return out

def evaluate(p,x):
    z=x@p['input_projection'].T
    fs=[z@f.T+b for f,b in zip(p['factors'],p['biases'])]
    ij,kl=MATCHINGS[p['matching']]
    u=fs[ij[0]]*fs[ij[1]]+p['pair_shifts'][0]
    v=fs[kl[0]]*fs[kl[1]]+p['pair_shifts'][1]
    return (u*v)@p['coefficients'].T+p['constant']
