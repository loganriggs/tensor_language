"""Three-multiplication conditional CP approximation with reused pair corrections.
Separate from the frozen full-conditional evaluator used by the queued run.
"""
import torch
MATCHINGS=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
PAIR_INDEX={(0,1):0,(0,2):1,(0,3):2,(1,2):3,(1,3):4,(2,3):5}

def compile_program(full,matching):
    a,b=MATCHINGS[matching]
    keys=['input_projection','factors','biases','coefficients','constant','writer']
    out={k:full[k] for k in keys if k in full}
    out['matching']=matching
    out['pair_weights']=full['pair_weights'][[PAIR_INDEX[a],PAIR_INDEX[b]]]
    return out

def evaluate(program,x):
    z=x@program['input_projection'].T
    a=[z@f.T+b for f,b in zip(program['factors'],program['biases'])]
    ij,kl=MATCHINGS[program['matching']]
    p=a[ij[0]]*a[ij[1]];q=a[kl[0]]*a[kl[1]]
    phi=p*q+p*program['pair_weights'][0]+q*program['pair_weights'][1]
    return phi@program['coefficients'].T+program['constant']

def price(program):
    d=program['input_projection'].shape[1];r=program['input_projection'].shape[0];k=program['factors'][0].shape[0];v=program['coefficients'].shape[0]
    return dict(variable_products=3*k,stored_floats=d*r+4*k*r+4*k+2*k+v*k+v+d*v,
                coefficient_multiplications=d*r+4*k*r+2*k+v*k,
                additions=r*(d-1)+4*k*r+2*k+v*k,
                matching_indices=1,convention='Floats include common1152x16writer; operationcounts exclude commonwriter equally. Two pair products reused by quartic and quadraticcorrections; exactconstantsretained.')
